# Copyright (c) Facebook, Inc. and its affiliates.

import os
from copy import deepcopy
from typing import Dict, Optional

import torch
import torch.nn.functional as F
from mmf.models.fashionvil.base import FashionViLBaseModel
from mmf.utils.configuration import get_mmf_cache_dir
from torch import Tensor
from transformers import BertTokenizer
from transformers.generation_beam_search import BeamSearchScorer
from transformers.modeling_bert import BertForPreTraining
from transformers.pytorch_utils import torch_int_div


class FashionViLForCaptioning(FashionViLBaseModel):
    def __init__(self, config):
        super().__init__(config)
        bert_masked_lm = BertForPreTraining.from_pretrained(
            self.config.bert_model_name,
            config=self.bert.config,
            cache_dir=os.path.join(get_mmf_cache_dir(), "distributed_{}".format(-1)),
        )
        self.head = deepcopy(bert_masked_lm.cls.predictions)
        self.bert._tie_or_clone_weights(
            self.head.decoder, self.bert.embeddings.word_embeddings
        )
        self.tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

    def flatten_for_bert(self, sample_list: Dict[str, Tensor]) -> Dict[str, Tensor]:
        to_be_flattened = ["input_ids", "segment_ids", "lm_label_ids"]
        if "input_ids_masked" in sample_list.keys():
            to_be_flattened.append("input_ids_masked")
        to_be_flattened_dim = ["image"]
        flattened = self.flatten(sample_list, to_be_flattened, to_be_flattened_dim)
        return flattened

    def add_post_flatten_params(
        self, sample_list: Dict[str, Tensor]
    ) -> Dict[str, Tensor]:
        b, l, _ = sample_list["image"].shape
        _, s = sample_list["input_mask"].shape
        device = sample_list["image"].device
        sample_list["visual_embeddings_type"] = torch.zeros(
            (b, l), device=device
        ).long()
        sample_list["attention_mask"] = self._get_causal_mask(b, s, l, device)
        return sample_list

    @staticmethod
    def _get_causal_mask(b, s, l, device):
        causal_mask = torch.tril(torch.ones((s, s), device=device).long())  # s, s
        causal_mask = causal_mask.unsqueeze(0).repeat(b, 1, 1)  # b, s, s
        attention_mask = torch.ones((b, s + l, s + l), device=device).long()
        attention_mask[:, :s, :s] = causal_mask
        attention_mask[:, s:, :s] = 0
        attention_mask = attention_mask.unsqueeze(1)
        return attention_mask

    @torch.no_grad()
    def _greedy_generation(
        self,
        sample_list: Dict[str, Tensor],
        mask_token_id: Optional[int] = 103,
        eos_token_id: Optional[int] = 102,
    ) -> Dict[str, Tensor]:
        b, l, _ = sample_list["image"].shape
        device = sample_list["image"].device

        input_ids = sample_list.input_ids[:, 0].unsqueeze(dim=-1)
        mask_ids = mask_token_id * torch.ones_like(input_ids)
        end_ids = eos_token_id * torch.ones_like(input_ids)

        seq_len = 1
        max_len = torch.max(torch.sum(sample_list.input_mask, dim=-1))
        while seq_len < max_len:
            next_embeddings, _, _ = self.bert.get_joint_embedding(
                torch.cat([input_ids, mask_ids], dim=-1),
                torch.zeros(b, seq_len + 1).long().to(device),
                sample_list["image"],
                sample_list["visual_embeddings_type"],
                self._get_causal_mask(b, seq_len + 1, l, device),
            )
            next_logits = self.head(next_embeddings[:, seq_len])
            # Greedy
            next_ids = torch.argmax(next_logits, dim=-1)
            input_ids = torch.cat([input_ids, next_ids[:, None]], dim=-1)
            seq_len = seq_len + 1
        input_ids = torch.cat([input_ids, end_ids], dim=-1)
        return input_ids

    @torch.no_grad()
    def _beam_generation(
        self,
        sample_list: Dict[str, Tensor],
        mask_token_id: Optional[int] = 103,
        eos_token_id: Optional[int] = 102,
        pad_token_id: Optional[int] = 0,
        num_beams: Optional[int] = 5,
    ) -> Dict[str, Tensor]:
        b, l, _ = sample_list["image"].shape
        device = sample_list["image"].device
        batch_beam_size = num_beams * b
        beam_scorer = BeamSearchScorer(
            batch_size=b,
            num_beams=num_beams,
            device=device,
        )

        input_ids = sample_list.input_ids[:, 0].unsqueeze(dim=-1)
        end_ids = eos_token_id * torch.ones_like(input_ids)
        input_ids = input_ids.repeat(num_beams, 1)
        mask_ids = mask_token_id * torch.ones_like(input_ids)

        beam_scores = torch.zeros((b, num_beams), dtype=torch.float, device=device)
        beam_scores[:, 1:] = -1e9
        beam_scores = beam_scores.view((batch_beam_size,))
        beam_indices = tuple(() for _ in range(batch_beam_size))

        seq_len = 1
        max_len = torch.max(torch.sum(sample_list.input_mask, dim=-1))
        image = (
            sample_list.image.unsqueeze(1).repeat(1, num_beams, 1, 1).flatten(end_dim=1)
        )
        visual_embeddings_type = (
            sample_list.visual_embeddings_type.unsqueeze(1)
            .repeat(1, num_beams, 1)
            .flatten(end_dim=1)
        )

        while True:
            next_embeddings, _, _ = self.bert.get_joint_embedding(
                torch.cat([input_ids, mask_ids], dim=-1),
                torch.zeros(batch_beam_size, seq_len + 1).long().to(device),
                image,
                visual_embeddings_type,
                self._get_causal_mask(batch_beam_size, seq_len + 1, l, device),
            )
            next_logits = self.head(next_embeddings[:, seq_len])
            next_scores = F.log_softmax(next_logits, dim=-1)

            # FIXME: logits processor here
            next_scores = next_scores + beam_scores[:, None].expand_as(next_scores)

            vocab_size = next_scores.shape[-1]
            next_scores = next_scores.view(b, num_beams * vocab_size)

            next_scores, next_tokens = torch.topk(
                next_scores, 2 * num_beams, dim=1, largest=True, sorted=True
            )

            next_indices = torch_int_div(next_tokens, vocab_size)
            next_tokens = next_tokens % vocab_size

            beam_outputs = beam_scorer.process(
                input_ids,
                next_scores,
                next_tokens,
                next_indices,
                pad_token_id=pad_token_id,
                eos_token_id=eos_token_id,
                beam_indices=beam_indices,
            )

            beam_scores = beam_outputs["next_beam_scores"]
            beam_next_tokens = beam_outputs["next_beam_tokens"]
            beam_idx = beam_outputs["next_beam_indices"]

            input_ids = torch.cat(
                [input_ids[beam_idx, :], beam_next_tokens.unsqueeze(-1)], dim=-1
            )
            beam_indices = tuple(
                beam_indices[beam_idx[i]] + (beam_idx[i],)
                for i in range(len(beam_indices))
            )
            seq_len = seq_len + 1

            if beam_scorer.is_done or seq_len >= max_len:
                break

        sequence_outputs = beam_scorer.finalize(
            input_ids,
            beam_scores,
            next_tokens,
            next_indices,
            pad_token_id=pad_token_id,
            eos_token_id=eos_token_id,
            max_length=max_len,
            beam_indices=beam_indices,
        )

        # in case the captions generation is not finished
        # eos token has the largest id
        sequence_outputs = torch.cat([sequence_outputs["sequences"], end_ids], dim=-1)
        return sequence_outputs

    def _postprocess_generation(
        self, targets, predictions, eos_token_id: Optional[int] = 102
    ):
        references = []
        captions = []
        for x, y in zip(targets, predictions):
            eos_x = (x == eos_token_id).nonzero(as_tuple=True)[0]
            eos_y = (y == eos_token_id).nonzero(as_tuple=True)[0]
            references.append([self.tokenizer.decode(x[1:eos_x])])
            captions.append(self.tokenizer.decode(y[1:eos_y]))
        output_dict = {"captions": captions, "references": references}
        print(captions[0], references[0])
        return output_dict

    def _forward(self, sample_list: Dict[str, Tensor]) -> Dict[str, Tensor]:
        if self.training:
            sequence_output, _, _ = self.bert.get_joint_embedding(
                sample_list["input_ids_masked"],
                sample_list["segment_ids"],
                sample_list["image"],
                sample_list["visual_embeddings_type"],
                sample_list["attention_mask"],
            )
            num_visual_tokens = sample_list["image"].shape[1]
            sequence_output = sequence_output[:, :-num_visual_tokens]
            logits = (
                self.head(sequence_output)
                .contiguous()
                .view(-1, self.bert.config.vocab_size)
            )
            labels = sample_list["lm_label_ids"].contiguous().view(-1)
            sample_list["targets"] = labels

            output_dict = {"scores": logits}
        else:
            if self.config.decoding_algorithm == "greedy":
                predictions = self._greedy_generation(sample_list)
            elif self.config.decoding_algorithm == "beam":
                predictions = self._beam_generation(sample_list)
            else:
                raise NotImplementedError
            output_dict = self._postprocess_generation(
                sample_list.input_ids, predictions
            )
        return output_dict
