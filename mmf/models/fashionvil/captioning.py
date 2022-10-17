# Copyright (c) Facebook, Inc. and its affiliates.

import os
from copy import deepcopy
from typing import Dict, Optional

import torch
from mmf.models.fashionvil.base import FashionViLBaseModel
from mmf.utils.configuration import get_mmf_cache_dir
from torch import Tensor
from transformers import BertTokenizer
from transformers.modeling_bert import BertForPreTraining


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
            # elif self.config.decoding_algorithm == "beam":
            #     predictions = self._beam_generation(sample_list)
            else:
                raise NotImplementedError
            output_dict = self._postprocess_generation(
                sample_list.input_ids, predictions
            )
        return output_dict
