# Copyright (c) Facebook, Inc. and its affiliates.

import datasets
import evaluate
from nltk.parse.corenlp import CoreNLPParser

from .cider_scorer import CiderScorer

tokenizer = CoreNLPParser()


class Cider(evaluate.Metric):
    def _info(self):
        return evaluate.MetricInfo(
            description=" ",
            citation=" ",
            features=[
                datasets.Features(
                    {
                        "predictions": datasets.Value("string", id="sequence"),
                        "references": datasets.Sequence(
                            datasets.Value("string", id="sequence"), id="references"
                        ),
                    }
                ),
                datasets.Features(
                    {
                        "predictions": datasets.Value("string", id="sequence"),
                        "references": datasets.Value("string", id="sequence"),
                    }
                ),
            ],
        )

    def _compute(
        self,
        predictions,
        references,
        n=4,
        sigma=6.0,
        dfMode="corpus",
        tokenizer=lambda x: list(tokenizer.tokenize(x)),
    ):
        references = [[tokenizer(r) for r in ref] for ref in references]
        predictions = [tokenizer(p) for p in predictions]

        cider_scorer = CiderScorer(
            n=n,
            sigma=sigma,
        )
        for ref, p in zip(references, predictions):
            cider_scorer += (p, ref)

        score, _ = cider_scorer.compute_score(dfMode)

        return score
