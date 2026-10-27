"""Score samples with the official `ragas` library (0.2.x API), using Claude as the judge.

Install with: pip install 'rag-eval-harness[ragas]'
"""

from __future__ import annotations

import math

COLUMN_MAP = {
    "faithfulness": "faithfulness",
    "answer_relevancy": "answer_relevancy",
    "llm_context_precision_with_reference": "context_precision",
    "context_recall": "context_recall",
}


class RagasEvaluator:
    name = "ragas"

    def __init__(self, model: str, embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"):
        try:
            from langchain_anthropic import ChatAnthropic
            from langchain_huggingface import HuggingFaceEmbeddings
            from ragas.embeddings import LangchainEmbeddingsWrapper
            from ragas.llms import LangchainLLMWrapper
        except ImportError as e:
            raise ImportError("install ragas extras: pip install 'rag-eval-harness[ragas]'") from e

        self.llm = LangchainLLMWrapper(ChatAnthropic(model=model, temperature=0, max_tokens=1024))
        self.embeddings = LangchainEmbeddingsWrapper(
            HuggingFaceEmbeddings(model_name=embedding_model)
        )

    def score_batch(self, samples: list[dict]) -> list[dict]:
        """samples: dicts with question, reference, answer, contexts."""
        from ragas import EvaluationDataset, evaluate
        from ragas.metrics import (
            Faithfulness,
            LLMContextPrecisionWithReference,
            LLMContextRecall,
            ResponseRelevancy,
        )

        dataset = EvaluationDataset.from_list(
            [
                {
                    "user_input": s["question"],
                    "reference": s["reference"],
                    "response": s["answer"],
                    "retrieved_contexts": s["contexts"],
                }
                for s in samples
            ]
        )
        result = evaluate(
            dataset=dataset,
            metrics=[
                Faithfulness(),
                ResponseRelevancy(),
                LLMContextPrecisionWithReference(),
                LLMContextRecall(),
            ],
            llm=self.llm,
            embeddings=self.embeddings,
            show_progress=False,
        )
        df = result.to_pandas()

        rows = []
        for _, rec in df.iterrows():
            row = {}
            for col, name in COLUMN_MAP.items():
                value = rec.get(col)
                row[name] = None if value is None or math.isnan(value) else float(value)
            rows.append(row)
        return rows
