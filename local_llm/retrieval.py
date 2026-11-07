"""A tiny local TF-IDF retriever over the included programming Q&A dataset."""

from __future__ import annotations

import math
import re
from collections import Counter
from typing import Iterable


STOP_WORDS = {
    "a", "an", "and", "are", "as", "at", "be", "by", "can", "do", "does", "for",
    "from", "how", "i", "in", "is", "it", "me", "of", "on", "or", "the", "to",
    "what", "when", "which", "why", "with", "you", "your", "difference", "explain",
}


def terms(text: str) -> list[str]:
    return [word.lower() for word in re.findall(r"[a-zA-Z_][a-zA-Z_0-9]*", text) if word.lower() not in STOP_WORDS]


class LocalRetriever:
    def __init__(self, examples: list[dict[str, str]]) -> None:
        self.examples = examples
        self.documents = [terms(item["question"] + " " + item["answer"]) for item in examples]
        self.document_frequency: Counter[str] = Counter()
        for words in self.documents:
            self.document_frequency.update(set(words))
        self.total_docs = max(1, len(self.documents))

    def _vector(self, words: Iterable[str]) -> dict[str, float]:
        counts = Counter(words)
        vector: dict[str, float] = {}
        for word, count in counts.items():
            df = self.document_frequency.get(word, 0)
            if df == 0:
                continue
            idf = math.log(1 + (self.total_docs / (1 + df)))
            vector[word] = (1 + math.log(count)) * idf
        norm = math.sqrt(sum(value * value for value in vector.values())) or 1.0
        return {word: value / norm for word, value in vector.items()}

    def search(self, query: str, limit: int = 3) -> list[dict[str, object]]:
        query_vector = self._vector(terms(query))
        if not query_vector:
            return []
        results: list[dict[str, object]] = []
        for item, document in zip(self.examples, self.documents):
            doc_vector = self._vector(document)
            score = sum(weight * doc_vector.get(word, 0.0) for word, weight in query_vector.items())
            results.append({"question": item["question"], "answer": item["answer"], "score": score})
        results.sort(key=lambda result: float(result["score"]), reverse=True)
        return results[:limit]
