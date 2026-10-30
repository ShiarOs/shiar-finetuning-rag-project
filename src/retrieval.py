from __future__ import annotations

import os
import re
import uuid
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv
from qdrant_client import QdrantClient, models
from sentence_transformers import SentenceTransformer


ROOT = Path(__file__).resolve().parents[1]
POLICY_DIR = ROOT / "data" / "knowledge_base"
COLLECTION = "northstar_policy"


@dataclass(frozen=True)
class Evidence:
    document_id: str
    title: str
    chunk_position: int
    source_text: str
    score: float

    @classmethod
    def from_payload(
        cls, payload: dict[str, object], score: float
    ) -> Evidence:
        document_id = payload.get("document_id")
        title = payload.get("title")
        chunk_position = payload.get("chunk_position")
        source_text = payload.get("source_text")
        if (
            not isinstance(document_id, str)
            or not isinstance(title, str)
            or not isinstance(chunk_position, int)
            or not isinstance(source_text, str)
        ):
            raise ValueError("Retrieved payload is missing valid policy metadata")
        return cls(document_id, title, chunk_position, source_text, score)


def split_text(text: str, max_words: int = 100, overlap: int = 20) -> list[str]:
    if max_words <= 0 or overlap < 0 or overlap >= max_words:
        raise ValueError("Require max_words > overlap >= 0")

    paragraphs = [part.strip() for part in re.split(r"\n\s*\n", text) if part.strip()]
    chunks: list[str] = []
    for paragraph in paragraphs:
        words = paragraph.split()
        start = 0
        while start < len(words):
            end = min(start + max_words, len(words))
            chunks.append(" ".join(words[start:end]))
            if end == len(words):
                break
            start = end - overlap
    return chunks


class PolicyRetriever:
    def __init__(self) -> None:
        load_dotenv(ROOT / ".env")
        model_name = os.getenv(
            "EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2"
        )
        revision = os.getenv("EMBEDDING_REVISION", "").strip()
        model_args = {"revision": revision} if revision else {}
        self.encoder = SentenceTransformer(model_name, **model_args)

        storage_path = os.getenv("VECTOR_STORAGE_PATH", "").strip()
        if storage_path:
            path = Path(storage_path)
            if not path.is_absolute():
                path = ROOT / path
            path.mkdir(parents=True, exist_ok=True)
            self.client = QdrantClient(path=str(path))
        else:
            self.client = QdrantClient(":memory:")

        self.top_k = int(os.getenv("TOP_K", "3"))
        self.min_score = float(os.getenv("MIN_SCORE", "0.30"))
        if self.top_k < 1:
            raise ValueError("TOP_K must be at least 1")
        if not -1.0 <= self.min_score <= 1.0:
            raise ValueError("MIN_SCORE must be between -1 and 1 for cosine scores")

    def build(self) -> None:
        paths = sorted(POLICY_DIR.glob("*.md"))
        if not paths:
            raise FileNotFoundError(f"No policy Markdown files in {POLICY_DIR}")

        chunks: list[dict[str, object]] = []
        for path in paths:
            text = path.read_text(encoding="utf-8")
            heading = re.search(r"^#\s+(.+)$", text, flags=re.MULTILINE)
            title = (
                heading.group(1).strip()
                if heading
                else path.stem.replace("-", " ").title()
            )
            for position, source_text in enumerate(split_text(text)):
                chunks.append(
                    {
                        "document_id": path.stem,
                        "title": title,
                        "chunk_position": position,
                        "source_text": source_text,
                    }
                )
        if not chunks:
            raise ValueError("Policy files produced no chunks")

        vectors = self.encoder.encode(
            [str(chunk["source_text"]) for chunk in chunks],
            normalize_embeddings=True,
        )
        vector_size = int(self.encoder.get_sentence_embedding_dimension())
        if self.client.collection_exists(COLLECTION):
            self.client.delete_collection(COLLECTION)
        self.client.create_collection(
            collection_name=COLLECTION,
            vectors_config=models.VectorParams(
                size=vector_size, distance=models.Distance.COSINE
            ),
        )
        points = []
        for chunk, vector in zip(chunks, vectors, strict=True):
            stable_key = (
                f"{chunk['document_id']}:{chunk['chunk_position']}:"
                f"{chunk['source_text']}"
            )
            point_id = str(uuid.uuid5(uuid.NAMESPACE_URL, stable_key))
            points.append(
                models.PointStruct(
                    id=point_id,
                    vector=vector.tolist(),
                    payload=chunk,
                )
            )
        self.client.upsert(collection_name=COLLECTION, points=points)

    def search(self, question: str) -> list[Evidence]:
        if not question.strip():
            raise ValueError("Question must not be empty")
        vector = self.encoder.encode(question, normalize_embeddings=True).tolist()
        result = self.client.query_points(
            collection_name=COLLECTION,
            query=vector,
            limit=self.top_k,
            with_payload=True,
        )
        evidence: list[Evidence] = []
        for point in result.points:
            payload = point.payload or {}
            score = float(point.score or 0.0)
            if score < self.min_score:
                continue
            evidence.append(Evidence.from_payload(payload, score))
        return evidence


if __name__ == "__main__":
    retriever = PolicyRetriever()
    retriever.build()
    question = "What is the usual standard shipping time?"
    for item in retriever.search(question):
        print(
            f"[{item.document_id}] score={item.score:.3f} "
            f"chunk={item.chunk_position}: {item.source_text}"
        )
