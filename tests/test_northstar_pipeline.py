from __future__ import annotations

import json
import zlib
from pathlib import Path

import numpy as np
import pytest
from qdrant_client import QdrantClient

from src import data_validation, evaluation, retrieval
from src.generation import ABSTENTION_TEXT, Answer, classify_answer, extract_citations
from src.retrieval import Evidence, PolicyRetriever, split_text


class FakeEncoder:
    """Deterministic bag-of-words encoder so retrieval tests need no model download."""

    dim = 64

    def get_sentence_embedding_dimension(self) -> int:
        return self.dim

    def _one(self, text: str) -> np.ndarray:
        vec = np.zeros(self.dim)
        for word in text.lower().split():
            vec[zlib.crc32(word.strip(".,?!").encode()) % self.dim] += 1.0
        norm = np.linalg.norm(vec)
        return vec / norm if norm else vec

    def encode(self, texts, normalize_embeddings: bool = True):
        if isinstance(texts, str):
            return self._one(texts)
        return np.array([self._one(t) for t in texts])


@pytest.fixture
def retriever() -> PolicyRetriever:
    instance = object.__new__(PolicyRetriever)
    instance.encoder = FakeEncoder()
    instance.client = QdrantClient(":memory:")
    instance.top_k = 3
    instance.min_score = 0.2
    instance.build()
    return instance


def evidence(doc: str = "refunds") -> Evidence:
    return Evidence(doc, "Refunds", 0, "text", 0.9)


# --- data contracts ---------------------------------------------------------
def test_repository_data_is_valid(capsys) -> None:
    data_validation.validate()
    assert "Validated" in capsys.readouterr().out


def test_duplicate_training_ids_rejected(tmp_path: Path, monkeypatch) -> None:
    record = {
        "id": "x",
        "reference_ids": ["refunds"],
        "context": "c",
        "question": "q",
        "answer": "a",
    }
    path = tmp_path / "train.jsonl"
    path.write_text(json.dumps(record) + "\n" + json.dumps(record) + "\n", encoding="utf-8")
    monkeypatch.setattr(data_validation, "TRAINING_FILE", path)
    with pytest.raises(ValueError, match="duplicate"):
        data_validation.validate()


def test_unknown_source_rejected(tmp_path: Path, monkeypatch) -> None:
    case = [
        {
            "id": "e1",
            "question": "q",
            "expected_sources": ["nope"],
            "required_terms": [],
            "forbidden_terms": [],
            "should_abstain": False,
        }
    ]
    path = tmp_path / "cases.json"
    path.write_text(json.dumps(case), encoding="utf-8")
    monkeypatch.setattr(data_validation, "EVALUATION_FILE", path)
    with pytest.raises(ValueError, match="unknown sources"):
        data_validation.validate()


def test_train_and_eval_ids_are_disjoint() -> None:
    train = {r["id"] for r in data_validation.read_training(data_validation.TRAINING_FILE)}
    held_out = {r["id"] for r in data_validation.read_evaluation(data_validation.EVALUATION_FILE)}
    assert not train & held_out


# --- retrieval --------------------------------------------------------------
def test_split_text_overlap_and_validation() -> None:
    text = " ".join(f"w{i}" for i in range(30))
    chunks = split_text(text, max_words=10, overlap=2)
    assert chunks[0].split()[-2:] == chunks[1].split()[:2]
    with pytest.raises(ValueError):
        split_text(text, max_words=5, overlap=5)


def test_retrieval_returns_policy_metadata(retriever: PolicyRetriever) -> None:
    results = retriever.search("How long does standard shipping take?")
    assert results
    top = results[0]
    assert top.document_id in {p.stem for p in retrieval.POLICY_DIR.glob("*.md")}
    assert top.title and top.source_text and top.chunk_position >= 0
    assert [r.score for r in results] == sorted((r.score for r in results), reverse=True)


def test_weak_match_returns_no_evidence(retriever: PolicyRetriever) -> None:
    retriever.min_score = 0.99
    assert retriever.search("Is there a vegetarian option in the cafeteria?") == []


def test_payload_without_metadata_is_rejected() -> None:
    with pytest.raises(ValueError):
        Evidence.from_payload({"document_id": "refunds"}, 0.5)


# --- generation / citations -------------------------------------------------
def test_extract_citations_deduplicates() -> None:
    assert extract_citations("A [refunds] and B [shipping] and C [refunds].") == [
        "refunds",
        "shipping",
    ]


def test_valid_citation_is_answered() -> None:
    result = classify_answer("Refunds take 5 days [refunds].", [evidence()])
    assert result.status == "answered" and result.citations == ["refunds"]


def test_unretrieved_citation_is_rejected() -> None:
    assert classify_answer("Ships fast [shipping].", [evidence()]).status == "rejected"


def test_uncited_answer_is_rejected() -> None:
    assert classify_answer("Refunds take 5 days.", [evidence()]).status == "rejected"


def test_no_evidence_abstains_without_generation() -> None:
    result = classify_answer("anything [refunds]", [])
    assert result.status == "abstained" and result.text == ABSTENTION_TEXT


def test_model_abstention_is_recognised() -> None:
    result = classify_answer("The evidence is insufficient to answer.", [evidence()])
    assert result.status == "abstained"


# --- evaluation / report ----------------------------------------------------
def test_metrics() -> None:
    assert evaluation.recall_at_k(["a", "b"], ["b", "c"]) == 0.5
    assert evaluation.reciprocal_rank(["a"], ["x", "a"]) == 0.5
    assert evaluation.recall_at_k([], ["a"]) is None
    assert evaluation.term_coverage("Five business days", ["five", "days"]) == 1.0
    assert evaluation.forbidden_hits("free forever", ["forever"]) == ["forever"]
    assert evaluation.style_compliant("Short answer [refunds].")
    assert not evaluation.style_compliant("One. Two. Three. Four.")


def test_score_case_decision_logic() -> None:
    case = {
        "id": "c1",
        "question": "q",
        "expected_sources": [],
        "required_terms": [],
        "forbidden_terms": [],
        "should_abstain": True,
    }
    row = evaluation.score_case(case, [], Answer("abstained", ABSTENTION_TEXT), 0.1)
    assert row["decision_correct"] is True


def test_write_report_contains_required_sections(tmp_path: Path) -> None:
    case = {
        "id": "c1",
        "question": "q",
        "expected_sources": ["refunds"],
        "required_terms": ["days"],
        "forbidden_terms": [],
        "should_abstain": False,
    }
    ok = evaluation.score_case(
        case, [evidence()], Answer("answered", "Five days [refunds].", ["refunds"]), 0.2
    )
    bad = dict(ok, status="rejected", decision_correct=False, reason="no citation")
    unsupported = dict(ok, id="c2", should_abstain=True, retrieved=[])
    results = {"baseline": [bad, unsupported], "hybrid": [ok, unsupported]}
    path = evaluation.write_report(
        results, {"c1": 0.8, "c2": 0.1}, {"MIN_SCORE": "0.30"}, tmp_path / "report.md"
    )
    text = path.read_text(encoding="utf-8")
    for heading in (
        "## Configuration",
        "## Aggregate results",
        "## Per-case evidence",
        "## Failure analysis",
        "## Threshold evidence",
        "## Cost and latency",
        "## Recommendation",
        "## Next experiments",
    ):
        assert heading in text
    assert "| decision_accuracy |" in text
