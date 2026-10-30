from __future__ import annotations

import json
import os
import re
import statistics
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from src.data_validation import EVALUATION_FILE, ROOT, read_evaluation, validate
from src.generation import Answer, Answerer
from src.retrieval import Evidence, PolicyRetriever

REPORT_PATH = ROOT / "reports" / "evaluation_report.md"
MAX_STYLE_WORDS = 80
MAX_STYLE_SENTENCES = 3
MODES = ("baseline", "hybrid")


def recall_at_k(expected: list[str], retrieved: list[str]) -> float | None:
    if not expected:
        return None
    return sum(1 for doc in expected if doc in retrieved) / len(expected)


def reciprocal_rank(expected: list[str], retrieved: list[str]) -> float | None:
    if not expected:
        return None
    for rank, doc in enumerate(retrieved, start=1):
        if doc in expected:
            return 1.0 / rank
    return 0.0


def term_coverage(text: str, terms: list[str]) -> float | None:
    if not terms:
        return None
    lowered = text.lower()
    return sum(1 for term in terms if term.lower() in lowered) / len(terms)


def forbidden_hits(text: str, terms: list[str]) -> list[str]:
    lowered = text.lower()
    return [term for term in terms if term.lower() in lowered]


def style_compliant(text: str) -> bool:
    sentences = [s for s in re.split(r"(?<=[.!?])\s+", text.strip()) if s]
    return 0 < len(sentences) <= MAX_STYLE_SENTENCES and len(text.split()) <= MAX_STYLE_WORDS


def citation_faithful(answer: Answer, expected: list[str], retrieved: list[str]) -> bool | None:
    """Every citation must be retrieved evidence; supported cases must also cite expected sources."""
    if answer.status != "answered":
        return None
    if any(cite not in retrieved for cite in answer.citations):
        return False
    return not expected or any(cite in expected for cite in answer.citations)


def score_case(
    case: dict[str, Any], evidence: list[Evidence], answer: Answer, latency: float
) -> dict[str, Any]:
    retrieved = [item.document_id for item in evidence]
    expected = case["expected_sources"]
    answered_text = answer.text if answer.status == "answered" else ""
    return {
        "id": case["id"],
        "should_abstain": case["should_abstain"],
        "status": answer.status,
        "reason": answer.reason,
        "answer": answer.text,
        "citations": answer.citations,
        "retrieved": retrieved,
        "recall": recall_at_k(expected, retrieved),
        "mrr": reciprocal_rank(expected, retrieved),
        "term_coverage": term_coverage(answered_text, case["required_terms"]),
        "forbidden": forbidden_hits(answered_text, case["forbidden_terms"]),
        "faithful": citation_faithful(answer, expected, retrieved),
        "style": style_compliant(answer.text),
        "decision_correct": case["should_abstain"] == (answer.status == "abstained"),
        "latency": latency,
    }


def mean(values: list[float | None]) -> float:
    clean = [v for v in values if v is not None]
    return statistics.fmean(clean) if clean else float("nan")


def aggregate(rows: list[dict[str, Any]]) -> dict[str, float]:
    answered = [r for r in rows if r["status"] == "answered"]
    return {
        "recall@k": mean([r["recall"] for r in rows]),
        "mrr": mean([r["mrr"] for r in rows]),
        "term_coverage": mean([r["term_coverage"] for r in rows if not r["should_abstain"]]),
        "citation_faithfulness": mean([float(r["faithful"]) for r in answered]),
        "forbidden_rate": mean([float(bool(r["forbidden"])) for r in rows]),
        "style_compliance": mean([float(r["style"]) for r in rows]),
        "decision_accuracy": mean([float(r["decision_correct"]) for r in rows]),
        "rejected_rate": mean([float(r["status"] == "rejected") for r in rows]),
        "mean_latency_s": mean([r["latency"] for r in rows]),
    }


def retrieve_all(cases: list[dict[str, Any]]) -> tuple[list[list[Evidence]], list[float]]:
    """Retrieve once so both paths use identical evidence; also log raw top-1 scores."""
    retriever = PolicyRetriever()
    retriever.build()
    evidence_per_case: list[list[Evidence]] = []
    raw_top: list[float] = []
    threshold = retriever.min_score
    for case in cases:
        retriever.min_score = -1.0
        raw = retriever.search(case["question"])
        raw_top.append(raw[0].score if raw else 0.0)
        retriever.min_score = threshold
        evidence_per_case.append([e for e in raw if e.score >= threshold])
    return evidence_per_case, raw_top


def run_mode(
    mode: str, cases: list[dict[str, Any]], evidence_per_case: list[list[Evidence]]
) -> list[dict[str, Any]]:
    answerer = Answerer(use_adapter=(mode == "hybrid"))
    rows = []
    for case, evidence in zip(cases, evidence_per_case, strict=True):
        start = time.perf_counter()
        answer = answerer.generate(case["question"], evidence)
        rows.append(score_case(case, evidence, answer, time.perf_counter() - start))
    return rows


def fmt(value: float) -> str:
    return "n/a" if value != value else f"{value:.2f}"


def write_report(
    results: dict[str, list[dict[str, Any]]],
    raw_top: dict[str, float],
    config: dict[str, Any],
    path: Path = REPORT_PATH,
) -> Path:
    aggs = {mode: aggregate(rows) for mode, rows in results.items()}
    lines = ["# Evaluation Report", ""]
    lines += [f"Generated: {datetime.now(timezone.utc).isoformat(timespec='seconds')}", ""]
    lines += ["## Configuration", ""]
    lines += [f"- {key}: `{value}`" for key, value in config.items()]
    lines += ["", "## Aggregate results", ""]
    modes = list(results)
    lines.append("| Metric | " + " | ".join(modes) + " |")
    lines.append("|---|" + "---|" * len(modes))
    for metric in next(iter(aggs.values())):
        lines.append(f"| {metric} | " + " | ".join(fmt(aggs[m][metric]) for m in modes) + " |")

    lines += ["", "## Per-case evidence", ""]
    for index, case_id in enumerate(r["id"] for r in next(iter(results.values()))):
        first = next(iter(results.values()))[index]
        lines.append(f"### {case_id}")
        lines.append(
            f"- should_abstain: {first['should_abstain']}; retrieved: {first['retrieved']}; "
            f"top-1 raw score: {raw_top.get(case_id, 0.0):.3f}"
        )
        for mode in modes:
            row = results[mode][index]
            lines.append(
                f"- **{mode}**: status={row['status']}, decision_correct={row['decision_correct']}, "
                f"citations={row['citations']}, forbidden={row['forbidden']}, "
                f"terms={fmt(row['term_coverage'] if row['term_coverage'] is not None else float('nan'))}"
            )
            lines.append(f"  - answer: {row['answer'][:300]!r}" + (f" ({row['reason']})" if row["reason"] else ""))
        lines.append("")

    lines += ["## Failure analysis", ""]
    failures = []
    for mode in modes:
        for row in results[mode]:
            problems = []
            if not row["decision_correct"]:
                problems.append("wrong answer/abstain decision")
            if row["status"] == "rejected":
                problems.append(f"rejected: {row['reason']}")
            if row["forbidden"]:
                problems.append(f"forbidden terms {row['forbidden']}")
            if row["term_coverage"] is not None and row["term_coverage"] < 1.0 and row["status"] == "answered":
                problems.append(f"missing required terms (coverage {row['term_coverage']:.2f})")
            if problems:
                failures.append((mode, row["id"], "; ".join(problems)))
    if failures:
        for mode in modes:
            mode_failures = [f for f in failures if f[0] == mode]
            lines.append(f"**{mode}**: {len(mode_failures)} problem case(s)")
            for _, case_id, text in mode_failures[:5]:
                lines.append(f"- `{case_id}`: {text}")
            lines.append("")
    else:
        lines.append("- No failures were observed on this evaluation set.")

    supported = [s for (cid, s) in raw_top.items() if cid in {r["id"] for r in next(iter(results.values())) if not r["should_abstain"]}]
    unsupported = [s for (cid, s) in raw_top.items() if cid in {r["id"] for r in next(iter(results.values())) if r["should_abstain"]}]
    lines += ["", "## Threshold evidence", ""]
    if supported and unsupported:
        lines.append(
            f"- Supported top-1 scores: min {min(supported):.3f}, max {max(supported):.3f}. "
            f"Unsupported top-1 scores: min {min(unsupported):.3f}, max {max(unsupported):.3f}."
        )
        lines.append(f"- Configured MIN_SCORE: {config.get('MIN_SCORE')}.")
        if min(supported) > max(unsupported):
            lines.append("- The two groups are separable on this set; any threshold between them works.")
        else:
            lines.append("- The groups overlap, so retrieval score alone cannot fully separate them.")
    lines += ["", "## Cost and latency", ""]
    for mode in modes:
        lines.append(f"- {mode}: mean latency {aggs[mode]['mean_latency_s']:.2f}s per question (CPU).")
    lines += [
        "",
        "## Limitations",
        "",
        "- Only 14 cases; differences of one case move metrics by about 7 points.",
        "- Some evaluation topics overlap with training topics, which can overstate the adapter benefit.",
        "- Citation validity proves a citation was retrieved, not that the claim is semantically supported.",
        "- Style is a proxy (at most 3 sentences and 80 words).",
        "",
        "## Recommendation",
        "",
    ]
    base, hyb = aggs[modes[0]], aggs[modes[-1]]
    gain = hyb["decision_accuracy"] - base["decision_accuracy"]
    delta_faith = hyb["citation_faithfulness"] - base["citation_faithfulness"]
    delta_terms = hyb["term_coverage"] - base["term_coverage"]
    delta_rejected = hyb["rejected_rate"] - base["rejected_rate"]
    gains = [gain > 0.05, delta_faith > 0.05, delta_terms > 0.05, delta_rejected < -0.05]
    losses = [gain < -0.05, delta_faith < -0.05, delta_terms < -0.05, delta_rejected > 0.05]
    if any(gains) and not any(losses):
        verdict = "The adapter shows a gain without regressions; keep it as a candidate and confirm on a larger held-out set."
    elif any(gains):
        verdict = "Results are mixed: the adapter improves some metrics and regresses others. Revise (more and better training data, validation split) before adopting it; the baseline remains the safer default."
    else:
        verdict = "The adapter shows no measurable gain on this set, so the extra training and serving cost is not justified yet. Revise the training data or loss and re-evaluate."
    lines.append(
        f"Hybrid minus baseline: decision accuracy {gain:+.2f}, citation faithfulness "
        f"{delta_faith:+.2f}, term coverage {delta_terms:+.2f}, rejected rate {delta_rejected:+.2f}. "
        + verdict
    )
    lines += [
        "",
        "## Next experiments",
        "",
        "- Prompt tokens are already masked (answer-only loss); next, add a validation split and more abstention examples.",
        "- Add a stricter generation prompt or constrained citation format so the 135M model always cites a retrieved source.",
        "- Sweep MIN_SCORE and TOP_K using the score distribution above.",
        "",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def main() -> None:
    validate()
    cases = read_evaluation(EVALUATION_FILE)
    evidence_per_case, top_scores = retrieve_all(cases)
    raw_top = {case["id"]: score for case, score in zip(cases, top_scores, strict=True)}
    results = {mode: run_mode(mode, cases, evidence_per_case) for mode in MODES}
    config = {
        key: os.getenv(key, default)
        for key, default in {
            "BASE_MODEL": "HuggingFaceTB/SmolLM2-135M-Instruct",
            "EMBEDDING_MODEL": "sentence-transformers/all-MiniLM-L6-v2",
            "TOP_K": "3",
            "MIN_SCORE": "0.30",
            "MAX_NEW_TOKENS": "80",
            "TRAIN_EPOCHS": "8",
            "LEARNING_RATE": "0.0002",
            "RANDOM_SEED": "42",
        }.items()
    }
    config["lora"] = "r=8, alpha=16, dropout=0.05, targets=q_proj,v_proj"
    path = write_report(results, raw_top, config)
    (ROOT / "artifacts").mkdir(exist_ok=True)
    (ROOT / "artifacts" / "results.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"Wrote {path}")


if __name__ == "__main__":
    main()
