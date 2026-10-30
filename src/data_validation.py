from __future__ import annotations

import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
POLICY_DIR = ROOT / "data" / "knowledge_base"
TRAINING_FILE = ROOT / "data" / "training" / "instruction_examples.jsonl"
EVALUATION_FILE = ROOT / "data" / "evaluation" / "cases.json"


def read_training(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"{path}:{line_number}: invalid JSON") from error
            if not isinstance(record, dict):
                raise ValueError(f"{path}:{line_number}: expected an object")
            records.append(record)
    return records


def read_evaluation(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as file:
        records = json.load(file)
    if not isinstance(records, list) or not all(
        isinstance(record, dict) for record in records
    ):
        raise ValueError(f"{path}: expected a JSON array of objects")
    return records


def require_text(record: dict[str, Any], key: str, where: str) -> None:
    value = record.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{where}: {key} must be a non-empty string")


def require_string_list(record: dict[str, Any], key: str, where: str) -> None:
    value = record.get(key)
    if not isinstance(value, list) or not all(
        isinstance(item, str) and item.strip() for item in value
    ):
        raise ValueError(f"{where}: {key} must be a list of non-empty strings")


def validate() -> None:
    source_ids = {path.stem for path in POLICY_DIR.glob("*.md")}
    if not source_ids:
        raise ValueError(f"No Markdown policies found in {POLICY_DIR}")

    training = read_training(TRAINING_FILE)
    evaluation = read_evaluation(EVALUATION_FILE)

    training_ids: set[str] = set()
    for index, record in enumerate(training, start=1):
        where = f"training record {index}"
        for field in ("id", "context", "question", "answer"):
            require_text(record, field, where)
        require_string_list(record, "reference_ids", where)
        if not record["reference_ids"]:
            raise ValueError(f"{where}: reference_ids must not be empty")
        if record["id"] in training_ids:
            raise ValueError(f"{where}: duplicate id {record['id']!r}")
        training_ids.add(record["id"])
        unknown = set(record["reference_ids"]) - source_ids
        if unknown:
            raise ValueError(f"{where}: unknown sources {sorted(unknown)}")

    evaluation_ids: set[str] = set()
    for index, record in enumerate(evaluation, start=1):
        where = f"evaluation record {index}"
        require_text(record, "id", where)
        require_text(record, "question", where)
        require_string_list(record, "expected_sources", where)
        require_string_list(record, "required_terms", where)
        require_string_list(record, "forbidden_terms", where)
        if not isinstance(record.get("should_abstain"), bool):
            raise ValueError(f"{where}: should_abstain must be a boolean")
        if bool(record["should_abstain"]) != (not record["expected_sources"]):
            raise ValueError(
                f"{where}: abstention label and expected sources disagree"
            )
        if record["id"] in evaluation_ids:
            raise ValueError(f"{where}: duplicate id {record['id']!r}")
        evaluation_ids.add(record["id"])
        unknown = set(record["expected_sources"]) - source_ids
        if unknown:
            raise ValueError(f"{where}: unknown sources {sorted(unknown)}")

    overlap = training_ids & evaluation_ids
    if overlap:
        raise ValueError(f"Training and evaluation IDs overlap: {sorted(overlap)}")

    print(
        f"Validated {len(source_ids)} policies, "
        f"{len(training)} training records, and {len(evaluation)} evaluation cases."
    )


if __name__ == "__main__":
    validate()
