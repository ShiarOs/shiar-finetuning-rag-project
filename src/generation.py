from __future__ import annotations

import os
import re
from dataclasses import dataclass, field

import torch
from dotenv import load_dotenv
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.data_validation import ROOT
from src.retrieval import Evidence
from src.training import render_example

CITATION_PATTERN = re.compile(r"\[([^\[\]]+)\]")
ABSTENTION_TEXT = (
    "I cannot answer this from the available Northstar policy evidence."
)
ABSTENTION_MARKERS = (
    "cannot answer",
    "can't answer",
    "insufficient",
    "not enough evidence",
    "do not have enough",
    "no evidence",
)


@dataclass
class Answer:
    status: str  # answered | abstained | rejected
    text: str
    citations: list[str] = field(default_factory=list)
    reason: str = ""


def format_context(evidence: list[Evidence]) -> str:
    return "\n".join(f"[{item.document_id}] {item.source_text}" for item in evidence)


def build_prompt(question: str, evidence: list[Evidence]) -> str:
    return render_example(format_context(evidence), question, "")


def extract_citations(text: str) -> list[str]:
    seen: list[str] = []
    for match in CITATION_PATTERN.findall(text):
        cite = match.strip()
        if cite not in seen:
            seen.append(cite)
    return seen


def classify_answer(text: str, evidence: list[Evidence]) -> Answer:
    """Validate a generated answer against the retrieved evidence."""
    text = text.strip()
    if not evidence:
        return Answer("abstained", ABSTENTION_TEXT, reason="no evidence above threshold")
    citations = extract_citations(text)
    allowed = {item.document_id for item in evidence}
    invalid = [cite for cite in citations if cite not in allowed]
    if invalid:
        return Answer(
            "rejected", text, citations, f"citations not in retrieved evidence: {invalid}"
        )
    if not citations:
        if any(marker in text.lower() for marker in ABSTENTION_MARKERS):
            return Answer("abstained", text, [], "model abstained")
        return Answer("rejected", text, [], "answer has no citation")
    return Answer("answered", text, citations)


class Answerer:
    """Generates answers with the base model or the base model plus the LoRA adapter."""

    def __init__(self, use_adapter: bool) -> None:
        load_dotenv(ROOT / ".env")
        name = os.getenv("BASE_MODEL", "HuggingFaceTB/SmolLM2-135M-Instruct")
        revision = os.getenv("BASE_MODEL_REVISION", "").strip()
        args = {"revision": revision} if revision else {}
        self.max_new_tokens = int(os.getenv("MAX_NEW_TOKENS", "80"))
        self.tokenizer = AutoTokenizer.from_pretrained(name, **args)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        model = AutoModelForCausalLM.from_pretrained(name, dtype=torch.float32, **args)
        if use_adapter:
            from peft import PeftModel

            adapter = ROOT / os.getenv("ADAPTER_PATH", "artifacts/support-lora-adapter")
            if not adapter.exists():
                raise FileNotFoundError(f"Adapter not found: {adapter}. Run training first.")
            model = PeftModel.from_pretrained(model, str(adapter))
        self.model = model.eval()
        self.max_length = int(os.getenv("MAX_SEQUENCE_LENGTH", "256"))

    def generate(self, question: str, evidence: list[Evidence]) -> Answer:
        if not evidence:
            return classify_answer("", evidence)
        prompt = build_prompt(question, evidence)
        # Keep the tail (question + "Answer:") when evidence is too long.
        budget = max(self.max_length * 2, 512)
        inputs = self.tokenizer(prompt, return_tensors="pt")
        if inputs["input_ids"].shape[1] > budget:
            inputs = {k: v[:, -budget:] for k, v in inputs.items()}
        with torch.no_grad():
            output = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=False,
                pad_token_id=self.tokenizer.pad_token_id,
            )
        new_tokens = output[0][inputs["input_ids"].shape[1]:]
        text = self.tokenizer.decode(new_tokens, skip_special_tokens=True)
        text = text.split("\n\n")[0].strip()
        return classify_answer(text, evidence)
