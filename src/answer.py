from __future__ import annotations

import argparse

from src.generation import Answerer
from src.retrieval import PolicyRetriever


def main() -> None:
    parser = argparse.ArgumentParser(description="Answer one question with retrieved evidence.")
    parser.add_argument("question")
    parser.add_argument(
        "--mode", choices=("baseline", "hybrid"), default="hybrid",
        help="hybrid reloads the saved LoRA adapter from ADAPTER_PATH",
    )
    args = parser.parse_args()

    retriever = PolicyRetriever()
    retriever.build()
    evidence = retriever.search(args.question)
    for item in evidence:
        print(f"evidence [{item.document_id}] score={item.score:.3f}")
    answer = Answerer(use_adapter=args.mode == "hybrid").generate(args.question, evidence)
    print(f"status: {answer.status}" + (f" ({answer.reason})" if answer.reason else ""))
    print(answer.text)


if __name__ == "__main__":
    main()
