# Checklist for the Fine-Tuning + RAG Project

<!-- If you are done with a topic mark the checkboxes with an `x` (like `[x]`) -->

- [X] I read and understood the tasks.
- [X] I validated the policy sources, training examples, and evaluation cases and kept them separate.
- [X] I chunked, embedded, and indexed the policy sources in a real vector index, preserving source metadata.
- [X] I defined a top-k setting and an evidence threshold so unsupported questions abstain.
- [X] I ran a real local LoRA training loop and saved the adapter separately from the base model.
- [X] I built baseline RAG and hybrid RAG + LoRA answer paths that use the same retriever and prompt format.
- [X] Generated citations are validated against the retrieved evidence.
- [X] I evaluated both systems on the same held-out cases and reported retrieval and answer metrics separately.
- [X] I wrote `reports/evaluation_report.md` with failure analysis, adapter trade-offs, and a recommendation.
- [X] I added tests and they pass from the documented environment.
- [X] I documented how to run the project and answered the submission questions.

Bonus:

- [X] I included a base-model-only result to show the value of retrieval.
- [X] I added an LLM-based judge as a separate metric while keeping a no-key evaluation path.
