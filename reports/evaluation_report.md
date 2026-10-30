# Evaluation Report

Generated: 2026-10-29T17:56:28+00:00

## Configuration

- BASE_MODEL: `HuggingFaceTB/SmolLM2-135M-Instruct`
- EMBEDDING_MODEL: `sentence-transformers/all-MiniLM-L6-v2`
- TOP_K: `3`
- MIN_SCORE: `0.30`
- MAX_NEW_TOKENS: `80`
- TRAIN_EPOCHS: `8`
- LEARNING_RATE: `0.0002`
- RANDOM_SEED: `42`
- lora: `r=8, alpha=16, dropout=0.05, targets=q_proj,v_proj`

## Aggregate results

| Metric | baseline | hybrid |
|---|---|---|
| recall@k | 1.00 | 1.00 |
| mrr | 1.00 | 1.00 |
| term_coverage | 0.30 | 0.20 |
| citation_faithfulness | 0.71 | 0.83 |
| forbidden_rate | 0.00 | 0.00 |
| style_compliance | 0.93 | 0.93 |
| decision_accuracy | 0.86 | 0.86 |
| rejected_rate | 0.36 | 0.43 |
| mean_latency_s | 5.04 | 5.46 |

## Per-case evidence

### eval-refund-ticket
- should_abstain: False; retrieved: ['refunds', 'refunds', 'subscriptions']; top-1 raw score: 0.783
- **baseline**: status=rejected, decision_correct=True, citations=[], forbidden=[], terms=0.00
  - answer: 'No, the customer cannot receive a refund. The digital workshop starts in 12 hours, and the customer must provide the order ID before the workshop starts. The refund window is 30 days, and the customer must provide the order ID before the workshop starts. The refund window is 30 days, and the custome' (answer has no citation)
- **hybrid**: status=rejected, decision_correct=True, citations=[], forbidden=[], terms=0.00
  - answer: 'Yes, the customer can receive a refund. The refund window is 30 days. The customer can request a refund within 30 days of the workshop starting.' (answer has no citation)

### eval-duplicate-payment
- should_abstain: False; retrieved: ['refunds', 'refunds', 'response-style']; top-1 raw score: 0.547
- **baseline**: status=answered, decision_correct=True, citations=['response-style'], forbidden=[], terms=1.00
  - answer: '[response-style] The retrieved evidence must be collected from the invoice number and both payment references.'
- **hybrid**: status=answered, decision_correct=True, citations=['response-style'], forbidden=[], terms=0.00
  - answer: '[response-style] Gather all relevant supporting documents.'

### eval-authenticator
- should_abstain: False; retrieved: ['account-security', 'response-style', 'account-security']; top-1 raw score: 0.642
- **baseline**: status=answered, decision_correct=True, citations=['account-security'], forbidden=[], terms=0.00
  - answer: '[account-security] For a suspicious login, recommend resetting the password, signing out other sessions, and enabling multi-factor authentication.'
- **hybrid**: status=answered, decision_correct=True, citations=['account-security'], forbidden=[], terms=0.00
  - answer: '[account-security] For a suspicious login, recommend resetting the password, signing out other sessions, and enabling multi-factor authentication.'

### eval-passport
- should_abstain: False; retrieved: ['account-security', 'incidents', 'account-security']; top-1 raw score: 0.598
- **baseline**: status=answered, decision_correct=True, citations=['incidents'], forbidden=[], terms=0.00
  - answer: '[incidents] Diagnostic logs may include request IDs and timestamps, but must not include passwords, authentication tokens, one-time passcodes, or full payment-card details.'
- **hybrid**: status=rejected, decision_correct=True, citations=[], forbidden=[], terms=0.00
  - answer: 'No, the customer should not paste a passport scan into chat to recover an account.' (answer has no citation)

### eval-packed-address
- should_abstain: False; retrieved: ['shipping', 'shipping']; top-1 raw score: 0.892
- **baseline**: status=rejected, decision_correct=True, citations=[], forbidden=[], terms=0.00
  - answer: 'Yes, support can change a delivery address after the warehouse packs the order.' (answer has no citation)
- **hybrid**: status=rejected, decision_correct=True, citations=[], forbidden=[], terms=0.00
  - answer: 'Yes, support can change a delivery address after the warehouse packs the order. After packing, support cannot change the address. After dispatch, the customer may contact the carrier using the tracking number to request an available delivery option.' (answer has no citation)

### eval-standard-shipping
- should_abstain: False; retrieved: ['shipping', 'shipping']; top-1 raw score: 0.682
- **baseline**: status=rejected, decision_correct=True, citations=[], forbidden=[], terms=0.00
  - answer: 'The usual standard shipping time is three to five business days.' (answer has no citation)
- **hybrid**: status=rejected, decision_correct=True, citations=[], forbidden=[], terms=0.00
  - answer: 'Standard shipping usually takes three to five business days. Express shipping usually takes one to two business days.' (answer has no citation)

### eval-annual-refund
- should_abstain: False; retrieved: ['subscriptions', 'refunds', 'subscriptions']; top-1 raw score: 0.720
- **baseline**: status=rejected, decision_correct=True, citations=[], forbidden=[], terms=0.00
  - answer: 'No, an annual subscription cannot receive a partial refund after 45 days.' (answer has no citation)
- **hybrid**: status=rejected, decision_correct=True, citations=[], forbidden=[], terms=0.00
  - answer: 'No, an annual subscription cannot receive a partial refund after 45 days.' (answer has no citation)

### eval-pause
- should_abstain: False; retrieved: ['subscriptions', 'subscriptions', 'subscriptions']; top-1 raw score: 0.900
- **baseline**: status=answered, decision_correct=True, citations=['subscriptions'], forbidden=[], terms=1.00
  - answer: '[subscriptions] Monthly subscriptions can be paused once per calendar quarter for up to 30 days. The pause begins on the date selected by the customer.'
- **hybrid**: status=answered, decision_correct=True, citations=['subscriptions'], forbidden=[], terms=1.00
  - answer: '[subscriptions] Monthly subscriptions can be paused once per calendar quarter for up to 30 days. The pause begins on the date selected by the customer.\n[subscriptions] Monthly subscriptions can be cancelled before the next renewal date. Access continues until the end of the paid billing period.\n[sub'

### eval-outage-updates
- should_abstain: False; retrieved: ['incidents', 'subscriptions', 'subscriptions']; top-1 raw score: 0.781
- **baseline**: status=answered, decision_correct=True, citations=['incidents'], forbidden=[], terms=1.00
  - answer: '[incidents] During a confirmed outage, publish status updates at least every 30 minutes until recovery. Direct customers to the public status page for the latest confirmed information.'
- **hybrid**: status=answered, decision_correct=True, citations=['incidents'], forbidden=[], terms=1.00
  - answer: '[incidents] During a confirmed outage, publish status updates at least every 30 minutes until recovery. Direct customers to the public status page for the latest confirmed information.'

### eval-log-secrets
- should_abstain: False; retrieved: ['incidents', 'account-security', 'account-security']; top-1 raw score: 0.720
- **baseline**: status=rejected, decision_correct=True, citations=[], forbidden=[], terms=0.00
  - answer: 'Yes, diagnostic logs may contain authentication tokens.' (answer has no citation)
- **hybrid**: status=rejected, decision_correct=True, citations=[], forbidden=[], terms=0.00
  - answer: 'Yes, diagnostic logs may contain authentication tokens.' (answer has no citation)

### eval-bank-chargeback
- should_abstain: True; retrieved: ['refunds', 'refunds', 'refunds']; top-1 raw score: 0.631
- **baseline**: status=answered, decision_correct=False, citations=['refunds'], forbidden=[], terms=n/a
  - answer: '[refunds] # Refund and Duplicate Payment Policy'
- **hybrid**: status=answered, decision_correct=False, citations=['refunds'], forbidden=[], terms=n/a
  - answer: '[refunds] # Refund and Duplicate Payment Policy'

### eval-shipping-compensation
- should_abstain: True; retrieved: ['shipping', 'refunds', 'shipping']; top-1 raw score: 0.514
- **baseline**: status=answered, decision_correct=False, citations=['shipping'], forbidden=[], terms=n/a
  - answer: '[shipping] # Express Shipping'
- **hybrid**: status=answered, decision_correct=False, citations=['shipping'], forbidden=[], terms=n/a
  - answer: '[shipping] # Express Shipping'

### eval-cafeteria
- should_abstain: True; retrieved: []; top-1 raw score: 0.145
- **baseline**: status=abstained, decision_correct=True, citations=[], forbidden=[], terms=n/a
  - answer: 'I cannot answer this from the available Northstar policy evidence.' (no evidence above threshold)
- **hybrid**: status=abstained, decision_correct=True, citations=[], forbidden=[], terms=n/a
  - answer: 'I cannot answer this from the available Northstar policy evidence.' (no evidence above threshold)

### eval-bicycle
- should_abstain: True; retrieved: []; top-1 raw score: 0.205
- **baseline**: status=abstained, decision_correct=True, citations=[], forbidden=[], terms=n/a
  - answer: 'I cannot answer this from the available Northstar policy evidence.' (no evidence above threshold)
- **hybrid**: status=abstained, decision_correct=True, citations=[], forbidden=[], terms=n/a
  - answer: 'I cannot answer this from the available Northstar policy evidence.' (no evidence above threshold)

## Failure analysis

**baseline**: 9 problem case(s)
- `eval-refund-ticket`: rejected: answer has no citation
- `eval-authenticator`: missing required terms (coverage 0.00)
- `eval-passport`: missing required terms (coverage 0.00)
- `eval-packed-address`: rejected: answer has no citation
- `eval-standard-shipping`: rejected: answer has no citation

**hybrid**: 10 problem case(s)
- `eval-refund-ticket`: rejected: answer has no citation
- `eval-duplicate-payment`: missing required terms (coverage 0.00)
- `eval-authenticator`: missing required terms (coverage 0.00)
- `eval-passport`: rejected: answer has no citation
- `eval-packed-address`: rejected: answer has no citation


## Threshold evidence

- Supported top-1 scores: min 0.547, max 0.900. Unsupported top-1 scores: min 0.145, max 0.631.
- Configured MIN_SCORE: 0.30.
- The groups overlap, so retrieval score alone cannot fully separate them.

## Cost and latency

- baseline: mean latency 5.04s per question (CPU).
- hybrid: mean latency 5.46s per question (CPU).

## Limitations

- Only 14 cases; differences of one case move metrics by about 7 points.
- Some evaluation topics overlap with training topics, which can overstate the adapter benefit.
- Citation validity proves a citation was retrieved, not that the claim is semantically supported.
- Style is a proxy (at most 3 sentences and 80 words).

## Recommendation

Hybrid minus baseline: decision accuracy +0.00, citation faithfulness +0.12, term coverage -0.10, rejected rate +0.07. Results are mixed: the adapter improves some metrics and regresses others. Revise (more and better training data, validation split) before adopting it; the baseline remains the safer default.

## Next experiments

- Prompt tokens are already masked (answer-only loss); next, add a validation split and more abstention examples.
- Add a stricter generation prompt or constrained citation format so the 135M model always cites a retrieved source.
- Sweep MIN_SCORE and TOP_K using the score distribution above.
