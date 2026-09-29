# Judge evaluation

The judge engine reads `inference/responses.jsonl` and evaluates samples that
contain both a base response and a fine-tuned response. The judge is called
through the same provider-independent `ModelClient` used by inference.

## Blind answer order

Before a request is created, the two responses are randomly assigned to labels
`A` and `B`. The configured `runtime.random_seed` makes this process
reproducible. The rendered prompt contains the question, anonymous labels, and
answer text—it does not contain base/fine-tuned model names or their mapping.

The private mapping is stored only in result metadata as `answer_order`, along
with `resolved_winner`. This allows later reporting to translate an `A` or `B`
decision back to `base` or `fine_tuned` without exposing provenance to the
judge.

## Prompt templates

Judge prompts are plain text files selected by
`evaluation.prompt_template`. The repository default is
`prompts/judge/default_pairwise.txt`. A template must contain:

- `{{question}}`
- `{{answer_a}}`
- `{{answer_b}}`
- `{{criteria}}`
- `{{output_schema}}`

Placeholders are replaced in one pass so placeholder-like text inside an
untrusted question or answer is not interpreted as template syntax.

## Structured output

The judge must return only a JSON object with this shape:

```json
{
  "winner": "A",
  "scores": {"A": 4, "B": 3},
  "criteria_scores": {
    "correctness": {"A": 4, "B": 3}
  },
  "reason": "Candidate A is more accurate."
}
```

`winner` must be `A`, `B`, or `tie`. Scores must be finite non-negative
numbers. Missing fields, additional fields, malformed JSON, invalid winners,
and malformed criterion scores are rejected and recorded as evaluation errors.

## Artifacts and failures

Results are written in inference order to:

```text
runs/<run_id>/evaluation/judge_results.jsonl
```

Judge API failures and invalid outputs do not abort other samples. They are
stored with `metadata.status: error`. Samples missing either inference response
are not sent to the judge and are stored with `metadata.status: skipped`.
Successful results can be reused when an existing run is resumed; failed and
skipped evaluations are attempted again when both responses are available.

Metrics and report generation consume this artifact through the separate
contract described in `docs/reporting.md`.

The structured parser accepts pure JSON, one fenced JSON object, or prose (also
including a leading `<think>...</think>` block) containing exactly one object
that validates against `JudgeDecision`. It scans bounded JSON values with
`JSONDecoder.raw_decode`; it does not use a greedy expression. No valid object,
malformed output, schema-invalid output, or more than one valid decision fails
closed as a per-sample judge error. The decision schema is never weakened and a
winner is never guessed.
