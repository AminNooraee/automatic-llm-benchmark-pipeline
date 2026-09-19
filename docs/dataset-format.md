# Dataset formats

The dataset layer is independent of model clients and converts every supported
source record into exactly two fields:

```json
{"id": "sample-000001", "prompt": "The prompt sent to both models"}
```

## Configuration

Automatic file and schema detection:

```yaml
dataset:
  path: datasets/benchmark.jsonl
  format: auto
```

Supported explicit file formats are `json`, `jsonl`, `csv`, and `parquet`. An
explicit format may be used when a file has a nonstandard extension.

Custom columns bypass schema auto-detection:

```yaml
dataset:
  path: datasets/custom.csv
  format: auto
  columns:
    prompt: question_text
    id: identifiers.record_id
```

Prompt and ID mappings support dotted paths for nested JSON-like records. The
ID mapping is optional. Without an ID column, deterministic IDs are generated
in the form `sample-000001`.

## File sources

- JSON: one object, an array of objects, or one top-level role/content message
  sequence.
- JSONL: one object per non-empty line.
- CSV: a header row followed by records.
- Parquet: records read in batches through PyArrow.

Local files are implemented through a source-adapter protocol. A future
Hugging Face source can implement the same raw-record stream contract without
changing schema detection, normalization, or downstream inference.

## Automatically detected schemas

- `prompt`: reads `prompt`.
- `question_answer`: reads `question`; `answer` is recognized but is not part
  of the normalized inference contract.
- `instruction_input`: reads `instruction` and appends non-empty `input` after
  a blank line.
- `openai_messages`: reads `messages` and selects the last user message.
- `chatml`: reads a structured `chat` list or serialized ChatML from `chatml`
  or marker-bearing `text`.
- `sharegpt`: reads `conversations` entries using `from` and `value`, selecting
  the last human/user message.
- `custom_columns`: reads the configured prompt mapping.

If more than one schema matches, ingestion stops and asks for an explicit
`dataset.columns.prompt` mapping. Detection samples up to 100 records and the
selected schema must remain valid for every subsequent record.

## Validation

Ingestion rejects malformed records, empty prompts, duplicate or empty IDs,
missing configured columns, unsupported formats, mixed schemas, and ambiguous
schemas. Errors identify the file, source row, detected fields, and possible
formats.

Successful ingestion writes:

```text
runs/<run-id>/dataset/
├── original_dataset.<format>
└── normalized_dataset.jsonl
```

The run manifest records source and schema formats, detection confidence,
record count, detected fields, source checksum, and artifact paths.
