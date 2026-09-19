# OpenAI-compatible model client

The model layer has one provider-independent asynchronous interface:

```text
generate(model_config, messages) -> GenerationResult
```

It sends non-streaming requests only to:

```text
POST {base_url}/chat/completions
```

There is no provider field, provider discovery, vendor SDK, or backend-specific
branch. Any LiteLLM, vLLM, OpenAI, or internal gateway deployment must expose
this common API surface.

## Model configuration

Each of `base`, `fine_tuned`, and `judge` uses the same schema:

```yaml
models:
  base:
    name: base-model
    base_url: http://localhost:8001/v1
    api_key: ${BASE_MODEL_API_KEY}  # optional; null omits Authorization
    timeout_seconds: 60
    generation_parameters:
      temperature: 0
      max_tokens: 512
```

`timeout` is accepted as an alias for `timeout_seconds`. The earlier
`model_name` and `request_parameters` spellings remain accepted as migration
aliases. Configuration snapshots use the canonical names and redact API keys.

The client owns the `model`, `messages`, and `stream` fields, so they cannot be
overridden through generation parameters. Streaming and multiple completions
are outside the current contract.

## Request and response contracts

Requests require a non-empty message list containing at least one user message.
Messages support the common system, developer, user, assistant, and tool roles
with textual content.

Successful results contain:

- assistant text;
- returned or requested model name;
- finish reason;
- optional token usage;
- completion and request IDs when supplied;
- total latency and attempt count.

Malformed JSON, missing choices, empty assistant content, and invalid usage
objects produce a normalized response error rather than leaking raw payloads
into downstream code.

## Retries and errors

The client retries timeouts, transport failures, HTTP 408/409/429, and selected
5xx responses. Backoff is bounded exponential delay with jitter and honors a
numeric `Retry-After` header. Authentication failures and other permanent 4xx
responses are not retried.

Normalized errors distinguish request validation, connection failure, timeout,
authentication failure, API failure, and malformed responses. Logs record only
operational metadata such as model name, attempt, and status code. Prompts,
response bodies, authorization headers, and API keys are not logged.
