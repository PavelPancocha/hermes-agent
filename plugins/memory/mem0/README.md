# Mem0 Memory Provider

Server-side LLM fact extraction with semantic search and hybrid multi-signal retrieval via the Mem0 Platform v3 API.

## Requirements

- `pip install mem0ai`
- Mem0 API key from [app.mem0.ai](https://app.mem0.ai)

## Setup

```bash
hermes memory setup    # select "mem0"
```

Or manually:
```bash
hermes config set memory.provider mem0
echo "MEM0_API_KEY=your-key" >> ~/.hermes/.env
```

## Config

Behavioral settings live in `$HERMES_HOME/mem0.json` (set them via `hermes memory setup`). Only the secret `MEM0_API_KEY` belongs in `~/.hermes/.env`.

| Key | Default | Description |
|-----|---------|-------------|
| `mode` | `platform` | `platform` (Mem0 Cloud), `self_hosted_http` (self-hosted REST API), or `oss` (in-process self-hosted) |
| `host` | | Self-hosted Mem0 REST API base URL for `self_hosted_http` mode |
| `user_id` | `hermes-user` | User identifier on Mem0 |
| `agent_id` | `hermes` | Agent identifier |
| `rerank` | `true` | Rerank search results for relevance (platform mode only) |

## Self-Hosted HTTP Mode

Use this when you already run the Mem0 REST server and want Hermes to call its
`/memories` and `/search` endpoints directly.

```bash
hermes memory setup mem0 --mode self_hosted_http \
  --host https://mem0-api.example.com \
  --api-key m0sk-... \
  --user-id person:pavel \
  --agent-id hermes:pavel-cli
```

Secrets are stored in `$HERMES_HOME/.env`; `host`, `user_id`, and `agent_id`
are stored in `$HERMES_HOME/mem0.json`.

## OSS (Self-Hosted In-Process) Mode

Run Mem0 locally/in-process with your own LLM, embedder, and vector store. This
does not call an already-running Mem0 REST server.

### Interactive Setup

```bash
hermes memory setup
# Select "mem0" → "Open Source (self-hosted)"
# Follow prompts for LLM, embedder, and vector store
```

### Agent-Driven Setup (Flags)

```bash
hermes memory setup mem0 --mode oss \
  --oss-llm openai --oss-llm-key sk-... \
  --oss-vector qdrant
```

### Supported Providers

| Component | Providers |
|-----------|-----------|
| LLM | openai, ollama |
| Embedder | openai, ollama |
| Vector Store | qdrant (local/server), pgvector |

### Flags Reference

| Flag | Description |
|------|-------------|
| `--mode` | `platform`, `self_hosted_http`, or `oss` |
| `--host` / `--api-url` | Self-hosted Mem0 REST API base URL |
| `--api-key` | Mem0 API key for `platform` or `self_hosted_http` |
| `--oss-llm` | LLM provider (default: openai) |
| `--oss-llm-key` | LLM API key |
| `--oss-embedder` | Embedder provider (default: openai) |
| `--oss-vector` | Vector store (default: qdrant) |
| `--oss-vector-path` | Qdrant local path |
| `--user-id` | User identifier |
| `--agent-id` | Agent identifier |

## Switching Modes

### Platform to OSS

```bash
hermes memory setup mem0 --mode oss --oss-llm-key sk-...
```

Or edit `$HERMES_HOME/mem0.json` directly:
```json
{
  "mode": "oss",
  "oss": {
    "llm": {"provider": "openai", "config": {"model": "gpt-5-mini"}},
    "embedder": {"provider": "openai", "config": {"model": "text-embedding-3-small"}},
    "vector_store": {"provider": "qdrant", "config": {"path": "~/.hermes/mem0_qdrant"}}
  }
}
```

### OSS to Platform

```bash
hermes memory setup mem0 --mode platform --api-key sk-...
```

### Dry Run (preview without writing)

```bash
hermes memory setup mem0 --mode oss --oss-llm-key sk-... --dry-run
```

## Tools

| Tool | Description |
|------|-------------|
| `mem0_list` | List all stored memories (paginated) |
| `mem0_search` | Semantic search by meaning |
| `mem0_add` | Store a fact verbatim (no LLM extraction) |
| `mem0_update` | Update a memory's text by ID |
| `mem0_delete` | Delete a memory by ID |

## Troubleshooting

### "Mem0 temporarily unavailable"

Circuit breaker tripped after 5 consecutive failures. Resets after 2 minutes.

- **Platform mode**: Check API key and internet connectivity.
- **OSS mode**: Check that your vector store (qdrant/pgvector) is running.

### OSS: Qdrant connection refused

```bash
# If using local Qdrant, check the storage path is writable:
ls -la ~/.hermes/mem0_qdrant

# If using Qdrant server, check it's reachable:
curl http://localhost:6333/healthz
```

### OSS: PGVector connection refused

```bash
# Verify PostgreSQL is running and accepting connections:
pg_isready -h localhost -p 5432
```

### OSS: Ollama not reachable

```bash
# Check Ollama is running:
curl http://localhost:11434/api/tags
```

### Memories not appearing

- `mem0_add` stores verbatim (no extraction). Use `sync_turn` for LLM extraction.
- Search uses semantic matching — try broader queries.
- Check `user_id` matches between sessions (`$HERMES_HOME/mem0.json`).
