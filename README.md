# 🧠 Super Memory

**Self-Evolving Neuro-Symbolic Agent** — coupling an LLM translation interface with a persistent MeTTa (Hyperon) AtomSpace knowledge graph.

## Architecture

```
┌──────────────────────────────────────────────────────┐
│                   Streamlit Dashboard                │
│  ┌─────────────────┐    ┌──────────────────────────┐ │
│  │  Chat & Dialog   │    │  AtomSpace Viewer        │ │
│  │  Query Input     │    │  Live Diff Box           │ │
│  │  Agent Reasoning │    │  Persistence Status      │ │
│  └────────┬────────┘    └──────────┬───────────────┘ │
│           │                        │                  │
│  ┌────────▼────────────────────────▼───────────────┐ │
│  │          Neuro-Symbolic Pipeline                 │ │
│  │  ┌─────────────┐  ┌────────────┐  ┌───────────┐ │ │
│  │  │ LLM Client  │→ │ Translator │→ │  MeTTa    │ │ │
│  │  │ (OpenAI/    │  │ NL ↔ MeTTa │  │  Engine   │ │ │
│  │  │  Gemini)    │  │            │  │           │ │ │
│  │  └─────────────┘  └────────────┘  └─────┬─────┘ │ │
│  └─────────────────────────────────────────┤───────┘ │
│                                            │         │
│  ┌─────────────────────────────────────────▼───────┐ │
│  │          Persistence Layer (Omega)              │ │
│  │  knowledge_base.metta  │  audit_log.jsonl       │ │
│  └─────────────────────────────────────────────────┘ │
└──────────────────────────────────────────────────────┘
```

## Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. (Optional) Set up LLM — works without it via regex fallback
cp .env.example .env
# Edit .env with your API key

# 3. Run the lifecycle test
python test_lifecycle.py

# 4. Launch the dashboard
streamlit run app.py
```

## Project Structure

| File | Purpose |
|------|---------|
| `core/metta_engine.py` | MeTTa/Hyperon wrapper, S-expr validation, atom CRUD, queries |
| `core/persistence.py` | Disk serialization, audit log, state hashing, unified diffs |
| `core/translator.py` | LLM client, Pydantic schemas, NL↔MeTTa translation, uncertainty detection |
| `app.py` | Streamlit dual-pane dashboard |
| `test_lifecycle.py` | Automated 5-step end-to-end verification |

## LLM Configuration

Works with any OpenAI-compatible API. Set via environment variables:

| Variable | Default | Description |
|----------|---------|-------------|
| `OPENAI_API_KEY` | *(empty)* | Falls back to regex translator if unset |
| `OPENAI_BASE_URL` | `https://api.openai.com/v1` | Change for OpenRouter/Gemini |
| `OPENAI_MODEL` | `gpt-4o-mini` | Any compatible model |

## Lifecycle Test

The `test_lifecycle.py` script validates:

1. **Empty Init** — Agent starts with blank AtomSpace
2. **Uncertainty** — Query returns empty → honest "I don't know"
3. **Learning** — Teach fact → MeTTa injection + diff + disk persist
4. **Process Kill** — Destroy all in-memory state
5. **Reload & Verify** — Fresh session answers correctly from disk
