"""
app.py
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Super Memory · Neuro-Symbolic Agent Dashboard

Split-screen Streamlit UI:
  Left  → Chat & Interaction (user dialog, query inputs, agent reasoning)
  Right → Auditable State (live AtomSpace, diffs, persistence status)

Launch:  streamlit run app.py
"""

from __future__ import annotations

import logging
import streamlit as st
from datetime import datetime
import os
from dotenv import load_dotenv
load_dotenv()
from dotenv import load_dotenv

# Load .env file for OpenRouter keys
load_dotenv()

from core.metta_engine import MeTTaEngine
from core.persistence import PersistenceManager, generate_diff, compute_state_hash
from core.translator import (
    LLMTranslator,
    IntentType,
    MeTTaTranslation,
    UncertaintyPayload,
    AnswerPayload,
)

# ── logging ──
logging.basicConfig(level=logging.INFO, format="%(name)s · %(message)s")
logger = logging.getLogger("super_memory")


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Page Config
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

st.set_page_config(
    page_title="Super Memory · Neuro-Symbolic Agent",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ── custom CSS ──
st.markdown("""
<style>
    /* ── Dark theme overrides ── */
    .stApp {
        background: linear-gradient(135deg, #0a0a0f 0%, #111128 50%, #0d0d1a 100%);
    }

    /* ── Diff box styling ── */
    .diff-box {
        background: #0d1117;
        border: 1px solid #30363d;
        border-radius: 8px;
        padding: 12px;
        font-family: 'JetBrains Mono', 'Fira Code', monospace;
        font-size: 0.82em;
        line-height: 1.5;
        overflow-x: auto;
        white-space: pre;
        color: #c9d1d9;
    }
    .diff-box .diff-add { color: #3fb950; }
    .diff-box .diff-del { color: #f85149; }
    .diff-box .diff-hdr { color: #58a6ff; font-weight: bold; }

    /* ── State indicator ── */
    .status-pill {
        display: inline-block;
        padding: 4px 12px;
        border-radius: 20px;
        font-size: 0.8em;
        font-weight: 600;
    }
    .status-loaded {
        background: rgba(63, 185, 80, 0.15);
        color: #3fb950;
        border: 1px solid rgba(63, 185, 80, 0.3);
    }
    .status-fresh {
        background: rgba(88, 166, 255, 0.15);
        color: #58a6ff;
        border: 1px solid rgba(88, 166, 255, 0.3);
    }

    /* ── AtomSpace viewer ── */
    .atom-viewer {
        background: #0d1117;
        border: 1px solid #30363d;
        border-radius: 8px;
        padding: 12px;
        font-family: 'JetBrains Mono', 'Fira Code', monospace;
        font-size: 0.85em;
        color: #c9d1d9;
        max-height: 300px;
        overflow-y: auto;
    }
    .atom-line { color: #e6db74; }
    .atom-empty { color: #6e7681; font-style: italic; }

    /* ── Chat bubbles ── */
    div[data-testid="stChatMessage"] {
        border-radius: 12px;
    }

    /* ── Section headers ── */
    .section-header {
        color: #58a6ff;
        font-size: 0.9em;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        margin-bottom: 8px;
        padding-bottom: 4px;
        border-bottom: 1px solid #21262d;
    }
</style>
""", unsafe_allow_html=True)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Session State Initialization
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def init_session():
    """Initialize all session state objects on first load."""
    if "engine" not in st.session_state:
        st.session_state.engine = MeTTaEngine()
        # Load the backward chainer (inference engine) on first init
        ontology_path = os.path.join(os.path.dirname(__file__), "ontology.metta")
        if os.path.exists(ontology_path):
            with open(ontology_path, "r", encoding="utf-8") as f:
                st.session_state.engine.load_program(f.read())
            logger.info("Backward chainer loaded from ontology.metta")

    if "persistence" not in st.session_state:
        st.session_state.persistence = PersistenceManager()

    if "catalogue" not in st.session_state:
        from core.catalogue import EntityCatalogue
        st.session_state.catalogue = EntityCatalogue()

    if "translator" not in st.session_state:
        st.session_state.translator = LLMTranslator()
        
    if "pending_request" not in st.session_state:
        st.session_state.pending_request = None

    if "messages" not in st.session_state:
        st.session_state.messages = []

    if "diffs" not in st.session_state:
        st.session_state.diffs = []

    if "loaded_from_disk" not in st.session_state:
        # Try to reload persisted state
        pm: PersistenceManager = st.session_state.persistence
        engine: MeTTaEngine = st.session_state.engine

        if pm.has_persisted_state():
            loaded = pm.load_state(engine)
            st.session_state.loaded_from_disk = loaded
            if loaded:
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": (
                        f"🔄 **Session restored** from disk.\n\n"
                        f"Loaded {len(engine.get_all_atoms())} atoms from "
                        f"a previous session.\n\n"
                        f"State hash: `{pm.current_hash}`"
                    ),
                })
        else:
            st.session_state.loaded_from_disk = False
            st.session_state.messages.append({
                "role": "assistant",
                "content": (
                    "🧠 **Super Memory initialised** with an empty knowledge base.\n\n"
                    "Teach me facts, ask me questions, and watch the AtomSpace evolve in real time."
                ),
            })

init_session()


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Helper: Format diff as HTML
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def diff_to_html(diff_text: str) -> str:
    """Convert unified diff text to syntax-highlighted HTML."""
    if diff_text == "(no changes)":
        return '<div class="diff-box"><span class="atom-empty">No changes</span></div>'

    lines = []
    for line in diff_text.split("\n"):
        if line.startswith("+++") or line.startswith("---"):
            lines.append(f'<span class="diff-hdr">{_esc(line)}</span>')
        elif line.startswith("@@"):
            lines.append(f'<span class="diff-hdr">{_esc(line)}</span>')
        elif line.startswith("+"):
            lines.append(f'<span class="diff-add">{_esc(line)}</span>')
        elif line.startswith("-"):
            lines.append(f'<span class="diff-del">{_esc(line)}</span>')
        else:
            lines.append(_esc(line))

    return f'<div class="diff-box">{"<br>".join(lines)}</div>'


def _esc(s: str) -> str:
    """HTML-escape a string."""
    return s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Core: Process User Input
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def process_input(user_input: str):
    """
    Main pipeline:
      1. Translate NL → MeTTa
      2. Execute against AtomSpace
      3. Generate verbal response
      4. Record audit entry + diff
    """
    engine: MeTTaEngine = st.session_state.engine
    pm: PersistenceManager = st.session_state.persistence
    translator: LLMTranslator = st.session_state.translator
    catalogue = st.session_state.catalogue
    catalogue.sync(engine.get_all_atoms())

    # Step 1: Translate
    translation = translator.translate(user_input, catalogue, st.session_state.pending_request)

    if hasattr(translation, "missing_concepts"):
        # UncertaintyPayload
        st.session_state.pending_request = None
        return (
            f"❓ {translation.message}\n\n"
            f"Missing concepts: `{', '.join(translation.missing_concepts)}`\n\n"
            f"💡 {translation.suggested_input}"
        )

    if translation.intent == IntentType.CLARIFICATION:
        st.session_state.pending_request = translation
        cands_str = "\n".join(f"- {c}" for c in translation.candidates)
        return (
            f"🤔 **Clarification needed:** {translation.message}\n\n"
            f"Multiple possibilities found for your request:\n{cands_str}\n\n"
            f"*(Type your selection below, e.g. 'the first one' or the exact name)*"
        )
        
    st.session_state.pending_request = None

    if translation.intent == IntentType.ASSERTION:
        # ── Learn a new fact ──
        before_snapshot = engine.get_state_snapshot()
        engine.add_atom(translation.metta_expression)
        catalogue.sync(engine.get_all_atoms())

        # Record & persist
        entry = pm.record_modification(
            engine, user_input, translation.metta_expression, "add"
        )

        st.session_state.diffs.insert(0, {
            "timestamp": entry.timestamp,
            "diff": entry.diff,
            "rule": translation.metta_expression,
            "operation": "add",
        })

        return (
            f"✅ **Learned new fact**\n\n"
            f"🛡️ **Jev Intent:** `{translation.intent.value}`\n\n"
            f"**MeTTa atom:** `{translation.metta_expression}`\n\n"
            f"*{translation.verbal_reasoning}*\n\n"
            f"State hash: `{pm.current_hash}`"
        )

    elif translation.intent == IntentType.QUERY:
        # ── Query the knowledge base ──
        # Try direct match first
        result = engine.query(
            translation.metta_expression,
            translation.query_template,
        )

        # If direct match fails, try backward chaining for deductive queries
        if result.is_empty and translation.metta_expression:
            bc_expr = translation.metta_expression
            # Strip outer match wrapper if present, extract the pattern
            import re as _re
            m = _re.match(r"\(match\s+&self\s+(.+?)\s+\$\w+\)$", bc_expr)
            if m:
                bc_expr = m.group(1)
            logger.info("Direct match empty, trying backward chaining: %s", bc_expr)
            try:
                result = engine.backward_chain(bc_expr)
            except Exception as e:
                logger.warning("Backward chaining failed: %s", e)

        if result.is_empty:
            uncertainty = translator.generate_answer(user_input, [])
            return (
                f"❓ **No results found**\n\n"
                f"🛡️ **Jev Intent:** `{translation.intent.value}`\n\n"
                f"*{uncertainty.message}*\n\n"
                f"**Missing concepts:** {', '.join(uncertainty.missing_concepts) if uncertainty.missing_concepts else 'unknown'}\n\n"
                f"💡 {uncertainty.suggested_input}"
            )
        else:
            answer = translator.generate_answer(
                user_input, result.raw, source_atoms=result.raw
            )
            return (
                f"📍 **Answer:** {answer.answer}\n\n"
                f"🛡️ **Jev Intent:** `{translation.intent.value}`\n\n"
                f"**Source atoms:** `{', '.join(answer.source_atoms)}`\n\n"
                f"**Query pattern:** `{translation.metta_expression}`"
            )

    elif translation.intent == IntentType.RETRACTION:
        # ── Remove a fact ──
        engine.remove_atom(translation.metta_expression)
        catalogue.sync(engine.get_all_atoms())

        entry = pm.record_modification(
            engine, user_input, translation.metta_expression, "remove"
        )

        st.session_state.diffs.insert(0, {
            "timestamp": entry.timestamp,
            "diff": entry.diff,
            "rule": translation.metta_expression,
            "operation": "remove",
        })

        return (
            f"🗑️ **Retracted**\n\n"
            f"🛡️ **Jev Intent:** `{translation.intent.value}`\n\n"
            f"**Removed atom:** `{translation.metta_expression}`\n\n"
            f"*{translation.verbal_reasoning}*\n\n"
            f"State hash: `{pm.current_hash}`"
        )

    elif translation.intent == IntentType.CORRECTION:
        # ── Correct a fact (requires evaluating both remove and add commands) ──
        # Since the LLM returns explicit !(remove-atom ...) and !(add-atom ...) scripts,
        # we can just use engine.run_raw() to execute both forms automatically.
        results = engine.run_raw(translation.metta_expression)
        
        entry = pm.record_modification(
            engine, user_input, translation.metta_expression, "correction"
        )

        st.session_state.diffs.insert(0, {
            "timestamp": entry.timestamp,
            "diff": entry.diff,
            "rule": translation.metta_expression,
            "operation": "correction",
        })

        return (
            f"🔄 **Knowledge Corrected**\n\n"
            f"**Execution Script:**\n```\n{translation.metta_expression}\n```\n\n"
            f"*{translation.verbal_reasoning}*\n\n"
            f"State hash: `{pm.current_hash}`"
        )

    elif translation.intent == IntentType.CLARIFICATION:
        return (
            f"🤔 **Clarification needed**\n\n"
            f"*{translation.verbal_reasoning}*"
        )

    else:
        # Conversation
        return f"💬 *{translation.verbal_reasoning}*"


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Layout
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

# Header
st.markdown("""
<div style="text-align: center; padding: 20px 0 10px;">
    <h1 style="
        background: linear-gradient(135deg, #58a6ff, #bc8cff, #f778ba);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 2.2em;
        margin-bottom: 4px;
    ">🧠 Super Memory</h1>
    <p style="color: #8b949e; font-size: 0.95em;">
        Self-Evolving Neuro-Symbolic Agent · MeTTa AtomSpace × LLM
    </p>
</div>
""", unsafe_allow_html=True)

# Split layout
left_col, right_col = st.columns([3, 2], gap="large")


# ── LEFT PANE: Chat & Interaction ──
with left_col:
    st.markdown(
        '<div class="section-header">💬 Agent Interaction</div>',
        unsafe_allow_html=True,
    )

    # Chat history
    chat_container = st.container(height=500)
    with chat_container:
        for msg in st.session_state.messages:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])

    # Input
    if user_input := st.chat_input("Teach me a fact or ask a question..."):
        # Add user message
        st.session_state.messages.append({
            "role": "user",
            "content": user_input,
        })

        # Process
        response = process_input(user_input)

        # Add assistant response
        st.session_state.messages.append({
            "role": "assistant",
            "content": response,
        })

        st.rerun()


# ── RIGHT PANE: Auditable State ──
with right_col:
    engine: MeTTaEngine = st.session_state.engine
    pm: PersistenceManager = st.session_state.persistence

    # ── Persistence Status ──
    st.markdown(
        '<div class="section-header">📡 System Status</div>',
        unsafe_allow_html=True,
    )

    status_cols = st.columns(3)
    with status_cols[0]:
        if st.session_state.loaded_from_disk:
            st.markdown(
                '<span class="status-pill status-loaded">🔄 Restored</span>',
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                '<span class="status-pill status-fresh">✨ Fresh</span>',
                unsafe_allow_html=True,
            )
    with status_cols[1]:
        atom_count = len(engine.get_all_atoms())
        st.metric("Atoms", atom_count)
    with status_cols[2]:
        mode = "Mock" if engine.is_mock else "Native"
        st.metric("Engine", mode)

    st.caption(f"Session: `{pm.session_id}` · Hash: `{pm.current_hash}`")

    # ── Live AtomSpace Viewer ──
    st.markdown(
        '<div class="section-header">🔬 Live AtomSpace</div>',
        unsafe_allow_html=True,
    )

    atoms = engine.get_all_atoms()
    # Filter out backward chainer rules from the viewer (they're infrastructure, not user data)
    display_atoms = [a for a in atoms if not a.strip().startswith("(= (bc")]
    if display_atoms:
        atom_html = "<br>".join(
            f'<span class="atom-line">{_esc(a)}</span>' for a in sorted(display_atoms)
        )
        st.markdown(
            f'<div class="atom-viewer">{atom_html}</div>',
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            '<div class="atom-viewer">'
            '<span class="atom-empty">∅ Empty — teach me something!</span>'
            '</div>',
            unsafe_allow_html=True,
        )

    # ── Diff History ──
    st.markdown(
        '<div class="section-header">📝 Knowledge Diffs</div>',
        unsafe_allow_html=True,
    )

    diffs = st.session_state.diffs
    if diffs:
        for i, d in enumerate(diffs[:10]):
            op_emoji = "➕" if d["operation"] == "add" else "➖"
            with st.expander(
                f"{op_emoji} `{d['rule']}`  —  {d['timestamp'][:19]}",
                expanded=(i == 0),
            ):
                st.markdown(
                    diff_to_html(d["diff"]),
                    unsafe_allow_html=True,
                )
    else:
        st.caption("No modifications yet. Teach me a fact to see diffs!")

    # ── Audit Log ──
    with st.expander("📋 Full Audit Log"):
        log_entries = pm.get_audit_log(limit=20)
        if log_entries:
            for entry in reversed(log_entries):
                st.json(entry)
        else:
            st.caption("No audit entries yet.")
