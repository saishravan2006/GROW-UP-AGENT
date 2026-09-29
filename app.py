"""
app.py
🧠 Super Memory AI - Architecture V2 (Tool-Augmented Neuro-Symbolic Agent)
"""
from __future__ import annotations

import logging
import streamlit as st
from datetime import datetime
import os
from dotenv import load_dotenv

load_dotenv()

# We no longer use translator.py! We use our new Agent!
from core.metta_engine import MeTTaEngine
from core.persistence import PersistenceManager
from core.catalogue import EntityCatalogue
from core.agent import NeuroSymbolicAgent, SYSTEM_PROMPT

logging.basicConfig(level=logging.INFO)

st.set_page_config(layout="wide", page_title="Super Memory AI - V2", page_icon="🧠")

st.markdown("""
    <style>
    .stApp { background-color: #0E1117; color: #FAFAFA; }
    .chat-container { border-right: 1px solid #333; padding-right: 20px; }
    </style>
""", unsafe_allow_html=True)

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Session State Initialization
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
if "initialized" not in st.session_state:
    engine = MeTTaEngine()
    from pathlib import Path; pm = PersistenceManager(Path(os.getcwd()))
    pm.load_state(engine)
    
    catalogue = EntityCatalogue()
    catalogue.sync(engine.get_all_atoms())
    
    st.session_state.engine = engine
    st.session_state.pm = pm
    st.session_state.catalogue = catalogue
    st.session_state.agent = NeuroSymbolicAgent(engine, catalogue, pm)
    
    st.session_state.messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    st.session_state.initialized = True

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# UI Layout
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
col1, col2 = st.columns([3, 2])

with col1:
    st.title("🧠 ENI (Architecture V2)")
    st.markdown("*Conversational Neuro-Symbolic Agent with strict MeTTa tool-use.*")
    st.divider()
    
    # Render chat history
    for msg in st.session_state.messages:
        if isinstance(msg, dict):
            role = msg.get("role")
            content = msg.get("content")
            if role in ("user", "assistant") and content:
                with st.chat_message(role):
                    st.markdown(content)
        else:
            # Handle openai API message objects
            if msg.role == "assistant" and msg.content:
                with st.chat_message("assistant"):
                    st.markdown(msg.content)

    # Chat Input
    if prompt := st.chat_input("Talk to ENI..."):
        with st.chat_message("user"):
            st.markdown(prompt)
            
        with st.spinner("ENI is thinking and using tools..."):
            response_text = st.session_state.agent.chat(st.session_state.messages, prompt)
            # The agent modifies st.session_state.messages in place inside chat()
            
        with st.chat_message("assistant"):
            st.markdown(response_text)
            
        st.rerun()

with col2:
    st.subheader("📚 Live AtomSpace")
    st.markdown("This is the grounded knowledge graph ENI is reading/writing from using tools.")
    
    atoms = st.session_state.engine.get_all_atoms()
    if not atoms:
        st.info("The AtomSpace is currently empty.")
    else:
        for a in atoms:
            st.code(a, language="lisp")
            
    st.divider()
    st.subheader("📝 Audit Log")
    log = st.session_state.pm.get_audit_log()
    for entry in reversed(log[-10:]):
        op = entry.get("operation", "add")
        op_color = "green" if op == "add" else "red"
        st.markdown(f"**{entry.get('timestamp', '')}**")
        st.markdown(f"> *\"{entry.get('user_input', '')}\"*")
        st.markdown(f":{op_color}[{op.upper()}] `{entry.get('diff', '')}`")
