"""
test_stage_c.py -- Assertion-based verification for Stage C.
Tests the Jev Decision API integration and the Clarification state machine.
"""
import os
import json
import sqlite3
from unittest.mock import patch, MagicMock

from dotenv import load_dotenv
load_dotenv()

from core.metta_engine import MeTTaEngine
from core.persistence import PersistenceManager
from core.agent import NeuroSymbolicAgent
from core.catalogue import EntityCatalogue
from core.jev_adapter import JevDecisionAPI

def setup_fresh():
    """Create a clean isolated test environment."""
    db_path = "data/test_ledger_c.sqlite"
    try:
        os.remove(db_path)
    except OSError:
        pass
    pm = PersistenceManager()
    pm._conn.close()
    pm._conn = sqlite3.connect(db_path, check_same_thread=False)
    pm._conn.row_factory = sqlite3.Row
    pm._init_db()

    engine = MeTTaEngine()
    catalogue = EntityCatalogue()
    agent = NeuroSymbolicAgent(engine, catalogue, pm)
    return engine, pm, agent

# ================================================================
# 1. Jev Adapter API formatting
# ================================================================
@patch("core.jev_adapter.requests.post")
def test_jev_adapter_format(mock_post):
    # Setup mock response
    mock_resp = MagicMock()
    mock_resp.json.return_value = {
        "answers": {
            "intent": {
                "choice": "clarification"
            }
        }
    }
    mock_post.return_value = mock_resp
    
    jev = JevDecisionAPI()
    intent = jev.route_intent("Is the lab big?")
    
    assert intent == "clarification"
    
    # Assert the correct payload was sent
    mock_post.assert_called_once()
    payload = mock_post.call_args[1]["json"]
    assert payload["model"] == "typesafe/jev-1.13"
    assert payload["state"] == "Is the lab big?"
    assert payload["questions"]["intent"]["type"] == "choice"
    assert "query" in payload["questions"]["intent"]["criteria"]
    
    print("[PASS] 1. Jev Adapter payload structure and parsing is correct")

# ================================================================
# 2. Persistence of Clarifications
# ================================================================
def test_clarification_persistence():
    _, pm, _ = setup_fresh()
    
    # No pending initially
    assert pm.get_pending_clarification() is None
    
    # Record a pending one
    text = "Are they coming?"
    c_id = pm.record_pending_clarification(text)
    
    # Should be retrievable
    pending = pm.get_pending_clarification()
    assert pending is not None
    assert pending["id"] == c_id
    assert pending["original_text"] == text
    
    # Resolve it
    pm.resolve_clarification(c_id)
    
    # Should no longer be pending
    assert pm.get_pending_clarification() is None
    print("[PASS] 2. Clarification Persistence (record, retrieve, resolve)")

# ================================================================
# 3. Agent State Machine: Clarification Short-Circuit
# ================================================================
@patch("core.agent.JevDecisionAPI.route_intent")
def test_agent_clarification_shortcircuit(mock_route):
    # Mock Jev returning 'clarification'
    mock_route.return_value = "clarification"
    
    _, pm, agent = setup_fresh()
    
    # Send a message
    messages = []
    response = agent.chat(messages, "The lab needs Level 4")
    
    # Agent should return the static clarification string immediately
    assert "clarify" in response.lower()
    
    # DB should have a pending clarification
    pending = pm.get_pending_clarification()
    assert pending is not None
    assert pending["original_text"] == "The lab needs Level 4"
    print("[PASS] 3. Agent short-circuits LLM when intent is clarification")

# ================================================================
# 4. Agent State Machine: Clarification Resolution
# ================================================================
@patch("core.agent.JevDecisionAPI.route_intent")
def test_agent_clarification_resolution(mock_route):
    # Mock Jev to return 'learn' on the resolved input
    mock_route.return_value = "learn"
    
    _, pm, agent = setup_fresh()
    
    # Mock Gemini client on the instance
    mock_llm_resp = MagicMock()
    mock_llm_resp.choices[0].message.tool_calls = None
    mock_llm_resp.choices[0].message.content = "OK"
    agent.client = MagicMock()
    agent.client.chat.completions.create.return_value = mock_llm_resp
    
    # Start by forcing a pending state
    pm.record_pending_clarification("The lab needs Level 4")
    
    # Send a message answering the clarification
    messages = []
    response = agent.chat(messages, "I mean LabAlpha")
    
    # The DB should now be resolved
    assert pm.get_pending_clarification() is None
    
    # The Jev router should have been called with the combined text
    mock_route.assert_called_with("The lab needs Level 4 [Clarification: I mean LabAlpha]")
    
    print("[PASS] 4. Agent successfully resolves and combines pending clarifications")

if __name__ == "__main__":
    print("\n=== Stage C Integration Tests ===\n")
    test_jev_adapter_format()
    test_clarification_persistence()
    test_agent_clarification_shortcircuit()
    test_agent_clarification_resolution()
    print("\n=== ALL 4 TESTS PASSED ===\n")
