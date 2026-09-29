"""
test_stage_d.py -- Assertion-based verification for Stage D.
Tests the Temporal Resolver (ValidDate, Recurrence, Exception).
"""
import os
import json
import sqlite3
from unittest.mock import patch

from dotenv import load_dotenv
load_dotenv()

from core.metta_engine import MeTTaEngine
from core.persistence import PersistenceManager
from core.agent import NeuroSymbolicAgent
from core.catalogue import EntityCatalogue
from core.jev_adapter import JevDecisionAPI

def setup_fresh():
    db_path = "data/test_ledger_d.sqlite"
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
    
    # Add predicates to graph so they pass schema validation
    engine.add_atom("(MeetingPlace dummy dummy)")
    engine.add_atom("(ValidDate dummy dummy)")
    engine.add_atom("(Recurrence dummy dummy dummy)")
    engine.add_atom("(Exception dummy dummy dummy)")
    
    agent = NeuroSymbolicAgent(engine, catalogue, pm)
    return engine, pm, agent

# ================================================================
# 1. Unstructured Temporal Rejection
# ================================================================
@patch("core.agent.JevDecisionAPI.route_intent")
def test_unstructured_temporal_rejection(mock_route):
    mock_route.return_value = "learn"
    _, pm, agent = setup_fresh()
    agent._schema_checked = True # bypass schema check for test
    
    u_id = pm.record_utterance("Lab meeting is on 2026-10-02")
    res = json.loads(agent._handle_tool_call(
        "learn_fact", {"metta_atom": "(MeetingPlace LabTeam RoomC)"}, u_id
    ))
    
    assert res["status"] == "error"
    assert "VALIDATION FAILED" in res["message"]
    assert "(ValidDate" in res["message"]
    
    print("[PASS] 1. Unstructured temporal language correctly rejected")

# ================================================================
# 2. Structured Temporal Acceptance
# ================================================================
@patch("core.agent.JevDecisionAPI.route_intent")
def test_structured_temporal_acceptance(mock_route):
    mock_route.return_value = "learn"
    _, pm, agent = setup_fresh()
    agent._schema_checked = True 
    
    u_id = pm.record_utterance("Lab meeting is on 2026-10-02")
    res = json.loads(agent._handle_tool_call(
        "learn_fact", {"metta_atom": '(ValidDate (MeetingPlace LabTeam RoomC) "2026-10-02")'}, u_id
    ))
    
    assert res["status"] == "success"
    assert '(ValidDate (MeetingPlace LabTeam RoomC) "2026-10-02")' in res["message"]
    
    print("[PASS] 2. Structured temporal language (ValidDate) successfully recorded")

# ================================================================
# 3. Nested Query Unification (Proof of temporal querying)
# ================================================================
def test_nested_temporal_query():
    engine, pm, agent = setup_fresh()
    
    # Engine should be able to query nested temporal structures natively
    engine.add_atom('(Recurrence (MeetingPlace RoboticsClub Lab5) Weekly Friday)')
    engine.add_atom('(Exception (MeetingPlace RoboticsClub Lab5) "2026-10-09" (MeetingPlace RoboticsClub RoomB))')
    
    res1 = engine.query('(Recurrence (MeetingPlace RoboticsClub $loc) $freq $day)', template='(Recurrence (MeetingPlace RoboticsClub $loc) $freq $day)')
    assert not res1.is_empty
    assert '(Recurrence (MeetingPlace RoboticsClub Lab5) Weekly Friday)' in res1.formatted
    
    res2 = engine.query('(Exception (MeetingPlace RoboticsClub $default_loc) $date (MeetingPlace RoboticsClub $new_loc))', template='(Exception (MeetingPlace RoboticsClub $default_loc) $date (MeetingPlace RoboticsClub $new_loc))')
    assert not res2.is_empty
    assert 'Exception' in res2.formatted
    
    print("[PASS] 3. Nested temporal bindings accurately unify in queries")


if __name__ == "__main__":
    print("\n=== Stage D Temporal Resolver Tests ===\n")
    test_unstructured_temporal_rejection()
    test_structured_temporal_acceptance()
    test_nested_temporal_query()
    print("\n=== ALL 3 TESTS PASSED ===\n")
