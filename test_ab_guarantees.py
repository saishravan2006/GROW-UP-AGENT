"""
test_ab_guarantees.py -- Assertion-based verification for Stage A/B.

Each test targets a specific guarantee. No LLM calls are made.
All assertions must pass for A/B to be considered verified.
"""
import os
import json
import sqlite3
import hashlib
from dotenv import load_dotenv
load_dotenv()
from core.metta_engine import MeTTaEngine, DerivationRecord
from core.persistence import PersistenceManager
from core.agent import NeuroSymbolicAgent
from core.catalogue import EntityCatalogue


def setup_fresh():
    """Create a clean isolated test environment."""
    db_path = "data/test_ledger.sqlite"
    try:
        os.remove(db_path)
    except OSError:
        pass
    pm = PersistenceManager()
    # Redirect to test DB
    pm._conn.close()
    pm._conn = sqlite3.connect(db_path, check_same_thread=False)
    pm._conn.row_factory = sqlite3.Row
    pm._init_db()

    engine = MeTTaEngine()
    catalogue = EntityCatalogue()
    agent = NeuroSymbolicAgent(engine, catalogue, pm)
    return engine, pm, agent


# ================================================================
# 1. Evidence-bound output with claim IDs for reported facts
# ================================================================
def test_evidence_bound_reported():
    engine, pm, agent = setup_fresh()
    u_id = pm.record_utterance("LabAlpha requires Level3")
    engine.add_atom("(RequiresClearance LabAlpha Level3)")
    pm.record_modification(engine, u_id, "(RequiresClearance LabAlpha Level3)", "add")

    res = json.loads(agent._handle_tool_call(
        "query_knowledge", {"metta_query": "(RequiresClearance LabAlpha $x)"}, "u_test"
    ))
    assert res["status"] == "success", f"Expected success, got {res}"
    assert len(res["evidence"]) == 1
    ev = res["evidence"][0]
    assert ev["status"] == "ANSWERED"
    assert ev["origin"] == "reported"
    assert ev["answer_bindings"] == "(RequiresClearance LabAlpha Level3)"
    assert len(ev["supporting_claim_ids"]) == 1
    assert ev["supporting_claim_ids"][0].startswith("c_")
    print("[PASS] 1. Evidence-bound output (reported fact with claim ID)")


# ================================================================
# 2. Backward chaining with derivation records
# ================================================================
def test_derivation_provenance():
    engine, pm, agent = setup_fresh()
    rule = "(= (CanAccess $user $lab) (, (RequiresClearance $lab $lvl) (HasClearance $user $lvl)))"
    engine.add_atom(rule)

    u1 = pm.record_utterance("LabB requires Level2")
    engine.add_atom("(RequiresClearance LabB Level2)")
    pm.record_modification(engine, u1, "(RequiresClearance LabB Level2)", "add")

    u2 = pm.record_utterance("Anush has Level2")
    engine.add_atom("(HasClearance Anush Level2)")
    pm.record_modification(engine, u2, "(HasClearance Anush Level2)", "add")

    res = json.loads(agent._handle_tool_call(
        "query_knowledge", {"metta_query": "(CanAccess Anush $x)"}, "u_test"
    ))
    assert res["status"] == "success"
    assert len(res["evidence"]) == 1
    ev = res["evidence"][0]
    assert ev["origin"] == "derived", f"Expected derived, got {ev['origin']}"
    assert ev["answer_bindings"] == "(CanAccess Anush LabB)"
    assert "applied_rule" in ev, "Derivation must include the applied rule"
    assert "CanAccess" in ev["applied_rule"]
    assert len(ev["premise_claim_ids"]) == 2, f"Expected 2 premise claims, got {ev['premise_claim_ids']}"
    assert all(pid.startswith("c_") for pid in ev["premise_claim_ids"])
    assert len(ev["premises"]) == 2
    print("[PASS] 2. Derivation provenance (rule + premise claim IDs)")


# ================================================================
# 3. Shared-variable mismatch blocks incorrect inferences
# ================================================================
def test_shared_variable_mismatch():
    engine, pm, agent = setup_fresh()
    engine.add_atom("(= (CanAccess $user $lab) (, (RequiresClearance $lab $lvl) (HasClearance $user $lvl)))")
    engine.add_atom("(RequiresClearance LabA Level3)")
    engine.add_atom("(RequiresClearance LabB Level2)")
    engine.add_atom("(HasClearance Anush Level1)")
    engine.add_atom("(HasClearance Anush Level2)")

    res = json.loads(agent._handle_tool_call(
        "query_knowledge", {"metta_query": "(CanAccess Anush $x)"}, "u_test"
    ))
    assert res["status"] == "success"
    answers = [e["answer_bindings"] for e in res["evidence"]]
    assert "(CanAccess Anush LabB)" in answers, f"Expected LabB in answers: {answers}"
    assert "(CanAccess Anush LabA)" not in answers, f"LabA should be blocked: {answers}"
    print("[PASS] 3. Shared-variable mismatch blocks LabA (Level3 != Level1/Level2)")


# ================================================================
# 4. Schema validation rejects ALL unknown predicates
# ================================================================
def test_schema_rejects_unknown_predicate():
    engine, pm, agent = setup_fresh()
    engine.add_atom("(MeetingPlace RoboticsClub Lab3)")

    # Even PascalCase unknown predicates must be rejected
    agent._schema_checked = True
    res = json.loads(agent._handle_tool_call(
        "learn_fact", {"metta_atom": "(InventedPredicate A B)"}, "u_test"
    ))
    assert res["status"] == "error"
    assert "not in the approved schema" in res["message"]

    # Known predicate should succeed
    agent._schema_checked = True
    res = json.loads(agent._handle_tool_call(
        "learn_fact", {"metta_atom": "(MeetingPlace ChessClub RoomA)"}, "u_test"
    ))
    assert res["status"] == "success"
    print("[PASS] 4. Schema rejects unknown predicates (including PascalCase)")


# ================================================================
# 5. Schema bypass (no get_schema call) is rejected
# ================================================================
def test_schema_bypass_rejected():
    engine, pm, agent = setup_fresh()
    agent._schema_checked = False
    res = json.loads(agent._handle_tool_call(
        "learn_fact", {"metta_atom": "(MeetingPlace A B)"}, "u_test"
    ))
    assert res["status"] == "error"
    assert "VALIDATION FAILED" in res["message"]
    print("[PASS] 5. Schema bypass rejected")


# ================================================================
# 6. Temporal scope: check BOTH atom and original utterance
# ================================================================
def test_temporal_scope_from_utterance():
    engine, pm, agent = setup_fresh()
    engine.add_atom("(MeetingDay RoboticsClub Friday)")

    # Register an utterance containing temporal language
    u_id = pm.record_utterance("The robotics club moves to Lab 5 next Friday")

    agent._schema_checked = True
    res = json.loads(agent._handle_tool_call(
        "learn_fact", {"metta_atom": "(MeetingDay RoboticsClub Friday)"}, u_id
    ))
    assert res["status"] == "error"
    assert "temporal" in res["message"].lower(), f"Expected temporal rejection, got: {res['message']}"

    # Atom-level temporal patterns
    agent._schema_checked = True
    res = json.loads(agent._handle_tool_call(
        "learn_fact", {"metta_atom": "(MeetingDay Club 2026-10-02)"}, "u_test"
    ))
    assert res["status"] == "error"

    # Variant: "tomorrow"
    u2 = pm.record_utterance("Lab meeting is tomorrow")
    agent._schema_checked = True
    res = json.loads(agent._handle_tool_call(
        "learn_fact", {"metta_atom": "(MeetingDay LabTeam Monday)"}, u2
    ))
    assert res["status"] == "error"
    print("[PASS] 6. Temporal scope checked in both atom and utterance")


# ================================================================
# 7. Idempotent retry: same operation ID is not duplicated
# ================================================================
def test_idempotent_retry():
    engine, pm, agent = setup_fresh()
    engine.add_atom("(MeetingPlace RoboticsClub Lab3)")

    u_id = pm.record_utterance("Chess club meets in Room B")
    agent._schema_checked = True
    res1 = json.loads(agent._handle_tool_call(
        "learn_fact", {"metta_atom": "(MeetingPlace ChessClub RoomB)"}, u_id
    ))
    assert res1["status"] == "success"
    op_id = res1["operation_id"]

    # Retry the exact same operation
    agent._schema_checked = True
    res2 = json.loads(agent._handle_tool_call(
        "learn_fact", {"metta_atom": "(MeetingPlace ChessClub RoomB)"}, u_id
    ))
    assert res2["status"] == "success"
    assert "Duplicate" in res2["message"], f"Expected duplicate skip, got: {res2}"

    # Verify only one claim exists in DB
    cursor = pm._conn.cursor()
    cursor.execute("SELECT COUNT(*) FROM claims WHERE op_id = ?", (op_id,))
    assert cursor.fetchone()[0] == 1
    print("[PASS] 7. Idempotent retry (duplicate op_id skipped, single DB row)")


# ================================================================
# 8. Recovery after SQLite commit but before MeTTa projection
# ================================================================
def test_recovery_after_partial_failure():
    engine, pm, agent = setup_fresh()
    u_id = pm.record_utterance("Test recovery scenario")

    # Simulate: SQLite commit succeeds
    pm.record_modification(engine, u_id, "(TestFact Recovery Works)", "add")

    # Simulate: MeTTa projection FAILS (engine was never updated)
    # Verify engine does NOT have the atom
    direct_query = engine.query("(TestFact Recovery Works)")
    assert direct_query.is_empty, "Engine should NOT have the atom yet (simulated projection failure)"

    # Recovery: load_state replays from SQLite
    pm.load_state(engine)

    # Now engine should have it
    recovered = engine.query("(TestFact Recovery Works)")
    assert not recovered.is_empty, "Engine should have the atom after recovery"

    # Retry the same operation ID -- should be idempotent
    op_id = "op_" + hashlib.sha256(f"{u_id}:(TestFact Recovery Works):add".encode()).hexdigest()[:12]
    assert pm.operation_exists(op_id), f"Operation {op_id} should exist after recovery"
    print("[PASS] 8. Recovery after partial failure (SQLite commit -> MeTTa crash -> reload)")


# ================================================================
# 9. UNVERIFIED label for sync errors
# ================================================================
def test_unverified_sync_error():
    engine, pm, agent = setup_fresh()
    # Add an atom directly to engine without recording in SQLite
    engine.add_atom("(OrphanFact NoRecord Exists)")

    res = json.loads(agent._handle_tool_call(
        "query_knowledge", {"metta_query": "(OrphanFact NoRecord $x)"}, "u_test"
    ))
    assert res["status"] == "success"
    assert len(res["evidence"]) == 1
    ev = res["evidence"][0]
    assert ev["status"] == "UNVERIFIED"
    assert ev["origin"] == "unknown"
    assert "sync error" in ev.get("warning", "").lower()
    print("[PASS] 9. UNVERIFIED label for atoms without claim ID or derivation")


# ================================================================
# 10. DerivationRecord structure from backward chainer
# ================================================================
def test_derivation_record_structure():
    engine, pm, agent = setup_fresh()
    rule = "(= (CanAccess $user $lab) (, (RequiresClearance $lab $lvl) (HasClearance $user $lvl)))"
    engine.add_atom(rule)
    engine.add_atom("(RequiresClearance LabB Level2)")
    engine.add_atom("(HasClearance Anush Level2)")

    result = engine.backward_chain("(CanAccess Anush $x)")
    assert not result.is_empty
    assert len(result.derivations) == 1

    drec = result.derivations[0]
    assert isinstance(drec, DerivationRecord)
    assert drec.answer == "(CanAccess Anush LabB)"
    assert drec.origin == "derived"
    assert "CanAccess" in drec.rule_text
    assert "(RequiresClearance LabB Level2)" in drec.premises
    assert "(HasClearance Anush Level2)" in drec.premises
    print("[PASS] 10. DerivationRecord structure from Python backward chainer")


# ================================================================
# Main
# ================================================================
if __name__ == "__main__":
    print("\n=== Stage A/B Guarantee Tests ===\n")
    test_evidence_bound_reported()
    test_derivation_provenance()
    test_shared_variable_mismatch()
    test_schema_rejects_unknown_predicate()
    test_schema_bypass_rejected()
    test_temporal_scope_from_utterance()
    test_idempotent_retry()
    test_recovery_after_partial_failure()
    test_unverified_sync_error()
    test_derivation_record_structure()
    print("\n=== ALL 10 TESTS PASSED ===\n")
