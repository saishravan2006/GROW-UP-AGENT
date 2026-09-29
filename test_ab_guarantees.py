import os
import json
import sqlite3
from dotenv import load_dotenv
load_dotenv()
from core.metta_engine import MeTTaEngine
from core.persistence import PersistenceManager
from core.agent import NeuroSymbolicAgent
from core.catalogue import EntityCatalogue

def setup_fresh():
    try:
        os.remove("data/test_ledger.sqlite")
    except:
        pass
    pm = PersistenceManager()
    pm._conn = sqlite3.connect("data/test_ledger.sqlite", check_same_thread=False)
    pm._conn.row_factory = sqlite3.Row
    pm._init_db()
    
    engine = MeTTaEngine()
    catalogue = EntityCatalogue()
    agent = NeuroSymbolicAgent(engine, catalogue, pm)
    return engine, pm, agent

def test_evidence_bound_output():
    engine, pm, agent = setup_fresh()
    u_id = pm.record_utterance("LabAlpha requires Level3")
    pm.record_modification(engine, u_id, "(RequiresClearance LabAlpha Level3)", "add")
    engine.add_atom("(RequiresClearance LabAlpha Level3)")
    
    res = agent._handle_tool_call("query_knowledge", {"metta_query": "(RequiresClearance LabAlpha $x)"}, "u_test")
    res_dict = json.loads(res)
    
    assert res_dict["status"] == "success"
    assert len(res_dict["evidence"]) == 1
    ev = res_dict["evidence"][0]
    assert ev["status"] == "ANSWERED"
    assert ev["answer_bindings"] == "(RequiresClearance LabAlpha Level3)"
    assert len(ev["supporting_claim_ids"]) == 1
    assert ev["supporting_claim_ids"][0].startswith("c_")
    print("[PASS] Evidence-bound output working correctly")

def test_shared_variable_mismatch_and_backtracking():
    engine, pm, agent = setup_fresh()
    engine.add_atom("(= (CanAccess $user $lab) (, (RequiresClearance $lab $lvl) (HasClearance $user $lvl)))")
    engine.add_atom("(RequiresClearance LabA Level3)")
    engine.add_atom("(RequiresClearance LabB Level2)")
    engine.add_atom("(HasClearance Anush Level1)")
    engine.add_atom("(HasClearance Anush Level2)")
    
    res = agent._handle_tool_call("query_knowledge", {"metta_query": "(CanAccess Anush $x)"}, "u_test")
    res_dict = json.loads(res)
    
    assert res_dict["status"] == "success"
    assert len(res_dict["evidence"]) == 1
    assert res_dict["evidence"][0]["answer_bindings"] == "(CanAccess Anush LabB)"
    print("[PASS] Shared-variable mismatch & Backtracking working correctly")
    
def test_write_validation_rejection():
    engine, pm, agent = setup_fresh()
    
    agent._schema_checked = False
    res = agent._handle_tool_call("learn_fact", {"metta_atom": "(InventedPredicate A B)"}, "u_test")
    res_dict = json.loads(res)
    assert res_dict["status"] == "error"
    assert "VALIDATION FAILED" in res_dict["message"]
    
    agent._schema_checked = True
    res = agent._handle_tool_call("learn_fact", {"metta_atom": "(MeetingDay Club Friday 2026-10-02)"}, "u_test")
    res_dict = json.loads(res)
    assert res_dict["status"] == "error"
    assert "Unstructured temporal exceptions" in res_dict["message"]
    print("[PASS] Write validation rejections working correctly")

def test_idempotent_recovery():
    engine, pm, agent = setup_fresh()
    u_id = pm.record_utterance("Test recovery")
    
    pm.record_modification(engine, u_id, "(Test Recovery Success)", "add")
    
    engine._atoms = []
    engine._metta.run("!(remove-atom &self (Test Recovery Success))")
    
    pm.load_state(engine)
    
    res = engine.query("(Test Recovery Success)")
    assert not res.is_empty
    print("[PASS] Idempotent SQLite to MeTTa recovery working correctly")

if __name__ == "__main__":
    print("\n--- Running Stage A/B Guarantees ---")
    test_evidence_bound_output()
    test_shared_variable_mismatch_and_backtracking()
    test_write_validation_rejection()
    test_idempotent_recovery()
    print("--- ALL TESTS PASSED ---\n")
