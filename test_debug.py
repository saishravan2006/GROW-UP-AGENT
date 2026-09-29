import os
import json
import sqlite3
from core.metta_engine import MeTTaEngine
from core.persistence import PersistenceManager
from core.agent import NeuroSymbolicAgent
from core.catalogue import EntityCatalogue

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

u_id = pm.record_utterance("LabAlpha requires Level3")
pm.record_modification(engine, u_id, "(RequiresClearance LabAlpha Level3)", "add")
engine.add_atom("(RequiresClearance LabAlpha Level3)")

res = agent._handle_tool_call("query_knowledge", {"metta_query": "(RequiresClearance LabAlpha $x)"}, "u_test")
res_dict = json.loads(res)
print("Evidence:", res_dict.get("evidence"))
