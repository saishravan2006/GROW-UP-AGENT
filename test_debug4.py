import os
import json
import sqlite3
from dotenv import load_dotenv
load_dotenv()
from core.metta_engine import MeTTaEngine
from core.persistence import PersistenceManager
from core.agent import NeuroSymbolicAgent
from core.catalogue import EntityCatalogue

pm = PersistenceManager()
pm._conn = sqlite3.connect("data/test_ledger.sqlite", check_same_thread=False)
pm._conn.row_factory = sqlite3.Row
pm._init_db()

engine = MeTTaEngine()
catalogue = EntityCatalogue()
agent = NeuroSymbolicAgent(engine, catalogue, pm)

agent._schema_checked = True
res = agent._handle_tool_call("learn_fact", {"metta_atom": "(MeetingDay Club Friday 2026-10-02)"}, "u_test")
res_dict = json.loads(res)
print("Response:", res_dict)
