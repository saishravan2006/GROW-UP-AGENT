import os
import sys
from dotenv import load_dotenv
load_dotenv()
from pathlib import Path
from core.metta_engine import MeTTaEngine
from core.persistence import PersistenceManager
from core.catalogue import EntityCatalogue
from core.agent import NeuroSymbolicAgent, SYSTEM_PROMPT

def run_test(name, messages, user_input, agent):
    print(f"\n{'='*50}\nTEST: {name}\n{'='*50}")
    print(f"USER: {user_input}")
    resp = agent.chat(messages, user_input)
    print(f"ENI: {resp}")
    return messages

engine = MeTTaEngine()
pm = PersistenceManager(Path(os.getcwd()))
pm.load_state(engine)
catalogue = EntityCatalogue()
catalogue.sync(engine.get_all_atoms())
agent = NeuroSymbolicAgent(engine, catalogue, pm)

messages = [{"role": "system", "content": SYSTEM_PROMPT}]

messages = run_test("1. Basic Teaching & Deductive Reasoning", messages, "(= (CanAccess  ) (, (RequiresClearance  ) (HasClearance  )))", agent)
messages = run_test("Teach Facts", messages, "LabAlpha requires Level3. Anush has Level2.", agent)
messages = run_test("Query Clearance (Should fail)", messages, "Can Anush access LabAlpha?", agent)
messages = run_test("Upgrade Clearance", messages, "Anush has Level3 clearance now.", agent)
messages = run_test("Query Clearance (Should succeed)", messages, "Can Anush access LabAlpha now?", agent)

# Reset for Ambiguity test
engine = MeTTaEngine()
catalogue = EntityCatalogue()
agent = NeuroSymbolicAgent(engine, catalogue, pm)
messages = [{"role": "system", "content": SYSTEM_PROMPT}]
messages = run_test("2. Ambiguity", messages, "AISecurityHackathon is an Event. AIHackathon2024 is an Event.", agent)
messages = run_test("Ambiguous Query", messages, "Where is the hackathon?", agent)
messages = run_test("Clarification Resolution", messages, "the first one", agent)

# Reset for Alias test
engine = MeTTaEngine()
catalogue = EntityCatalogue()
agent = NeuroSymbolicAgent(engine, catalogue, pm)
messages = [{"role": "system", "content": SYSTEM_PROMPT}]
messages = run_test("3. Alias", messages, "AISecurityHackathon is an Event. (= (Alias SecHack) AISecurityHackathon). AISecurityHackathon is located in BuildingC", agent)
messages = run_test("Alias Query", messages, "Where is SecHack?", agent)

