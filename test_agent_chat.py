import os
import sys
from dotenv import load_dotenv
load_dotenv()
from pathlib import Path
from core.metta_engine import MeTTaEngine
from core.persistence import PersistenceManager
from core.catalogue import EntityCatalogue
from core.agent import NeuroSymbolicAgent, SYSTEM_PROMPT

engine = MeTTaEngine()
pm = PersistenceManager(Path(os.getcwd()))
pm.load_state(engine)
catalogue = EntityCatalogue()
catalogue.sync(engine.get_all_atoms())
agent = NeuroSymbolicAgent(engine, catalogue, pm)

messages = [{"role": "system", "content": SYSTEM_PROMPT}]

print("Testing Chat...")
resp = agent.chat(messages, "Learn that AISecurityHackathon requires Level 3 clearance.")
print("\nAGENT:\n", resp)
