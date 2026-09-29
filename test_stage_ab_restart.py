import os
import logging
logging.basicConfig(level=logging.DEBUG)
from dotenv import load_dotenv
load_dotenv()

from core.agent import NeuroSymbolicAgent, SYSTEM_PROMPT
from core.metta_engine import MeTTaEngine
from core.catalogue import EntityCatalogue
from core.persistence import PersistenceManager

print("\n--- RESTARTING APPLICATION ---")
pm = PersistenceManager()
engine = MeTTaEngine()
pm.load_state(engine) # Should load from SQLite
catalogue = EntityCatalogue()
catalogue.sync(engine.get_all_atoms())

agent = NeuroSymbolicAgent(engine, catalogue, pm)

messages = [{"role": "system", "content": SYSTEM_PROMPT}]

print("\n--- TEST 4: Query After Restart ---")
res = agent.chat(messages, "Where does the robotics club meet?")
print("Agent:", res)
