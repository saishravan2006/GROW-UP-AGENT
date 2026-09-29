import os
from dotenv import load_dotenv
load_dotenv()

from core.agent import NeuroSymbolicAgent, SYSTEM_PROMPT
from core.metta_engine import MeTTaEngine
from core.catalogue import EntityCatalogue
from core.persistence import PersistenceManager

# Delete old sqlite to start fresh
try:
    os.remove("data/evidence_ledger.sqlite")
except OSError:
    pass

pm = PersistenceManager()
engine = MeTTaEngine()
pm.load_state(engine) # Should be empty
catalogue = EntityCatalogue()

agent = NeuroSymbolicAgent(engine, catalogue, pm)

messages = [{"role": "system", "content": SYSTEM_PROMPT}]

print("\n--- TEST 1: Unknown Query ---")
res = agent.chat(messages, "Who is meeting in Lab 3?")
print("Agent:", res)

print("\n--- TEST 2: Teaching ---")
res = agent.chat(messages, "The robotics club meets in Lab 3 every Friday.")
print("Agent:", res)

print("\n--- TEST 3: Paraphrased Query ---")
res = agent.chat(messages, "Where does the robotics club meet on Fridays?")
print("Agent:", res)
