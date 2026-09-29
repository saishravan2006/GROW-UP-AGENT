import logging
logging.basicConfig(level=logging.DEBUG)
from core.metta_engine import MeTTaEngine
from core.persistence import PersistenceManager
pm = PersistenceManager()
engine = MeTTaEngine()
pm.load_state(engine)
print("Atoms loaded:", list(engine.get_all_atoms()))
