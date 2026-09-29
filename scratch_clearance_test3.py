from core.metta_engine import MeTTaEngine
engine = MeTTaEngine()
import logging
logging.basicConfig(level=logging.DEBUG)
engine.load_program("""
(= (CanAccess $user $lab)
   (, (RequiresClearance $lab $lvl)
      (HasClearance $user $lvl)))

(RequiresClearance LabA Level3)
(HasClearance Anush Level2)
""")

print("--- PYTHON BACKWARD CHAIN ---")
res3 = engine.backward_chain("(CanAccess Anush LabA)")
print("Result:", res3)
