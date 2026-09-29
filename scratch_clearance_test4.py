from core.metta_engine import MeTTaEngine
engine = MeTTaEngine()
engine.load_program("""
(= (CanAccess $user $lab)
   (, (RequiresClearance $lab $lvl)
      (HasClearance $user $lvl)))

(RequiresClearance LabA Level3)
(HasClearance Anush Level2)
""")
print(engine._rules)
