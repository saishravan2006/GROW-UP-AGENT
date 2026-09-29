import sys
sys.path.insert(0, r"C:\Users\saish\.gemini\antigravity-ide\brain\b93c716c-c6a8-45f7-ba9a-bb5333a7a8ef\scratch")
from metta_engine import MeTTaEngine

engine = MeTTaEngine()
engine.add_atom('(= (CanAccess  ) (, (RequiresClearance  ) (HasClearance  )))')
engine.add_atom('(RequiresClearance LabAlpha Level3)')
engine.add_atom('(HasClearance Anush Level2)')
engine.add_atom('(HasClearance Anush Level3)')

print("STARTING TEST")
res = engine.backward_chain("(CanAccess Anush LabAlpha)")
print("RESULT:", res.raw)
