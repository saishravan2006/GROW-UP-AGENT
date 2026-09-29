import sys
sys.path.insert(0, r"C:\Users\saish\.gemini\antigravity-ide\brain\b93c716c-c6a8-45f7-ba9a-bb5333a7a8ef\scratch")
from metta_engine import MeTTaEngine
engine = MeTTaEngine()
engine.add_atom('(= (CanAccess  ) (, (RequiresClearance  ) (HasClearance  )))')
print("ATOMS:")
for a in engine.get_all_atoms():
    print(repr(a))
