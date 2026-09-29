from core.metta_engine import MeTTaEngine
engine = MeTTaEngine()
engine.add_atom("(RequiresClearance LabAlpha Level3)")
print("Query:", engine.query("(RequiresClearance LabAlpha )"))
print("BC:", engine.backward_chain("(RequiresClearance LabAlpha )"))
