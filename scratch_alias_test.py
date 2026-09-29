from core.metta_engine import MeTTaEngine
engine = MeTTaEngine()

engine.add_atom("(= (Alias PotheriLake) PotheriLakeCleanup)")
engine.add_atom("(RequiresClearance PotheriLakeCleanup Level1)")

print("--- NATIVE QUERY WITH ALIAS ---")
res = engine.query("(RequiresClearance PotheriLake $lvl)", "(RequiresClearance PotheriLake $lvl)")
print("Result:", res)
