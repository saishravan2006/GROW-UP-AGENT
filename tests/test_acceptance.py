"""
tests/test_acceptance.py
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Deterministic acceptance tests for GROW-UP-AGENT.
All external model responses are mocked. Tests the actual execution
path used by app.py — backward_chain, catalogue, clarification lifecycle.
"""
import os
import sys
import time

# Ensure no API calls
os.environ["TYPESAFE_API_KEY"] = ""
os.environ["GEMINI_API_KEY"] = ""
os.environ["OPENROUTER_API_KEY"] = ""

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.metta_engine import MeTTaEngine
from core.catalogue import EntityCatalogue
from core.translator import (
    LLMTranslator, IntentType, ClarificationPayload, 
    UncertaintyPayload, MeTTaTranslation, Jev
)

PASS = 0
FAIL = 0

def check(name, condition, detail=""):
    global PASS, FAIL
    if condition:
        PASS += 1
        print(f"  ✅ {name}")
    else:
        FAIL += 1
        print(f"  ❌ {name}: {detail}")


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Test 1: Backward Chainer — Mismatched clearance
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
print("\n═══ TEST 1: Mismatched clearance → no proof ═══")
engine = MeTTaEngine()
engine.add_atom("(= (CanAccess $user $lab) (, (RequiresClearance $lab $lvl) (HasClearance $user $lvl)))")
engine.add_atom("(RequiresClearance LabA Level3)")
engine.add_atom("(HasClearance Anush Level2)")

result = engine.backward_chain("(CanAccess Anush LabA)")
check("Mismatch Level3/Level2 → empty", result.is_empty, f"Got: {result.raw}")


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Test 2: Backward Chainer — Matching clearance
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
print("\n═══ TEST 2: Matching clearance → proof ═══")
engine2 = MeTTaEngine()
engine2.add_atom("(= (CanAccess $user $lab) (, (RequiresClearance $lab $lvl) (HasClearance $user $lvl)))")
engine2.add_atom("(RequiresClearance LabA Level2)")
engine2.add_atom("(HasClearance Anush Level2)")

result2 = engine2.backward_chain("(CanAccess Anush LabA)")
check("Match Level2/Level2 → proof", not result2.is_empty, f"Got: {result2.raw}")
check("Result is exact", "(CanAccess Anush LabA)" in result2.raw, f"Got: {result2.raw}")


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Test 3: Multiple candidate bindings → correct backtracking
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
print("\n═══ TEST 3: Multiple candidate bindings → backtracking ═══")
engine3 = MeTTaEngine()
engine3.add_atom("(= (CanAccess $user $lab) (, (RequiresClearance $lab $lvl) (HasClearance $user $lvl)))")
engine3.add_atom("(RequiresClearance LabA Level3)")
engine3.add_atom("(RequiresClearance LabB Level2)")
engine3.add_atom("(RequiresClearance LabC Level2)")
engine3.add_atom("(HasClearance Anush Level2)")

result3 = engine3.backward_chain("(CanAccess Anush $x)")
check("Backtracking finds LabB and LabC", 
      "(CanAccess Anush LabB)" in result3.raw and "(CanAccess Anush LabC)" in result3.raw,
      f"Got: {result3.raw}")
check("Backtracking excludes LabA",
      "(CanAccess Anush LabA)" not in result3.raw,
      f"Got: {result3.raw}")


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Test 4: Ambiguous hackathon → clarification
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
print("\n═══ TEST 4: Ambiguous hackathon → clarification ═══")
engine4 = MeTTaEngine()
engine4.add_atom("(IsA AISecurityHackathon Event)")
engine4.add_atom("(IsA AIHackathon2024 Event)")

cat4 = EntityCatalogue()
cat4.sync(engine4.get_all_atoms())

trans = LLMTranslator()
result4 = trans.translate("Can Anush enter the hackathon?", catalogue=cat4)

check("Returns ClarificationPayload", isinstance(result4, ClarificationPayload), f"Got: {type(result4).__name__}")
if isinstance(result4, ClarificationPayload):
    check("Intent is CLARIFICATION", result4.intent == IntentType.CLARIFICATION)
    check("Has 2 candidates", len(result4.candidates) == 2, f"Got: {result4.candidates}")
    check("Compiler not called (no metta_expression)", not hasattr(result4, 'metta_expression') or not getattr(result4, 'metta_expression', ''))


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Test 5: "The second one" → resumes correct pending request
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
print("\n═══ TEST 5: 'The second one' resumes pending ═══")
if isinstance(result4, ClarificationPayload):
    result5 = trans.translate("the second one", catalogue=cat4, pending=result4)
    check("Returns MeTTaTranslation (not clarification)", isinstance(result5, MeTTaTranslation), f"Got: {type(result5).__name__}")
    if isinstance(result5, MeTTaTranslation):
        check("Intent is QUERY", result5.intent == IntentType.QUERY)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Test 6: Explicit event → exact canonical identity
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
print("\n═══ TEST 6: Explicit event → canonical identity ═══")
engine6 = MeTTaEngine()
engine6.add_atom("(IsA RoboticsClub Club)")

cat6 = EntityCatalogue()
cat6.sync(engine6.get_all_atoms())

# When only one club exists, "the club" should resolve uniquely
result6_intent, result6_cands, result6_unresolved = Jev.route_intent("Where is RoboticsClub", cat6, None)
check("Unique entity resolves", len(result6_cands) == 1 and result6_cands[0].canonical_id == "RoboticsClub",
      f"Got: {[c.canonical_id for c in result6_cands]}")


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Test 7: Unknown event → no invented symbol
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
print("\n═══ TEST 7: Unknown event → no invented symbol ═══")
engine7 = MeTTaEngine()
cat7 = EntityCatalogue()
cat7.sync(engine7.get_all_atoms())

result7 = trans.translate("Where is the quantum lab?", catalogue=cat7)
# With no entities in graph, regex fallback runs
# It should NOT fabricate a symbol with $rel — it should use its normal parse
check("Query about unknown doesn't crash", result7 is not None)
if isinstance(result7, MeTTaTranslation):
    check("Low confidence for unknown query", result7.confidence <= 0.85, f"Got: {result7.confidence}")


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Test 8: Teaching flow accepts unknown entities
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
print("\n═══ TEST 8: Teaching accepts unknown entities ═══")
engine8 = MeTTaEngine()
cat8 = EntityCatalogue()
cat8.sync(engine8.get_all_atoms())

result8 = trans.translate("The robotics lab is located in Building C", catalogue=cat8)
check("Teaching returns ASSERTION", isinstance(result8, MeTTaTranslation) and result8.intent == IntentType.ASSERTION,
      f"Got: {type(result8).__name__} / {getattr(result8, 'intent', 'N/A')}")
if isinstance(result8, MeTTaTranslation):
    check("Has metta_expression", bool(result8.metta_expression), f"Got: '{result8.metta_expression}'")


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Test 9: Verified unique alias → exact canonical identity
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
print("\n═══ TEST 9: Alias resolution ═══")
engine9 = MeTTaEngine()
engine9.add_atom("(IsA AISecurityHackathon Event)")
engine9.add_atom("(= (Alias SecHack) AISecurityHackathon)")

cat9 = EntityCatalogue()
cat9.sync(engine9.get_all_atoms())

resolved, collisions = cat9.resolve_exact("SecHack")
check("Alias resolves to canonical", resolved == "AISecurityHackathon", f"Got: {resolved}")

resolved2, collisions2 = cat9.resolve_exact("UnknownThing")
check("Unknown returns None", resolved2 is None and collisions2 is None)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Test 10: Persistence / ordinary query still works
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
print("\n═══ TEST 10: Ordinary query flows ═══")
engine10 = MeTTaEngine()
engine10.add_atom("(Location RoboticsLab BuildingC)")

result10 = engine10.query("(Location RoboticsLab $x)", "$x")
check("Direct query works", not result10.is_empty, f"Got: {result10.raw}")
check("Returns BuildingC", any("BuildingC" in r for r in result10.raw), f"Got: {result10.raw}")


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Test 11: Catalogue search PascalCase splitting
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
print("\n═══ TEST 11: Catalogue PascalCase search ═══")
engine11 = MeTTaEngine()
engine11.add_atom("(IsA AISecurityHackathon Event)")
engine11.add_atom("(IsA AIHackathon2024 Event)")

cat11 = EntityCatalogue()
cat11.sync(engine11.get_all_atoms())

results11 = cat11.search("hackathon")
check("'hackathon' finds both events", len(results11) >= 2, 
      f"Got {len(results11)}: {[c.canonical_id for c in results11]}")

results11b = cat11.search("security")
check("'security' finds AISecurityHackathon", 
      any(c.canonical_id == "AISecurityHackathon" for c in results11b),
      f"Got: {[c.canonical_id for c in results11b]}")


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Test 12: Latency measurement
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
print("\n═══ TEST 12: Latency ═══")
engine_lat = MeTTaEngine()
for i in range(100):
    engine_lat.add_atom(f"(Fact Entity{i} Value{i})")
engine_lat.add_atom("(= (CanAccess $user $lab) (, (RequiresClearance $lab $lvl) (HasClearance $user $lvl)))")
engine_lat.add_atom("(RequiresClearance LabX Level2)")
engine_lat.add_atom("(HasClearance TestUser Level2)")

cat_lat = EntityCatalogue()

t0 = time.perf_counter()
cat_lat.sync(engine_lat.get_all_atoms())
t_sync = (time.perf_counter() - t0) * 1000

t0 = time.perf_counter()
cat_lat.search("hackathon access level")
t_search = (time.perf_counter() - t0) * 1000

t0 = time.perf_counter()
engine_lat.backward_chain("(CanAccess TestUser LabX)")
t_bc = (time.perf_counter() - t0) * 1000

print(f"  Catalogue sync (103 atoms): {t_sync:.1f} ms")
print(f"  Catalogue search: {t_search:.1f} ms")
print(f"  Backward chain: {t_bc:.1f} ms")
check("Catalogue retrieval < 50ms", t_search < 50, f"Got: {t_search:.1f} ms")
check("Total < 1000ms", t_sync + t_search + t_bc < 1000, f"Got: {t_sync + t_search + t_bc:.1f} ms")


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
print(f"\n{'═' * 50}")
print(f"  RESULTS: {PASS} passed, {FAIL} failed")
print(f"{'═' * 50}")
sys.exit(0 if FAIL == 0 else 1)
