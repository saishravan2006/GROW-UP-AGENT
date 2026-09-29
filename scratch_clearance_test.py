from core.metta_engine import MeTTaEngine
engine = MeTTaEngine()

engine.load_program("""
(= (CanAccess $user $lab)
   (, (RequiresClearance $lab $lvl)
      (HasClearance $user $lvl)))

(RequiresClearance LabA Level3)
(HasClearance Anush Level2)
""")

print("--- NATIVE QUERY ---")
res1 = engine.query("(CanAccess Anush LabA)", "(CanAccess Anush LabA)")
print("Native query result:", res1)

print("--- NATIVE BACKWARD CHAIN (using !(bc ...)) ---")
engine.load_program("""
(= (bc $goal) (match &self $goal $goal))
(= (bc $goal) (match &self (= $goal $body) (bc $body)))
(= (bc (, $a $b)) (let* (($res-a (bc $a)) ($res-b (bc $b))) (, $res-a $res-b)))
""")
try:
    res2 = engine.run_raw("!(bc (CanAccess Anush LabA))")
    print("bc output:", res2)
except Exception as e:
    print("bc error:", e)

print("--- PYTHON BACKWARD CHAIN ---")
res3 = engine.backward_chain("(CanAccess Anush LabA)")
print("Python backward chain result:", res3)
