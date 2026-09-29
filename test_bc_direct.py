import os
import sys
from dotenv import load_dotenv
load_dotenv()
from pathlib import Path
from core.metta_engine import MeTTaEngine

engine = MeTTaEngine()
engine.add_atom('(= (CanAccess  ) (, (RequiresClearance  ) (HasClearance  )))')
engine.add_atom('(RequiresClearance LabAlpha Level3)')
engine.add_atom('(HasClearance Anush Level2)')

print("TEST 1: Anush Level2")
res = engine.backward_chain("(CanAccess Anush LabAlpha)")
print(res.raw)

engine.add_atom('(HasClearance Anush Level3)')
print("TEST 2: Anush Level3")
res2 = engine.backward_chain("(CanAccess Anush LabAlpha)")
print(res2.raw)
