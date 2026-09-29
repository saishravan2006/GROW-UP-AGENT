from core.metta_engine import MeTTaEngine
engine = MeTTaEngine()
engine.add_atom('(= (CanAccess  ) (, (RequiresClearance  ) (HasClearance  )))')
engine.add_atom('(RequiresClearance LabAlpha Level3)')
engine.add_atom('(HasClearance Anush Level2)')
engine.add_atom('(HasClearance Anush Level3)')

import logging
logging.basicConfig(level=logging.DEBUG)

res = engine.backward_chain("(CanAccess Anush LabAlpha)")
print("RESULT:", res.raw)
