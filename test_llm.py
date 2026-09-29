import sys
from dotenv import load_dotenv
load_dotenv()
from core.translator import LLMTranslator
from core.catalogue import EntityCatalogue
from core.metta_engine import MeTTaEngine
import json

engine = MeTTaEngine()
engine.add_atom("(IsA AISecurityHackathon Event)")
engine.add_atom("(IsA AIHackathon2024 Event)")
engine.add_atom("(= (Alias SecHack) AISecurityHackathon)")
cat = EntityCatalogue()
cat.sync(engine.get_all_atoms())

t = LLMTranslator()

from core.translator import ClarificationPayload
pending = ClarificationPayload(
    message="Which one?",
    pending_request="Where is the hackathon?",
    candidates=["AISecurityHackathon", "AIHackathon2024"],
    unresolved_slots=["target"]
)

print('TESTING: the first one')
try:
    res = t.translate('the first one', cat, pending)
    print(res.intent)
    if hasattr(res, 'metta_expression'): print(res.metta_expression)
    if hasattr(res, 'missing_concepts'): print(res.missing_concepts)
except Exception as e:
    print('ERROR:', e)

print('\nTESTING: Where is SecHack?')
try:
    res2 = t.translate('Where is SecHack?', cat, None)
    print(res2.intent)
    if hasattr(res2, 'metta_expression'): print(res2.metta_expression)
    if hasattr(res2, 'missing_concepts'): print(res2.missing_concepts)
except Exception as e:
    print('ERROR:', e)
