import os
from dotenv import load_dotenv
load_dotenv()
from core.translator import LLMTranslator
t = LLMTranslator()
res = t.translate('An agent is compromised if their handler is captured and they are stationed in a hostile zone.')
print('--- LLM TRANSLATION ---')
print('Intent:', res.intent)
print('Expression:', res.metta_expression)
print('Query Template:', res.query_template)
print('Verbal:', res.verbal_reasoning)
