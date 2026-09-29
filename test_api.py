import os
import sys
from dotenv import load_dotenv
load_dotenv()
from core.translator import LLMTranslator

t = LLMTranslator()
print(f'API KEY: {t.api_key[0:5]}...')
print(f'Use Fallback: {t._use_fallback}')

try:
    print('Testing API...')
    resp = t._client.chat.completions.create(
        model=t.model,
        messages=[{"role": "user", "content": "Hello"}],
        max_tokens=10
    )
    print('API SUCCESS:', resp.choices[0].message.content)
except Exception as e:
    print('API FAILED:', type(e).__name__, str(e))
