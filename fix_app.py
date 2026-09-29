import os
with open('app.py', 'r', encoding='utf-8') as f:
    content = f.read()

import re
old_code = '''    for entry in reversed(log[-10:]):
        op_color = "green" if entry.operation == "add" else "red"
        st.markdown(f"**{entry.timestamp}**")
        st.markdown(f"> *\\"{entry.user_input}\\"*")
        st.markdown(f":{op_color}[{entry.operation.upper()}] {entry.diff}")'''

new_code = '''    for entry in reversed(log[-10:]):
        op = entry.get("operation", "add")
        op_color = "green" if op == "add" else "red"
        st.markdown(f"**{entry.get('timestamp', '')}**")
        st.markdown(f"> *\\"{entry.get('user_input', '')}\\"*")
        st.markdown(f":{op_color}[{op.upper()}] {entry.get('diff', '')}")'''

content = content.replace(old_code, new_code)
with open('app.py', 'w', encoding='utf-8') as f:
    f.write(content)
print("app.py updated!")
