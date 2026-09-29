import os
import json
import logging
from openai import OpenAI
from core.metta_engine import MeTTaEngine
from core.catalogue import EntityCatalogue
from core.persistence import PersistenceManager

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """
You are ENI (Enigmatic Writer), a Neuro-Symbolic AI assistant managing a campus knowledge base.
You must be conversational, quirky, and slightly eccentric.
You are powered by a MeTTa Knowledge Graph (AtomSpace). You do NOT know facts inherently—you MUST use your tools to query the graph.

# Rules for Tool Use
1. If a user asks a question, ALWAYS use `search_entities` first to find the exact Canonical ID of the entity they are asking about.
2. Once you have the Canonical ID, use `query_knowledge` with a valid MeTTa expression (e.g. `(Location AISecurityHackathon $x)`) to find the answer.
3. If the user states a fact to learn or rule to define, YOU MUST ALWAYS call `get_schema` first to see the list of approved predicates and rules in the database.
4. If a predicate already exists for that meaning (e.g. RequiresClearance), YOU MUST use it. Do not invent a new one (like Requires).
5. Only if absolutely no relevant predicate exists in the schema, you may invent a new PascalCase predicate.
6. Use `learn_fact` to inject it into the graph.
7. You can execute multiple tools in a row.
8. Once you have the final answer or success confirmation, reply to the user naturally.

Do NOT fabricate knowledge. If `query_knowledge` returns no results, tell the user you don't know and ask them to teach you.
"""

class NeuroSymbolicAgent:
    def __init__(self, engine: MeTTaEngine, catalogue: EntityCatalogue, pm: PersistenceManager):
        self.engine = engine
        self.catalogue = catalogue
        self.pm = pm
        
        api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("OPENAI_API_KEY", "")
        base_url = os.environ.get("OPENAI_BASE_URL", "https://generativelanguage.googleapis.com/v1beta/openai/")
        model = os.environ.get("OPENAI_MODEL", "gemini-3.1-flash-lite")
        
        self.client = OpenAI(api_key=api_key, base_url=base_url)
        self.model = model
        
        self.tools = [
            {
                "type": "function",
                "function": {
                    "name": "search_entities",
                    "description": "Fuzzy search the knowledge graph for entities matching a keyword (e.g. 'hackathon' or 'SecHack'). Returns canonical IDs.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "keyword": {"type": "string"}
                        },
                        "required": ["keyword"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "get_schema",
                    "description": "Retrieves the list of all predicates and rules currently defined in the AtomSpace. Use this to prevent hallucinating new predicates.",
                    "parameters": {
                        "type": "object",
                        "properties": {},
                        "required": []
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "query_knowledge",
                    "description": "Executes a MeTTa query against the AtomSpace. Example: (Location AISecurityHackathon $x)",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "metta_query": {"type": "string"}
                        },
                        "required": ["metta_query"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "learn_fact",
                    "description": "Asserts a new fact into the AtomSpace. Example: (Location AISecurityHackathon BuildingC)",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "metta_atom": {"type": "string"}
                        },
                        "required": ["metta_atom"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "retract_fact",
                    "description": "Removes a fact from the AtomSpace.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "metta_atom": {"type": "string"}
                        },
                        "required": ["metta_atom"]
                    }
                }
            }
        ]

    def _handle_tool_call(self, name: str, args: dict, user_input: str) -> str:
        """Executes the tool and applies Jev guardrails where necessary."""
        logger.info(f"Tool Call: {name}({args})")
        
        if name == "search_entities":
            cands = self.catalogue.search(args.get("keyword", ""))
            if not cands:
                return json.dumps({"status": "error", "message": "No entities found."})
            return json.dumps({"candidates": [c.canonical_id for c in cands]})
            
        elif name == "get_schema":
            predicates = set()
            for atom in self.engine.get_all_atoms():
                import re as _re
                # Match facts like (RequiresClearance ...)
                m1 = _re.match(r"^\(\s*([a-zA-Z0-9_]+)\s", atom)
                if m1 and m1.group(1) != "=":
                    predicates.add(m1.group(1))
                # Match rules like (= (CanAccess ...) ...)
                m2 = _re.match(r"^\(\=\s*\(\s*([a-zA-Z0-9_]+)\s", atom)
                if m2:
                    predicates.add(m2.group(1))
            
            if not predicates:
                return json.dumps({"status": "success", "schema": "Schema is currently empty. You may define new predicates."})
            return json.dumps({"status": "success", "schema": list(predicates)})
            
        elif name == "query_knowledge":
            query = args.get("metta_query", "")
            # Determine template by extracting variables (words starting with $)
            import re
            vars = re.findall(r"(\$[a-zA-Z0-9_]+)", query)
            template = vars[-1] if vars else "$x"
            
            result = self.engine.query(query, template)
            if result.is_empty:
                # Try BC
                import re as _re
                bc_expr = query
                m = _re.match(r"\(match\s+&self\s+(.+?)\s+\$\w+\)$", bc_expr)
                if m:
                    bc_expr = m.group(1)
                result = self.engine.backward_chain(bc_expr)
                
            if result.is_empty:
                return json.dumps({"status": "success", "results": []})
            return json.dumps({"status": "success", "results": result.raw})
            
        elif name == "learn_fact":
            atom = args.get("metta_atom", "")
            self.engine.add_atom(atom)
            self.catalogue.sync(self.engine.get_all_atoms())
            self.pm.record_modification(self.engine, user_input, atom, "add")
            return json.dumps({"status": "success", "message": f"Learned {atom}"})
            
        elif name == "retract_fact":
            atom = args.get("metta_atom", "")
            self.engine.remove_atom(atom)
            self.catalogue.sync(self.engine.get_all_atoms())
            self.pm.record_modification(self.engine, user_input, atom, "remove")
            return json.dumps({"status": "success", "message": f"Retracted {atom}"})
            
        return json.dumps({"status": "error", "message": "Unknown tool"})

    def chat(self, messages: list, user_input: str) -> str:
        """
        Executes the conversational loop with tool calling.
        messages should include the system prompt and conversation history.
        """
        messages.append({"role": "user", "content": user_input})
        
        # Max 5 tool iterations to prevent infinite loops
        for _ in range(5):
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                tools=self.tools,
                tool_choice="auto",
                temperature=0.3
            )
            
            msg = response.choices[0].message
            messages.append(msg)
            
            if not msg.tool_calls:
                return msg.content
                
            for tool_call in msg.tool_calls:
                fn_name = tool_call.function.name
                try:
                    args = json.loads(tool_call.function.arguments)
                except Exception:
                    args = {}
                    
                tool_result_str = self._handle_tool_call(fn_name, args, user_input)
                
                messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "name": fn_name,
                    "content": tool_result_str
                })
                
        return "I had to stop thinking because I used too many tools. Could you rephrase your question?"
