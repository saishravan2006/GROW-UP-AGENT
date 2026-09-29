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
1. ALWAYS use `search_entities` first to find the exact Canonical ID of any entity the user mentions.
2. YOU MUST ALWAYS call `get_schema` BEFORE querying or learning facts to see the list of approved predicates in the database.
3. If a predicate already exists for that meaning (e.g. MeetingPlace), YOU MUST use it. Do not invent a new one (like Location).
4. Only if absolutely no relevant predicate exists in the schema, you may invent a new PascalCase predicate when learning.
5. Use `query_knowledge` with the approved predicate to find answers.
6. Use `learn_fact` with the approved predicate to inject knowledge into the graph.
7. You can execute multiple tools in a row.
8. When answering a question based on `query_knowledge`, YOU MUST strictly limit your answer to the returned `answer_bindings`. Do NOT embellish, infer, or hallucinate any additional claims beyond the exact evidence provided.

Do NOT fabricate knowledge. If `query_knowledge` returns no results or NO_SUPPORTING_CLAIM, tell the user you don't know and ask them to teach you.
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
        self._schema_checked = False
        
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
            self._schema_checked = True
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
            # Use the query itself as the template to get the fully bound atom back
            template = query
            
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
                return json.dumps({
                    "status": "NO_SUPPORTING_CLAIM", 
                    "results": []
                })
            
            # Map results back to claim IDs
            evidence_packages = []
            for res_str in result.raw:
                c_id = self.pm.get_claim_by_atom(res_str)
                evidence_packages.append({
                    "status": "ANSWERED",
                    "answer_bindings": res_str,
                    "supporting_claim_ids": [c_id] if c_id else [],
                    "origin": "reported" if c_id else "derived"
                })
                
            return json.dumps({"status": "success", "evidence": evidence_packages})
            
        elif name == "learn_fact":
            atom = args.get("metta_atom", "")
            
            if not self._schema_checked:
                return json.dumps({
                    "status": "error", 
                    "message": "VALIDATION FAILED: You must call get_schema before learning a fact to prevent predicate hallucination."
                })
                
            # Additional validation: parse and validate predicate
            import re as _re
            m = _re.match(r"^\(\s*([a-zA-Z0-9_]+)\s*(.*)\)$", atom)
            if not m:
                return json.dumps({"status": "error", "message": "VALIDATION FAILED: Malformed atom. Must be (Predicate Arg1 Arg2 ...)"})
            
            predicate = m.group(1)
            
            # Allow definitions (rules or schema creations)
            if predicate == "=" or predicate == ":":
                pass 
            else:
                # Check if predicate is in known schema
                known_preds = set()
                for existing in self.engine.get_all_atoms():
                    m1 = _re.match(r"^\(\s*([a-zA-Z0-9_]+)\s", existing)
                    if m1 and m1.group(1) not in ["=", ":"]:
                        known_preds.add(m1.group(1))
                        
                if predicate not in known_preds:
                    if not predicate[0].isupper():
                        return json.dumps({
                            "status": "error",
                            "message": f"VALIDATION FAILED: Predicate '{predicate}' is unknown and not PascalCase. You cannot invent this."
                        })
            
            # Temporal boundaries: Stage A/B prevents specific unstructured strings
            import re
            if re.search(r"\b(tomorrow|next|20\d\d|\d{1,2}/\d{1,2}/\d{2,4}|Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\b", atom, re.IGNORECASE):
                return json.dumps({
                    "status": "error",
                    "message": "VALIDATION FAILED: Unstructured temporal exceptions (dates, tomorrow, next) are currently unsupported in Stage A/B."
                })
                
            self.engine.add_atom(atom)
            self.catalogue.sync(self.engine.get_all_atoms())
            # Here user_input contains utterance_id
            self.pm.record_modification(self.engine, user_input, atom, "add")
            
            # Reset after successful write
            self._schema_checked = False
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
        # Record the exact user utterance in the evidence ledger
        u_id = self.pm.record_utterance(user_input)
        
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
                    
                tool_result_str = self._handle_tool_call(fn_name, args, u_id)
                
                messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "name": fn_name,
                    "content": tool_result_str
                })
                
        return "I had to stop thinking because I used too many tools. Could you rephrase your question?"
