import os
import json
import logging
from openai import OpenAI
from core.metta_engine import MeTTaEngine
from core.catalogue import EntityCatalogue
from core.persistence import PersistenceManager
from core.jev_adapter import JevDecisionAPI

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

# Temporal Exceptions & Recurrences (Stage D)
When the user mentions dates, recurrence, or exceptions, YOU MUST wrap the base fact in a structured temporal record:
- For a specific date: `(ValidDate (MeetingPlace RoboticsClub Lab5) "2026-10-02")`
- For a recurring event: `(Recurrence (MeetingPlace RoboticsClub Lab5) Weekly Friday)`
- For a one-time exception to a recurring event: `(Exception (MeetingPlace RoboticsClub Lab5) "2026-10-09" (MeetingPlace RoboticsClub RoomB))`
NEVER pass raw temporal text (like "next Friday") inside a normal predicate.

Do NOT fabricate knowledge. If `query_knowledge` returns no results or NO_SUPPORTING_CLAIM, tell the user you don't know and ask them to teach you.
"""

class NeuroSymbolicAgent:
    def __init__(self, engine: MeTTaEngine, catalogue: EntityCatalogue, pm: PersistenceManager):
        self.engine = engine
        self.catalogue = catalogue
        self.pm = pm
        self.jev = JevDecisionAPI()
        
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
                    "name": "propose_schema",
                    "description": "Propose a new schema predicate when you encounter a concept that cannot be expressed using the existing schema. This explicitly defines the predicate's meaning and arguments.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "predicate_name": {"type": "string", "description": "PascalCase name of the new predicate (e.g. Loves)"},
                            "argument_types": {"type": "array", "items": {"type": "string"}, "description": "List of expected arguments (e.g. ['Person', 'Person'])"},
                            "meaning": {"type": "string", "description": "A clear definition of what this predicate represents."}
                        },
                        "required": ["predicate_name", "argument_types", "meaning"]
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
                if m1 and m1.group(1) not in ("=", ":"):
                    if m1.group(1) == "SchemaDefinition":
                        m_def = _re.match(r"^\(\s*SchemaDefinition\s+([a-zA-Z0-9_]+)\s", atom)
                        if m_def: predicates.add(m_def.group(1))
                    else:
                        predicates.add(m1.group(1))
                # Match rules like (= (CanAccess ...) ...)
                m2 = _re.match(r"^\(\=\s*\(\s*([a-zA-Z0-9_]+)\s", atom)
                if m2:
                    predicates.add(m2.group(1))
            
            if not predicates:
                return json.dumps({"status": "success", "schema": "Schema is currently empty. You may define new predicates."})
            return json.dumps({"status": "success", "schema": list(predicates)})
            
        elif name == "propose_schema":
            pred = args.get("predicate_name")
            args_types = args.get("argument_types", [])
            meaning = args.get("meaning")
            
            if not pred or not pred.istitle() or not pred.isalnum():
                return json.dumps({"status": "error", "message": "Predicate must be valid PascalCase."})
                
            # Write a SchemaDefinition atom to the DB
            types_str = " ".join(args_types) if args_types else "Entity"
            schema_atom = f'(SchemaDefinition {pred} "{types_str}" "{meaning}")'
            
            import hashlib
            op_id = "op_schema_" + hashlib.sha256(f"{user_input}:{schema_atom}".encode()).hexdigest()[:12]
            
            if self.pm.operation_exists(op_id):
                return json.dumps({"status": "success", "message": f"Duplicate schema operation skipped."})
                
            self.engine.add_atom(schema_atom)
            self.pm.record_modification(self.engine, user_input, schema_atom, "add", op_id=op_id)
            
            return json.dumps({
                "status": "success", 
                "message": f"Successfully registered new predicate '{pred}'. You may now use it in learn_fact."
            })
            
        elif name == "query_knowledge":
            query = args.get("metta_query", "")
            template = query
            
            result = self.engine.query(query, template)
            used_bc = False
            if result.is_empty:
                import re as _re
                bc_expr = query
                m = _re.match(r"\(match\s+&self\s+(.+?)\s+\$\w+\)$", bc_expr)
                if m:
                    bc_expr = m.group(1)
                result = self.engine.backward_chain(bc_expr)
                used_bc = True
                
            if result.is_empty:
                return json.dumps({"status": "NO_SUPPORTING_CLAIM", "results": []})
            
            # Build derivation-indexed lookup for BC results
            derivation_map = {}
            if used_bc:
                for drec in result.derivations:
                    derivation_map[drec.answer] = drec
            
            evidence_packages = []
            for res_str in result.raw:
                c_id = self.pm.get_claim_by_atom(res_str)
                if c_id:
                    evidence_packages.append({
                        "status": "ANSWERED",
                        "answer_bindings": res_str,
                        "supporting_claim_ids": [c_id],
                        "origin": "reported"
                    })
                elif res_str in derivation_map:
                    drec = derivation_map[res_str]
                    premise_claim_ids = []
                    for p in drec.premises:
                        pid = self.pm.get_claim_by_atom(p)
                        if pid:
                            premise_claim_ids.append(pid)
                    evidence_packages.append({
                        "status": "ANSWERED",
                        "answer_bindings": res_str,
                        "origin": "derived",
                        "applied_rule": drec.rule_text,
                        "premise_claim_ids": premise_claim_ids,
                        "premises": drec.premises
                    })
                else:
                    # No claim ID and no derivation record = sync error
                    evidence_packages.append({
                        "status": "UNVERIFIED",
                        "answer_bindings": res_str,
                        "origin": "unknown",
                        "warning": "No claim ID or derivation record found. Possible sync error."
                    })
                
            return json.dumps({"status": "success", "evidence": evidence_packages})
            
        elif name == "learn_fact":
            atom = args.get("metta_atom", "")
            
            if not self._schema_checked:
                return json.dumps({
                    "status": "error", 
                    "message": "VALIDATION FAILED: You must call get_schema before learning a fact to prevent predicate hallucination."
                })
                
            import re as _re
            m = _re.match(r"^\(\s*([a-zA-Z0-9_]+)\s*(.*)\)$", atom)
            if not m:
                return json.dumps({"status": "error", "message": "VALIDATION FAILED: Malformed atom. Must be (Predicate Arg1 Arg2 ...)"})
            
            predicate = m.group(1)
            
            if predicate not in ("=", ":", "ValidDate", "Recurrence", "Exception"):
                known_preds = set()
                for existing in self.engine.get_all_atoms():
                    m1 = _re.match(r"^\(\s*([a-zA-Z0-9_]+)\s", existing)
                    if m1 and m1.group(1) not in ("=", ":"):
                        if m1.group(1) == "SchemaDefinition":
                            m_def = _re.match(r"^\(\s*SchemaDefinition\s+([a-zA-Z0-9_]+)\s", existing)
                            if m_def: known_preds.add(m_def.group(1))
                        else:
                            known_preds.add(m1.group(1))
                        
                if predicate not in known_preds:
                    return json.dumps({
                        "status": "error",
                        "message": f"VALIDATION FAILED: Predicate '{predicate}' is not in the approved schema. Use get_schema to see approved predicates, or propose a new schema explicitly."
                    })
            
            # Temporal boundaries: check both atom AND original utterance
            import re
            utterance_text = self.pm.get_utterance_text(user_input) or ""
            combined_text = atom + " " + utterance_text
            
            is_temporal_wrapper = predicate in ("ValidDate", "Recurrence", "Exception")
            
            if not is_temporal_wrapper and re.search(r"\b(tomorrow|next\s+\w+day|next\s+week|20\d\d|\d{1,2}/\d{1,2}/\d{2,4}|Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\b", combined_text, re.IGNORECASE):
                return json.dumps({
                    "status": "error",
                    "message": "VALIDATION FAILED: Temporal scope detected in utterance or atom, but you did not use a structured temporal wrapper. You MUST wrap the fact in (ValidDate ...), (Recurrence ...), or (Exception ...). Do not put raw dates in standard predicates."
                })
            
            # Idempotent write: generate operation ID, check for duplicate
            import hashlib
            op_id = "op_" + hashlib.sha256(f"{user_input}:{atom}:add".encode()).hexdigest()[:12]
            if self.pm.operation_exists(op_id):
                return json.dumps({"status": "success", "message": f"Duplicate operation {op_id} already committed. Skipped."})
                
            self.engine.add_atom(atom)
            self.catalogue.sync(self.engine.get_all_atoms())
            self.pm.record_modification(self.engine, user_input, atom, "add", op_id=op_id)
            
            self._schema_checked = False
            return json.dumps({"status": "success", "message": f"Learned {atom}", "operation_id": op_id})
            
        elif name == "retract_fact":
            atom = args.get("metta_atom", "")
            self.engine.remove_atom(atom)
            self.catalogue.sync(self.engine.get_all_atoms())
            self.pm.record_modification(self.engine, user_input, atom, "remove")
            return json.dumps({"status": "success", "message": f"Retracted {atom}"})
            
        return json.dumps({"status": "error", "message": "Unknown tool"})

    def chat(self, messages: list, user_input: str) -> str:
        """
        Executes the conversational loop with Jev intent routing and tool calling.
        messages should include the system prompt and conversation history.
        """
        # --- STAGE C: Jev API & Clarification State Machine ---
        
        # 1. Check for pending clarification
        pending = self.pm.get_pending_clarification()
        if pending:
            # Combine the original ambiguous text with the user's clarification
            combined_input = pending["original_text"] + " [Clarification: " + user_input + "]"
            self.pm.resolve_clarification(pending["id"])
            logger.info(f"Resolved clarification {pending['id']} with combined input: {combined_input}")
            eval_input = combined_input
            # Add to messages so Gemini sees context
            messages.append({"role": "user", "content": eval_input})
            u_id = self.pm.record_utterance(eval_input)
        else:
            eval_input = user_input
            messages.append({"role": "user", "content": eval_input})
            u_id = self.pm.record_utterance(eval_input)

        # 2. Fast Intent Routing via Jev
        intent = self.jev.route_intent(eval_input)
        
        if intent == "clarification":
            # State Machine: Short-circuit LLM, go into PendingClarification
            self.pm.record_pending_clarification(eval_input)
            response_text = "I'm not quite sure what you mean. Could you clarify?"
            messages.append({"role": "assistant", "content": response_text})
            return response_text
            
        # Optional: Inject the Jev intent into the prompt to guide Gemini
        messages.append({
            "role": "system", 
            "content": f"[JEV ROUTER] The user intent was classified as: {intent.upper()}"
        })
        
        # 3. Slow Brain (Gemini) executes with tools
        for _ in range(5):
            try:
                response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                tools=self.tools,
                tool_choice="auto",
                temperature=0.3
            )
            except Exception as e:
                logger.error(f"LLM API Error: {e}")
                return f"My brain just disconnected (API Error). Can we try that again?"
            
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
