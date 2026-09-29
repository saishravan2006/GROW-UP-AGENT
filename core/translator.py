"""
core/translator.py
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Neuro-Symbolic Translation Pipeline.

Implements the "Fast/Slow" Neuro-Symbolic Loop:
  • Step 1: Jev (Fast Intent Router) categorizes user input instantly.
  • Step 2: OpenRouter (DeepSeek R1 / Qwen3 Coder) acts as a slow MeTTa code compiler.
  • Step 3: Jev (Syntax Guardrail) strictly validates the S-expression output.

Converts between natural language and MeTTa S-expressions.
"""

from __future__ import annotations

import json
import logging
import os
import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Pydantic Schemas for Structured Metadata
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class IntentType(str, Enum):
    ASSERTION = "assertion"   # User is teaching a fact (learn)
    QUERY = "query"           # User is asking a question
    RETRACTION = "retraction" # User is removing a fact
    CORRECTION = "correction" # User is correcting a fact (remove old + add new)
    CLARIFICATION = "clarification"  # Ambiguous, need more info
    CONVERSATION = "conversation"    # General chat, no KB op


class MeTTaTranslation(BaseModel):
    """Structured output from the LLM translation step."""

    intent: IntentType = Field(
        description="The detected intent of the user's input"
    )
    metta_expression: str = Field(
        default="",
        description="The MeTTa S-expression.",
    )
    query_template: str = Field(
        default="$x",
        description="The projection variable(s) for match queries.",
    )
    verbal_reasoning: str = Field(
        default="",
        description="Natural language explanation of what the agent understood",
    )
    confidence: float = Field(
        default=0.8,
        ge=0.0,
        le=1.0,
        description="Confidence score 0-1 for the translation",
    )

class ClarificationPayload(BaseModel):
    """Returned when the request is ambiguous and needs clarification."""
    intent: IntentType = Field(default=IntentType.CLARIFICATION)
    message: str = Field(description="The clarification question to ask the user")
    pending_request: str = Field(description="The original ambiguous request string")
    candidates: list[str] = Field(description="List of Candidate IDs that matched")
    resolved_slots: dict[str, str] = Field(default_factory=dict, description="Concepts that were successfully resolved")
    unresolved_slots: list[str] = Field(default_factory=list, description="Concepts that are ambiguous")


class UncertaintyPayload(BaseModel):
    """Returned when the AtomSpace has no answer."""

    message: str = Field(description="Honest admission that the information is not in the knowledge base")
    suggested_input: str = Field(default="", description="Suggested phrasing the user could provide")
    missing_concepts: list[str] = Field(default_factory=list, description="Concepts/entities the agent lacks data about")


class AnswerPayload(BaseModel):
    """Returned when the AtomSpace successfully answers a query."""

    answer: str = Field(description="The natural language answer")
    source_atoms: list[str] = Field(default_factory=list, description="The MeTTa atoms that grounded this answer")
    confidence: float = Field(default=1.0)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Jev: The Fast Front Door & Syntax Guardrail
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class Jev:
    """
    Neuro-Symbolic Decision Engine powered by Typesafe AI's System One model.

    Uses the real Jev API when TYPESAFE_API_KEY is set:
      • Step 1 (Intent Router):  Choice primitive — classifies user input into
        one of 6 intent categories in a single non-autoregressive forward pass.
      • Step 3 (Syntax Guardrail):  Noul primitive — returns a yes/no probability
        that the compiler output is a valid, balanced S-expression.

    Falls back to local heuristics when no API key is available so the system
    never breaks even without network access.
    """

    _client: Any = None
    _available: bool = False

    @classmethod
    def _ensure_client(cls) -> bool:
        """Lazy-init the OpenRouter client for Jev exactly once."""
        if cls._client is not None:
            return cls._available

        api_key = os.environ.get("OPENROUTER_API_KEY") or os.environ.get("TYPESAFE_API_KEY", "")
        if not api_key or "your-" in api_key:
            logger.info("Jev: No OPENROUTER_API_KEY — using local heuristic fallback")
            cls._available = False
            return False

        try:
            # We will use requests to hit the OpenRouter alpha decisions endpoint
            cls._client = api_key 
            cls._available = True
            logger.info("Jev: OpenRouter (jev-1.13) engine online ✓")
        except Exception as e:
            logger.warning("Jev: Failed to init OpenRouter client (%s) — local fallback", e)
            cls._available = False
        return cls._available

    # ── Step 1: Intent Router ──────────────────────────────────

    @classmethod
    def route_intent(cls, text: str, catalogue, pending) -> tuple[IntentType, list, list]:
        """
        Step 1: Returns IntentType, list of resolved Candidates, list of unresolved canonical strings.
        """
        if pending and pending.candidates:
            lower = text.lower().strip()
            # Handle "the second one", "the first one", etc.
            resolved = None
            if "first" in lower or lower == "1":
                resolved = [c for c in catalogue.candidates.values() if c.canonical_id == pending.candidates[0]]
            elif "second" in lower or lower == "2":
                if len(pending.candidates) > 1:
                    resolved = [c for c in catalogue.candidates.values() if c.canonical_id == pending.candidates[1]]
            else:
                for c in pending.candidates:
                    if c.lower() in lower:
                        resolved = [cand for cand in catalogue.candidates.values() if cand.canonical_id == c]
                        break
            if resolved:
                orig_intent = cls._local_route_intent(pending.pending_request)
                return orig_intent, resolved, []

        intent = cls._local_route_intent(text)
        if intent in (IntentType.CONVERSATION, IntentType.CLARIFICATION):
            return intent, [], []

        # Assertions (teaching) accept unknown entities freely
        if intent == IntentType.ASSERTION:
            return intent, [], []

        # Find unresolved candidates
        resolved = []
        unresolved = []
        
        if catalogue:
            candidates = catalogue.search(text)
            words = set(re.sub(r'[^\w\s]', '', text.lower()).split())
            
            for c in candidates:
                if c.display_label.lower() in text.lower() or c.canonical_id.lower() in text.lower() or any(a.lower() in text.lower() for a in c.aliases):
                    resolved.append(c)
                    
            for w in words:
                if len(w) > 4:
                    matches = [c for c in candidates if w in c.canonical_id.lower() or w in c.display_label.lower() or any(w in a.lower() for a in c.aliases)]
                    if len(matches) > 1 and not any(m in resolved for m in matches):
                        unresolved.extend([m.canonical_id for m in matches])

        if unresolved:
            return IntentType.CLARIFICATION, [], list(set(unresolved))
            
        return intent, resolved, []

    @classmethod
    def _jev_route_intent(cls, text: str) -> IntentType:
        """Hit Jev via OpenRouter alpha decisions endpoint."""
        import requests
        
        headers = {
            "Authorization": f"Bearer {cls._client}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://super-memory.local",
            "X-Title": "Super Memory"
        }
        
        payload = {
            "model": "typesafe/jev-1.13",
            "state": text,
            "questions": {
                "intent": {
                    "type": "choice",
                    "instructions": (
                        "Classify this user input for a knowledge-base agent. "
                        "The user is either teaching a fact, asking a question, "
                        "correcting old info, retracting info, requesting "
                        "clarification, or just chatting."
                    ),
                    "criteria": {
                        "assertion": "User is teaching or stating a fact, rule, or relationship",
                        "query": "User is asking a question or requesting information",
                        "correction": "User is updating or correcting a previously stated fact",
                        "retraction": "User wants to remove or forget a fact",
                        "clarification": "Input is ambiguous and needs more context",
                        "conversation": "General chat unrelated to knowledge operations",
                    }
                }
            }
        }
        
        response = requests.post("https://openrouter.ai/api/alpha/decisions", headers=headers, json=payload)
        response.raise_for_status()
        data = response.json()
        
        choice_val = data["answers"]["intent"]["choice"]
        logger.info("Jev (OpenRouter) routed intent: %s (conf=%.3f)", 
                    choice_val, data["answers"]["intent"]["confidence"])

        mapping = {
            "assertion": IntentType.ASSERTION,
            "query": IntentType.QUERY,
            "correction": IntentType.CORRECTION,
            "retraction": IntentType.RETRACTION,
            "clarification": IntentType.CLARIFICATION,
            "conversation": IntentType.CONVERSATION,
        }
        return mapping.get(choice_val, IntentType.CONVERSATION)

    @staticmethod
    def _local_route_intent(text: str) -> IntentType:
        """Local heuristic fallback — zero network, sub-millisecond."""
        lower = text.lower().strip()

        # Query detection
        if any(lower.startswith(q) for q in (
            "where", "what", "who", "when", "how", "which",
            "is there", "are there", "does", "do ", "can ",
        )) or "?" in text:
            return IntentType.QUERY

        # Correction detection
        if any(kw in lower for kw in (
            "correction", "wrong", "actually", "delayed to", "moved to", "changed to"
        )):
            return IntentType.CORRECTION

        # Retraction detection
        if any(kw in lower for kw in (
            "forget", "remove", "delete", "retract", "no longer"
        )):
            return IntentType.RETRACTION

        # Too short to be a real fact
        if len(lower.split()) < 2:
            return IntentType.CONVERSATION

        return IntentType.ASSERTION

    # ── Step 3: Syntax Guardrail ───────────────────────────────

    @classmethod
    def validate_syntax(cls, output: str) -> bool:
        """
        Step 3: 'Noul' (Yes/No) evaluation.
        Uses Typesafe Jev when available; local algorithm otherwise.
        Always runs the local check first (it's free), then optionally
        cross-validates with Jev for borderline cases.
        """
        local_ok = cls._local_validate_syntax(output)

        # If local check already rejects it, no need to burn an API call
        if not local_ok:
            return False

        # If Jev is available, cross-validate for extra safety
        if cls._ensure_client():
            try:
                return cls._jev_validate_syntax(output)
            except Exception as e:
                logger.warning("Jev API syntax validation failed (%s), using local result", e)

        return local_ok

    @classmethod
    def _jev_validate_syntax(cls, output: str) -> bool:
        """Hit Jev via OpenRouter with a Noul check."""
        import requests
        
        headers = {
            "Authorization": f"Bearer {cls._client}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://super-memory.local",
            "X-Title": "Super Memory"
        }
        
        payload = {
            "model": "typesafe/jev-1.13",
            "state": output,
            "questions": {
                "valid_sexpr": {
                    "type": "noul",
                    "instructions": (
                        "Is this output a valid MeTTa S-expression? "
                        "It must start with '(' or '!', end with ')', "
                        "have perfectly balanced parentheses, contain no "
                        "conversational text, and not be empty. "
                        "Return yes only if it is strictly a syntactically "
                        "correct S-expression."
                    )
                }
            }
        }
        
        response = requests.post("https://openrouter.ai/api/alpha/decisions", headers=headers, json=payload)
        response.raise_for_status()
        data = response.json()

        prob = data["answers"]["valid_sexpr"]["noul"]
        is_valid = prob >= 0.5
        logger.info("Jev (OpenRouter) syntax check: valid=%s (p=%.3f)", is_valid, prob)

        # Trust Jev's judgment if confidence is high; otherwise fall back to local
        if prob >= 0.7 or prob <= 0.3:
            return is_valid
        return cls._local_validate_syntax(output)

    @staticmethod
    def _local_validate_syntax(output: str) -> bool:
        """Local algorithmic syntax validation — zero network, deterministic."""
        cleaned = output.strip()

        # Must start with ( or ! and end with )
        if not (cleaned.startswith("(") or cleaned.startswith("!")) or not cleaned.endswith(")"):
            return False

        # Check balanced parentheses
        depth = 0
        for ch in cleaned:
            if ch == "(":
                depth += 1
            elif ch == ")":
                depth -= 1
            if depth < 0:
                return False

        # Must be completely balanced and not empty
        return depth == 0 and len(cleaned) > 2


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# LLM Translation Client (DeepSeek R1 / OpenRouter)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

COMPILER_SYSTEM_PROMPT = """\
You are a strict MeTTa code compiler. Your ONLY job is to convert natural language into formal MeTTa logic.

## Examples
User: "The Potheri Lake cleanup drive starts at 8 AM."
Intent is 'assertion'. Output ONLY:
(Time PotheriLakeCleanup 0800)

User: "The environmental club's lake cleaning activity is the exact same event as the Potheri Lake cleanup."
Intent is 'assertion' but dictates an equivalence. Output ONLY:
(= (EnvironmentalClubLakeCleaning) (PotheriLakeCleanup))

User: "Actually, the environmental club's lake cleaning activity has been delayed to 9 AM."
Intent is 'correction'. Output BOTH remove and add exactly like this:
!(remove-atom &self (Time EnvironmentalClubLakeCleaning 0800))
!(add-atom &self (Time EnvironmentalClubLakeCleaning 0900))

User: "What time do I need to show up for the Potheri Lake cleanup drive?"
Intent is 'query'. Output ONLY:
(Time PotheriLakeCleanup $time)

User: "Who is the manager of the location where the biosensor kits are stored?"
Intent is 'query'. Output a chained query ONLY:
(, (Location BiosensorKits $loc) (Manager $loc $mgr))

User: "A user can access a lab if the lab requires a specific clearance level and the user has that exact clearance level."
Output ONLY:
(= (CanAccess $user $lab) (, (RequiresClearance $lab $lvl) (HasClearance $user $lvl)))

User: "A system is secure if the firewall is active and the port is closed."
Output ONLY:
(= (IsSecure $system) (, (IsActive Firewall) (IsClosed Port)))

## Rules
1. Use PascalCase for entity names and predicates (no spaces, no quotes).
2. Prefer standard predicates like "Location" or "Time".
3. Variables start with $.
4. Keep expressions FLAT.
5. NEVER include 'match &self' in queries.
6. For conditional logic ("If X and Y, then Z"), ALWAYS unpack it using equivalence and chained variables: (= (Z_Predicate $var) (, (X_Predicate $var) (Y_Predicate $var))). Break down complex sentences into individual logical relations. DO NOT combine them into massive single strings.

CRITICAL: Output ONLY the raw MeTTa S-expression(s). Do NOT output conversational text. Do NOT use markdown.
"""

ANSWER_SYSTEM_PROMPT = """\
You are the verbal interface for a knowledge agent backed by a MeTTa AtomSpace.
Given the user's question and the query results from the AtomSpace, produce
a clear, natural language answer. Ground your answer ONLY in the provided results.
If results are empty, say you don't have that information — never hallucinate.

Respond ONLY with valid JSON matching the schema provided.
"""


@dataclass
class LLMTranslator:
    """
    Implements the fast/slow loop using Jev + OpenRouter (DeepSeek R1).
    Falls back to regex-based heuristics if API fails or is unset.
    """

    api_key: str = field(default="")
    base_url: str = field(default="https://generativelanguage.googleapis.com/v1beta/openai/")
    model: str = field(default="gemini-2.5-pro")
    _client: Any = field(init=False, repr=False, default=None)
    _use_fallback: bool = field(init=False, default=False)

    def __post_init__(self) -> None:
        self.api_key = self.api_key or os.environ.get("GEMINI_API_KEY") or os.environ.get("OPENAI_API_KEY", "")
        self.base_url = os.environ.get("OPENAI_BASE_URL", self.base_url)
        self.model = os.environ.get("OPENAI_MODEL", self.model)

        if self.api_key and "your-" not in self.api_key:
            try:
                from openai import OpenAI
                self._client = OpenAI(
                    api_key=self.api_key,
                    base_url=self.base_url,
                )
                self._use_fallback = False
                logger.info(
                    "LLM compiler online → %s @ %s",
                    self.model,
                    self.base_url,
                )
            except ImportError:
                logger.warning("openai package not installed, using fallback")
                self._use_fallback = True
        else:
            logger.info("No valid API key set — using regex fallback translator")
            self._use_fallback = True

    # ── primary translation ──

    def translate(self, user_input: str, catalogue=None, pending=None):
        """Execute the Jev Fast/Slow Translation loop with input normalization."""
        clean_input = user_input.strip(' "\'').replace('?', '').strip()

        intent, resolved_cands, unresolved_strings = Jev.route_intent(clean_input, catalogue, pending)

        if intent == IntentType.CLARIFICATION:
            original_req = pending.pending_request if pending else clean_input
            return ClarificationPayload(
                message="Which one do you mean?",
                pending_request=original_req,
                candidates=unresolved_strings,
                unresolved_slots=["target"]
            )
            
        if intent == IntentType.CONVERSATION:
            return MeTTaTranslation(intent=intent, verbal_reasoning="I am listening.", confidence=1.0)

        # Let the compiler handle query if we have no unresolved but also no resolved
        if self._use_fallback:
            return self._fallback_translate(clean_input, intent, resolved_cands)
        return self._llm_translate(clean_input, intent, resolved_cands, pending)

    def generate_answer(
        self,
        question: str,
        query_results: list[str],
        source_atoms: list[str] | None = None,
    ) -> AnswerPayload | UncertaintyPayload:
        if not query_results:
            return self._generate_uncertainty(question)

        if self._use_fallback:
            return self._fallback_answer(question, query_results, source_atoms)
        return self._llm_answer(question, query_results, source_atoms)

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # DeepSeek R1 / OpenRouter Loop
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    def _llm_translate(self, user_input: str, intent: IntentType, resolved_cands: list, pending) -> MeTTaTranslation | UncertaintyPayload:
        """
        The "Fast/Slow" Neuro-Symbolic Loop.
        """


        # Step 2: OpenRouter / DeepSeek R1 (The Slow Translator acting as compiler)
        prompt = (
            f"Intent Category: {intent.value}\n"
            f"User input text: {pending.pending_request if pending else user_input}\n"
        )
        if resolved_cands:
            symbols = ", ".join([c.canonical_id for c in resolved_cands])
            prompt += f"Use ONLY these verified graph symbols: {symbols}\n"
        prompt += "\nOutput ONLY the corresponding MeTTa code enclosed in parentheses."

        max_retries = 2
        for attempt in range(max_retries + 1):
            try:
                response = self._client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": COMPILER_SYSTEM_PROMPT + "\nYou are provided with verified graph symbols. You MUST use these exact symbols. Do not invent symbols for known entities."},
                        {"role": "user", "content": prompt},
                    ],
                    temperature=0.0,
                    max_tokens=256,
                )

                raw_content = response.choices[0].message.content or ""
                
                # Clean up DeepSeek R1 <think> blocks and markdown
                cleaned = re.sub(r"<think>.*?</think>", "", raw_content, flags=re.DOTALL).strip()
                cleaned = re.sub(r"^```[a-zA-Z]*\n?", "", cleaned)
                cleaned = re.sub(r"\n?```$", "", cleaned).strip()

                # Step 3: Jev (The Syntax Guardrail)
                is_valid = Jev.validate_syntax(cleaned)
                
                if is_valid:
                    # Determine query template variable heuristic
                    query_template = "$x"
                    if intent == IntentType.QUERY:
                        vars = re.findall(r"(\$[a-zA-Z0-9_]+)", cleaned)
                        if vars:
                            query_template = vars[-1]
                            
                    return MeTTaTranslation(
                        intent=intent,
                        metta_expression=cleaned,
                        query_template=query_template,
                        verbal_reasoning=f"Compiled natural language to MeTTa {intent.value}.",
                        confidence=0.95
                    )
                else:
                    logger.warning("Jev syntax guardrail rejected output (Attempt %d): %s", attempt + 1, cleaned)
                    # Loop loops back to retry via OpenRouter
                    
            except Exception as e:
                logger.error("LLM compiler call failed: %s", e)
                break
                
        # If we exhausted retries or failed completely, fallback to local Ollama Qwen 2.5 Coder
        logger.error("Primary compiler failed. Falling back to local Ollama (qwen2.5-coder:3b).")
        try:
            return self._ollama_translate(user_input, intent)
        except Exception as e:
            logger.error("Ollama fallback failed: %s. Using regex heuristic.", e)
            return self._fallback_translate(user_input, intent, resolved_cands)

    def _ollama_translate(self, user_input: str, intent: IntentType) -> MeTTaTranslation:
        """Fallback to local Ollama model."""
        prompt = (
            f"Intent Category: {intent.value}\n"
            f"User input text: {user_input}\n\n"
            f"Output ONLY the corresponding MeTTa code enclosed in parentheses."
        )
        
        from openai import OpenAI
        local_client = OpenAI(base_url="http://localhost:11434/v1", api_key="ollama")
        
        response = local_client.chat.completions.create(
            model="qwen2.5-coder:3b",
            messages=[
                {"role": "system", "content": COMPILER_SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
            temperature=0.0,
        )
        
        raw_content = response.choices[0].message.content or ""
        cleaned = re.sub(r"<think>.*?</think>", "", raw_content, flags=re.DOTALL).strip()
        cleaned = re.sub(r"^```[a-zA-Z]*\n?", "", cleaned)
        cleaned = re.sub(r"\n?```$", "", cleaned).strip()
        
        if Jev.validate_syntax(cleaned):
            query_template = "$x"
            if intent == IntentType.QUERY:
                vars = re.findall(r"(\$[a-zA-Z0-9_]+)", cleaned)
                if vars:
                    query_template = vars[-1]
                    
            return MeTTaTranslation(
                intent=intent,
                metta_expression=cleaned,
                query_template=query_template,
                verbal_reasoning=f"Compiled natural language to MeTTa {intent.value} (via Local Qwen).",
                confidence=0.90
            )
        else:
            raise ValueError(f"Ollama syntax guardrail rejected output: {cleaned}")

    def _llm_answer(
        self,
        question: str,
        query_results: list[str],
        source_atoms: list[str] | None,
    ) -> AnswerPayload:
        """Use the LLM to generate a verbal answer from query results."""
        try:
            user_msg = (
                f"Question: {question}\n"
                f"AtomSpace query results: {query_results}\n"
                f"Source atoms: {source_atoms or []}"
            )

            # Note: For answering, we still use structured outputs if the model supports it.
            # DeepSeek R1 can struggle with JSON schema, but we'll try parsing it natively.
            response = self._client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": ANSWER_SYSTEM_PROMPT},
                    {"role": "user", "content": user_msg},
                ],
                temperature=0.0,
                max_tokens=512,
            )

            content = response.choices[0].message.content or ""
            content = re.sub(r"<think>.*?</think>", "", content, flags=re.DOTALL).strip()
            content = re.sub(r"^```json\n?", "", content)
            content = re.sub(r"\n?```$", "", content).strip()
            
            data = json.loads(content)
            return AnswerPayload(**data)

        except Exception as e:
            logger.error("LLM answer generation failed: %s", e)
            return self._fallback_answer(question, query_results, source_atoms)

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # Regex Fallback (offline / no API key)
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    def _fallback_translate(self, user_input: str, intent: IntentType, resolved_cands: list) -> MeTTaTranslation | UncertaintyPayload:
        """
        Heuristic NL → MeTTa translator.
        """
        text = user_input.strip()

        if intent == IntentType.QUERY:
            return self._fallback_parse_query(text)
        elif intent == IntentType.RETRACTION:
            return self._fallback_parse_retraction(text)
        elif intent == IntentType.ASSERTION:
            return self._fallback_parse_assertion(text)
            
        return MeTTaTranslation(intent=intent, verbal_reasoning="Listening...")

    def _fallback_parse_assertion(self, text: str) -> MeTTaTranslation:
        m = re.search(r"(.+?)\s+(?:are|is)\s+(?:stored|located|kept|found)\s+(?:in|at)\s+(.+?)\.?\s*$", text, re.IGNORECASE)
        if m:
            subject = self._to_pascal(m.group(1))
            location = self._to_pascal(m.group(2))
            expr = f"(Location {subject} {location})"
            return MeTTaTranslation(intent=IntentType.ASSERTION, metta_expression=expr, verbal_reasoning="Learned location.", confidence=0.85)

        m = re.search(r"(.+?)\s+(?:is|are)\s+(.+?)\.?\s*$", text, re.IGNORECASE)
        if m:
            subject = self._to_pascal(m.group(1))
            obj = self._to_pascal(m.group(2))
            expr = f"(Is {subject} {obj})"
            return MeTTaTranslation(intent=IntentType.ASSERTION, metta_expression=expr, verbal_reasoning="Learned property.", confidence=0.7)

        m = re.search(r"(.+?)\s+(?:has|holds)\s+(.+?)\.?\s*$", text, re.IGNORECASE)
        if m:
            subject = self._to_pascal(m.group(1))
            obj = self._to_pascal(m.group(2))
            if "clearance" in text.lower():
                expr = f"(HasClearance {subject} {obj.replace('Clearance', '')})"
            else:
                expr = f"(Has {subject} {obj})"
            return MeTTaTranslation(intent=IntentType.ASSERTION, metta_expression=expr, verbal_reasoning="Learned ownership.", confidence=0.7)

        words = text.rstrip(".!").split()
        if len(words) >= 2:
            atoms = [self._to_pascal(w) for w in words[:4]]
            expr = f"(Fact {' '.join(atoms)})"
            return MeTTaTranslation(intent=IntentType.ASSERTION, metta_expression=expr, verbal_reasoning="Learned fact.", confidence=0.5)

        return MeTTaTranslation(intent=IntentType.CLARIFICATION, verbal_reasoning="Ambiguous input.")

    def _fallback_parse_query(self, text: str) -> MeTTaTranslation:
        lower = text.lower().rstrip("?").strip()

        m = re.search(r"where\s+(?:are|is)\s+(?:the\s+)?(.+?)(?:\s+stored|\s+located|\s+kept|\s+found)?$", lower, re.IGNORECASE)
        if m:
            subject = self._to_pascal(m.group(1))
            return MeTTaTranslation(intent=IntentType.QUERY, metta_expression=f"(Location {subject} $x)", query_template="$x", verbal_reasoning="Querying location...", confidence=0.85)

        m = re.search(r"what\s+(?:is|are)\s+(?:the\s+)?(.+)$", lower, re.IGNORECASE)
        if m:
            subject = self._to_pascal(m.group(1))
            return MeTTaTranslation(intent=IntentType.QUERY, metta_expression=f"(Is {subject} $x)", query_template="$x", verbal_reasoning="Querying property...", confidence=0.75)

        words = lower.split()
        keywords = [w for w in words if w not in {"where", "what", "who", "when", "how", "which", "is", "are", "the", "a", "an", "do", "does", "can", "stored", "located", "kept", "found"}]
        if keywords:
            subject = self._to_pascal(" ".join(keywords[:3]))
            return MeTTaTranslation(intent=IntentType.QUERY, metta_expression=f"($rel {subject} $x)", query_template="($rel $x)", verbal_reasoning="Querying...", confidence=0.5)

        return MeTTaTranslation(intent=IntentType.QUERY, metta_expression="($rel $subj $obj)", query_template="($rel $subj $obj)", verbal_reasoning="Broad query...", confidence=0.3)

    def _fallback_parse_retraction(self, text: str) -> MeTTaTranslation:
        cleaned = re.sub(r"^(forget|remove|delete|retract)\s+(that\s+)?", "", text.lower()).strip().rstrip(".")
        assertion = self._fallback_parse_assertion(cleaned)
        return MeTTaTranslation(intent=IntentType.RETRACTION, metta_expression=assertion.metta_expression, verbal_reasoning="Retracting fact.", confidence=assertion.confidence)

    # ── uncertainty ──

    def _generate_uncertainty(self, question: str) -> UncertaintyPayload:
        words = question.lower().split()
        stop = {"where", "what", "who", "when", "how", "which", "is", "are", "the", "a", "an", "do", "does", "can", "stored", "located", "kept", "found", "?"}
        concepts = [w for w in words if w not in stop and len(w) > 2]
        return UncertaintyPayload(
            message="I don't have information about that in my knowledge base. My AtomSpace returned no matching atoms for your query.",
            suggested_input="You can teach me by saying something like: \"[The thing] is stored in [location]\" or \"[Subject] is [property]\".",
            missing_concepts=concepts[:5],
        )

    def _fallback_answer(self, question: str, query_results: list[str], source_atoms: list[str] | None) -> AnswerPayload:
        results_str = ", ".join(r.replace("(", "").replace(")", "").strip() for r in query_results)
        return AnswerPayload(answer=f"Based on my knowledge base: {results_str}", source_atoms=source_atoms or query_results, confidence=1.0)

    # ── helpers ──

    @staticmethod
    def _to_pascal(text: str) -> str:
        cleaned = re.sub(r"\b(the|a|an)\b", "", text, flags=re.IGNORECASE)
        cleaned = re.sub(r"[^\w\s]", "", cleaned)
        words = cleaned.split()
        return "".join(w.capitalize() for w in words if w)
