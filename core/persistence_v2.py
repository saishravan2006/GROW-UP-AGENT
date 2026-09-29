"""
core/persistence.py
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Omega-style State Engine.

Handles:
  • AtomSpace serialisation to disk  (.metta files)
  • Session reload from persisted state
  • Append-only audit ledger  (audit_log.jsonl)
  • State hashing for integrity verification
  • Git-style unified diff generation  (before ↔ after)
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from difflib import unified_diff
from pathlib import Path
from typing import Any

from core.metta_engine import MeTTaEngine

logger = logging.getLogger(__name__)

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Audit Log Entry
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

@dataclass
class AuditEntry:
    """Single knowledge-modification event."""
    timestamp: str
    session_id: str
    natural_language_input: str
    extracted_metta_rule: str
    before_state_hash: str
    after_state_hash: str
    diff: str
    operation: str = "add"  # "add" | "remove" | "modify"

    def to_dict(self) -> dict[str, Any]:
        return {
            "timestamp": self.timestamp,
            "session_id": self.session_id,
            "natural_language_input": self.natural_language_input,
            "extracted_metta_rule": self.extracted_metta_rule,
            "before_state_hash": self.before_state_hash,
            "after_state_hash": self.after_state_hash,
            "diff": self.diff,
            "operation": self.operation,
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Diff Generator
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def compute_state_hash(snapshot: str) -> str:
    """SHA-256 of the normalised AtomSpace snapshot."""
    return hashlib.sha256(snapshot.encode("utf-8")).hexdigest()[:16]


def generate_diff(before: str, after: str) -> str:
    """
    Produce a Git-style unified diff between two AtomSpace snapshots.

    Returns a human-readable string showing added (+) and removed (-)
    atoms with contextual headers.
    """
    before_lines = before.splitlines(keepends=True)
    after_lines = after.splitlines(keepends=True)

    diff_lines = list(
        unified_diff(
            before_lines,
            after_lines,
            fromfile="AtomSpace (before)",
            tofile="AtomSpace (after)",
            lineterm="",
        )
    )

    if not diff_lines:
        return "(no changes)"

    return "\n".join(diff_lines)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Persistence Manager
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

@dataclass
class PersistenceManager:
    """
    Manages the full persistence lifecycle:
      1. Save / Load the AtomSpace to/from a .metta file
      2. Write append-only audit log entries
      3. Track session identity
      4. Compute and expose diffs
    """

    data_dir: Path = field(default_factory=lambda: Path("data"))
    session_id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    _last_snapshot: str = field(init=False, default="")
    _last_hash: str = field(init=False, default="")

    def __post_init__(self) -> None:
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self._last_snapshot = ""
        self._last_hash = compute_state_hash("")

    # ── paths ──

    @property
    def kb_path(self) -> Path:
        return self.data_dir / "knowledge_base.metta"

    @property
    def audit_path(self) -> Path:
        return self.data_dir / "audit_log.jsonl"

    @property
    def meta_path(self) -> Path:
        return self.data_dir / "session_meta.json"

    # ── knowledge base I/O ──

    def save_state(self, engine: MeTTaEngine) -> str:
        """
        Persist the current AtomSpace to disk.
        Returns the new state hash.
        """
        snapshot = engine.get_state_snapshot()
        self.kb_path.write_text(
            f"; Super Memory Knowledge Base\n"
            f"; Saved: {datetime.now(timezone.utc).isoformat()}\n"
            f"; Session: {self.session_id}\n\n"
            + "\n".join(
                atom for atom in sorted(engine.get_all_atoms())
            )
            + "\n",
            encoding="utf-8",
        )

        new_hash = compute_state_hash(snapshot)
        self._last_snapshot = snapshot
        self._last_hash = new_hash

        # Also save session metadata
        self.meta_path.write_text(
            json.dumps(
                {
                    "session_id": self.session_id,
                    "last_save": datetime.now(timezone.utc).isoformat(),
                    "state_hash": new_hash,
                    "atom_count": len(engine.get_all_atoms()),
                },
                indent=2,
            ),
            encoding="utf-8",
        )

        logger.info("State saved → %s  (hash: %s)", self.kb_path, new_hash)
        return new_hash

    def load_state(self, engine: MeTTaEngine) -> bool:
        """
        Reload a previously persisted AtomSpace from disk.
        Returns True if a saved state was found and loaded.
        """
        if not self.kb_path.exists():
            logger.info("No persisted state found at %s", self.kb_path)
            return False

        content = self.kb_path.read_text(encoding="utf-8")

        # Filter out comment lines
        lines = [
            line.strip()
            for line in content.splitlines()
            if line.strip() and not line.strip().startswith(";")
        ]

        if not lines:
            logger.info("Persisted state is empty")
            return False

        # Load each atom into the engine
        for line in lines:
            try:
                engine.add_atom(line)
            except Exception as e:
                logger.warning("Failed to reload atom %r: %s", line, e)

        self._last_snapshot = engine.get_state_snapshot()
        self._last_hash = compute_state_hash(self._last_snapshot)

        # Reload session metadata if available
        if self.meta_path.exists():
            try:
                meta = json.loads(self.meta_path.read_text(encoding="utf-8"))
                prev_session = meta.get("session_id", "unknown")
                logger.info(
                    "Loaded %d atoms from previous session %s",
                    len(lines),
                    prev_session,
                )
            except Exception:
                pass

        return True

    def has_persisted_state(self) -> bool:
        """Check if a .metta file exists on disk."""
        return self.kb_path.exists() and self.kb_path.stat().st_size > 0

    # ── audit logging ──

    def record_modification(
        self,
        engine: MeTTaEngine,
        nl_input: str,
        metta_rule: str,
        operation: str = "add",
    ) -> AuditEntry:
        """
        Record a knowledge modification event.

        1. Captures before/after snapshots
        2. Computes diff
        3. Appends to audit_log.jsonl
        4. Saves state to disk

        Returns the AuditEntry for UI display.
        """
        before_snapshot = self._last_snapshot
        before_hash = self._last_hash

        # The modification has already been applied to the engine
        after_snapshot = engine.get_state_snapshot()
        after_hash = compute_state_hash(after_snapshot)

        diff = generate_diff(before_snapshot, after_snapshot)

        entry = AuditEntry(
            timestamp=datetime.now(timezone.utc).isoformat(),
            session_id=self.session_id,
            natural_language_input=nl_input,
            extracted_metta_rule=metta_rule,
            before_state_hash=before_hash,
            after_state_hash=after_hash,
            diff=diff,
            operation=operation,
        )

        # Append to audit log
        with self.audit_path.open("a", encoding="utf-8") as f:
            f.write(entry.to_json() + "\n")

        # Persist full state
        self.save_state(engine)

        logger.info(
            "Audit: %s  %s → %s  (diff: %d chars)",
            operation,
            before_hash,
            after_hash,
            len(diff),
        )

        return entry

    def get_audit_log(self, limit: int = 50) -> list[dict[str, Any]]:
        """Read the most recent audit entries."""
        if not self.audit_path.exists():
            return []

        entries = []
        for line in self.audit_path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line:
                try:
                    entries.append(json.loads(line))
                except json.JSONDecodeError:
                    continue

        return entries[-limit:]

    # ── state introspection ──

    @property
    def current_hash(self) -> str:
        return self._last_hash

    @property
    def current_snapshot(self) -> str:
        return self._last_snapshot

    def refresh_snapshot(self, engine: MeTTaEngine) -> None:
        """Update the cached snapshot from the live engine."""
        self._last_snapshot = engine.get_state_snapshot()
        self._last_hash = compute_state_hash(self._last_snapshot)
