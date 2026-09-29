import sqlite3
import json
import logging
import uuid
import time
from pathlib import Path
from dataclasses import dataclass, field
from typing import Any

from core.metta_engine import MeTTaEngine

logger = logging.getLogger(__name__)

@dataclass
class PersistenceManager:
    data_dir: Path = field(default_factory=lambda: Path("data"))
    filename: str = field(default_factory=lambda: "evidence_ledger.sqlite")
    session_id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    _conn: sqlite3.Connection = field(init=False, repr=False)
    
    def __post_init__(self) -> None:
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(self.db_path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._init_db()
        
    @property
    def db_path(self) -> Path:
        return self.data_dir / self.filename
        
    @property
    def kb_path(self) -> Path:
        return self.data_dir / "knowledge_base.metta"
        
    @property
    def audit_path(self) -> Path:
        return self.data_dir / "audit_log.jsonl"
        
    def _init_db(self):
        cursor = self._conn.cursor()
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS utterances (
                id TEXT PRIMARY KEY,
                session_id TEXT,
                text TEXT,
                timestamp REAL
            )
        """)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS claims (
                id TEXT PRIMARY KEY,
                utterance_id TEXT,
                metta_atom TEXT,
                operation TEXT,
                status TEXT,
                timestamp REAL,
                op_id TEXT UNIQUE,
                FOREIGN KEY(utterance_id) REFERENCES utterances(id)
            )
        """)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS pending_clarifications (
                id TEXT PRIMARY KEY,
                session_id TEXT,
                original_text TEXT,
                status TEXT,
                timestamp REAL
            )
        """)
        self._conn.commit()
        
    def load_state(self, engine: MeTTaEngine) -> bool:
        """
        Rebuilds the MeTTa engine from active claims in SQLite.
        """
        cursor = self._conn.cursor()
        # Order by timestamp to replay history deterministically
        cursor.execute("SELECT metta_atom, operation FROM claims WHERE status = 'active' ORDER BY timestamp ASC")
        rows = cursor.fetchall()
        
        if not rows:
            logger.info("Persisted state is empty")
            return False
            
        for row in rows:
            if row["operation"] == "add":
                engine.add_atom(row["metta_atom"])
            elif row["operation"] == "remove":
                engine.remove_atom(row["metta_atom"])
                
        logger.info(f"Loaded {len(rows)} claims from SQLite ledger.")
        return True
        
    def record_utterance(self, text: str) -> str:
        """Saves exact user text and returns the utterance ID."""
        u_id = f"u_{uuid.uuid4().hex[:8]}"
        cursor = self._conn.cursor()
        cursor.execute(
            "INSERT INTO utterances (id, session_id, text, timestamp) VALUES (?, ?, ?, ?)",
            (u_id, self.session_id, text, time.time())
        )
        self._conn.commit()
        return u_id
        
    def record_modification(self, engine: MeTTaEngine, utterance_id: str, diff_text: str, operation: str = "add", op_id: str = None) -> None:
        """
        Atomically records a semantic write with idempotent operation ID.
        """
        c_id = f"c_{uuid.uuid4().hex[:8]}"
        if op_id is None:
            import hashlib
            op_id = "op_" + hashlib.sha256(f"{utterance_id}:{diff_text}:{operation}".encode()).hexdigest()[:12]
        cursor = self._conn.cursor()
        cursor.execute(
            "INSERT INTO claims (id, utterance_id, metta_atom, operation, status, timestamp, op_id) VALUES (?, ?, ?, ?, ?, ?, ?)",
            (c_id, utterance_id, diff_text, operation, "active", time.time(), op_id)
        )
        self._conn.commit()
        
        # Export for inspection compatibility
        self.export_snapshots(engine)
        
        # Also write to audit_log for UI backward compatibility during Stage A/B
        entry = {
            "operation": operation,
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S.000000+00:00", time.gmtime()),
            "user_input": f"Utterance {utterance_id}",
            "diff": diff_text
        }
        with open(self.audit_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry) + "\n")
    
    def operation_exists(self, op_id: str) -> bool:
        """Check whether an operation ID has already been committed."""
        cursor = self._conn.cursor()
        cursor.execute("SELECT 1 FROM claims WHERE op_id = ?", (op_id,))
        return cursor.fetchone() is not None
    
    def get_utterance_text(self, utterance_id: str) -> str | None:
        """Return the original text of an utterance by ID."""
        cursor = self._conn.cursor()
        cursor.execute("SELECT text FROM utterances WHERE id = ?", (utterance_id,))
        row = cursor.fetchone()
        return row["text"] if row else None
        
    def record_pending_clarification(self, text: str) -> str:
        """Records a user input that requires clarification."""
        c_id = f"clar_{uuid.uuid4().hex[:8]}"
        cursor = self._conn.cursor()
        cursor.execute(
            "INSERT INTO pending_clarifications (id, session_id, original_text, status, timestamp) VALUES (?, ?, ?, ?, ?)",
            (c_id, self.session_id, text, "pending", time.time())
        )
        self._conn.commit()
        return c_id

    def get_pending_clarification(self) -> dict | None:
        """Retrieves the most recent pending clarification for this session."""
        cursor = self._conn.cursor()
        cursor.execute("SELECT id, original_text FROM pending_clarifications WHERE session_id = ? AND status = 'pending' ORDER BY timestamp DESC LIMIT 1", (self.session_id,))
        row = cursor.fetchone()
        if row:
            return {"id": row["id"], "original_text": row["original_text"]}
        return None

    def resolve_clarification(self, clar_id: str) -> None:
        """Marks a clarification as resolved."""
        cursor = self._conn.cursor()
        cursor.execute("UPDATE pending_clarifications SET status = 'resolved' WHERE id = ?", (clar_id,))
        self._conn.commit()
            
    def export_snapshots(self, engine: MeTTaEngine) -> None:
        """Writes the current graph state to flat files for inspection."""
        with open(self.kb_path, "w", encoding="utf-8") as f:
            for atom in engine.get_all_atoms():
                f.write(f"{atom}\n")
            # Export rules too
            for rule in getattr(engine, "_rules", []):
                f.write(f"{rule}\n")
                
    def get_claim_by_atom(self, atom: str) -> str | None:
        cursor = self._conn.cursor()
        cursor.execute("SELECT id FROM claims WHERE metta_atom = ? AND status = 'active' AND operation = 'add' ORDER BY timestamp DESC LIMIT 1", (atom,))
        row = cursor.fetchone()
        return row["id"] if row else None
        
    def get_audit_log(self, limit: int = 50) -> list[dict[str, Any]]:
        cursor = self._conn.cursor()
        cursor.execute("""
            SELECT c.operation, c.timestamp, c.metta_atom, u.text as user_input 
            FROM claims c 
            LEFT JOIN utterances u ON c.utterance_id = u.id 
            ORDER BY c.timestamp DESC LIMIT ?
        """, (limit,))
        
        results = []
        for row in cursor.fetchall():
            results.append({
                "operation": row["operation"],
                "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S.000000+00:00", time.gmtime(row["timestamp"])),
                "user_input": row["user_input"] or "",
                "diff": row["metta_atom"]
            })
        return list(reversed(results))
