import json
import sqlite3
import os
import uuid
import time
import hashlib
from datetime import datetime

def migrate():
    if not os.path.exists("data"):
        os.makedirs("data")
    conn = sqlite3.connect("data/evidence_ledger.sqlite")
    cursor = conn.cursor()
    
    try:
        cursor.execute("CREATE TABLE IF NOT EXISTS utterances (id TEXT PRIMARY KEY, session_id TEXT, text TEXT, timestamp REAL)")
        cursor.execute("CREATE TABLE IF NOT EXISTS claims (id TEXT PRIMARY KEY, utterance_id TEXT, metta_atom TEXT, operation TEXT, status TEXT, timestamp REAL, op_id TEXT UNIQUE, FOREIGN KEY(utterance_id) REFERENCES utterances(id))")
        conn.commit()
    except sqlite3.OperationalError:
        pass
        
    print("Starting migration of V2 knowledge...")
    
    if not os.path.exists("data/audit_log.jsonl"):
        print("No V2 audit log found at 'data/audit_log.jsonl'. Creating dummy file for testing if necessary.")
        return
        
    migrated_count = 0
    with open("data/audit_log.jsonl", "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip(): continue
            record = json.loads(line)
            
            u_id = f"u_migrated_{uuid.uuid4().hex[:8]}"
            
            # V2 timestamps were ISO format. Convert to UNIX epoch REAL
            ts_str = record.get("timestamp")
            if ts_str:
                try:
                    ts = datetime.fromisoformat(ts_str.replace("Z", "+00:00")).timestamp()
                except Exception:
                    ts = time.time()
            else:
                ts = time.time()
                
            cursor.execute(
                "INSERT INTO utterances (id, session_id, text, timestamp) VALUES (?, ?, ?, ?)",
                (u_id, "v2_legacy_migration", record.get("user_input", "SYSTEM_MIGRATION"), ts)
            )
            
            c_id = f"c_{uuid.uuid4().hex[:12]}"
            
            # V3 requires an idempotent op_id
            op_id = "op_migrated_" + hashlib.sha256(f"{u_id}:{record.get('metta_atom')}:{record.get('operation')}".encode()).hexdigest()[:12]
            
            # Diff was called 'diff' or 'metta_atom' depending on the version
            atom = record.get("diff") or record.get("metta_atom")
            if atom is None: continue
            
            cursor.execute(
                "INSERT OR IGNORE INTO claims (id, utterance_id, metta_atom, operation, status, timestamp, op_id) VALUES (?, ?, ?, ?, ?, ?, ?)",
                (c_id, u_id, atom, record.get("operation", "add"), "active", ts, op_id)
            )
            migrated_count += 1
            
    conn.commit()
    conn.close()
    print(f"Successfully migrated {migrated_count} facts from V2 to V3.")

if __name__ == "__main__":
    migrate()
