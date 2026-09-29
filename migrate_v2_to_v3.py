import json
import sqlite3
import os
import uuid
from datetime import datetime

def migrate():
    if not os.path.exists("data"):
        os.makedirs("data")
    conn = sqlite3.connect("data/evidence_ledger.sqlite")
    cursor = conn.cursor()
    
    try:
        cursor.execute("SELECT COUNT(*) FROM claims WHERE status = 'active'")
        if cursor.fetchone()[0] > 0:
            print("Database already contains claims. Aborting migration.")
            return
    except sqlite3.OperationalError:
        print("Database not initialized yet. Skipping migration.")
        return
        
    print("Starting migration of V2 knowledge...")
    
    if not os.path.exists("audit_log.jsonl_v2"):
        print("No V2 audit log found at 'audit_log.jsonl_v2'.")
        return
        
    migrated_count = 0
    with open("audit_log.jsonl_v2", "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip(): continue
            record = json.loads(line)
            
            u_id = f"u_migrated_{uuid.uuid4().hex[:8]}"
            cursor.execute(
                "INSERT INTO utterances (id, raw_text, timestamp) VALUES (?, ?, ?)",
                (u_id, record.get("user_input", "SYSTEM_MIGRATION"), record.get("timestamp", datetime.utcnow().isoformat()))
            )
            
            c_id = f"c_{uuid.uuid4().hex[:12]}"
            cursor.execute(
                "INSERT INTO claims (id, utterance_id, metta_atom, operation, status, timestamp) VALUES (?, ?, ?, ?, ?, ?)",
                (c_id, u_id, record["metta_atom"], record["operation"], "active", record.get("timestamp", datetime.utcnow().isoformat()))
            )
            migrated_count += 1
            
    conn.commit()
    conn.close()
    print(f"Successfully migrated {migrated_count} facts from V2 to V3.")

if __name__ == "__main__":
    migrate()
