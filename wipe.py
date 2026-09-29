from core.persistence import PersistenceManager
from pathlib import Path
import os
pm = PersistenceManager(Path(os.getcwd()))
atoms_file = pm.data_dir / "atoms.json"
audit_file = pm.data_dir / "audit.json"
if atoms_file.exists(): atoms_file.unlink()
if audit_file.exists(): audit_file.unlink()
print("Wiped database!")
