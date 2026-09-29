import os
from core.metta_engine import MeTTaEngine

# Initialize the engine
engine = MeTTaEngine()

# Synthetic Campus Data
facts = [
    # Hierarchy and Locations
    "(Location BiosensorKits TechParkLab3)",
    "(Location TechParkLab3 TechPark)",
    "(Location TechPark SRMIST)",
    
    # Management and Personnel
    "(Manager TechParkLab3 DrSmith)",
    "(Manager MainBlockLab DrJones)",
    "(Adviser EnvironmentalClub DrGreen)",
    
    # Roles and Permissions
    "(RequiresClearance TechParkLab3 Level2)",
    "(HasClearance Anush Level2)",
    "(Role Anush Student)",
    
    # Events
    "(Time PotheriLakeCleanup 0900)",
    "(Location PotheriLakeCleanup PotheriLake)"
]

print("Injecting campus knowledge...")
for fact in facts:
    engine.add_atom(fact)

# Save the updated state to disk
with open("knowledge_base.metta", "w") as f:
    f.write(engine.get_state_snapshot())

print(f"Injection complete. Total atoms in graph: {len(engine.get_all_atoms())}")
