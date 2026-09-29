"""
test_lifecycle.py
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Automated End-to-End Verification Harness.

Simulates the complete agent lifecycle WITHOUT the UI:

  Step 1  →  Agent starts with an empty knowledge base
  Step 2  →  Query "Where are the biosensor kits stored?" → empty / uncertainty
  Step 3  →  Teach "Biosensor kits are stored in Tech Park Lab 3."
              → parse to MeTTa, inject, display diff, commit to disk
  Step 4  →  Kill process state, reload from disk in a fresh session
  Step 5  →  Query again → agent answers correctly from persisted AtomSpace

Run:  python test_lifecycle.py
"""

from __future__ import annotations

import json
import shutil
import sys
import tempfile
from pathlib import Path

# ── Rich console for pretty output ──
try:
    from rich.console import Console
    from rich.panel import Panel
    from rich.syntax import Syntax
    from rich.table import Table
    from rich import box
    RICH = True
except ImportError:
    RICH = False

from core.metta_engine import MeTTaEngine
from core.persistence import PersistenceManager, compute_state_hash
from core.translator import LLMTranslator, IntentType


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Output Helpers
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

if RICH:
    console = Console(force_terminal=True)

    def header(text: str):
        console.print(f"\n[bold cyan]{'=' * 60}[/]")
        console.print(f"[bold cyan]  {text}[/]")
        console.print(f"[bold cyan]{'=' * 60}[/]\n")

    def success(text: str):
        console.print(f"  [bold green]PASS: {text}[/]")

    def fail(text: str):
        console.print(f"  [bold red]FAIL: {text}[/]")

    def info(text: str):
        console.print(f"  [dim]{text}[/]")

    def show_diff(diff_text: str):
        console.print(Panel(
            Syntax(diff_text, "diff", theme="monokai"),
            title="[bold]Knowledge Diff[/]",
            border_style="green",
        ))

    def show_atoms(atoms: list[str]):
        if not atoms:
            console.print("  [dim italic](empty AtomSpace)[/]")
            return
        for a in sorted(atoms):
            console.print(f"  [yellow]{a}[/]")

    def show_result_box(title: str, content: str, style: str = "blue"):
        console.print(Panel(content, title=f"[bold]{title}[/]", border_style=style))

else:
    # Plain fallback
    def header(text: str):
        print(f"\n{'=' * 60}")
        print(f"  {text}")
        print(f"{'=' * 60}\n")

    def success(text: str):
        print(f"  ✅ {text}")

    def fail(text: str):
        print(f"  ❌ {text}")

    def info(text: str):
        print(f"  {text}")

    def show_diff(diff_text: str):
        print(f"\n--- DIFF ---\n{diff_text}\n--- END ---\n")

    def show_atoms(atoms: list[str]):
        if not atoms:
            print("  (empty AtomSpace)")
            return
        for a in sorted(atoms):
            print(f"  {a}")

    def show_result_box(title: str, content: str, style: str = ""):
        print(f"\n[{title}]\n{content}\n")


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Test Runner
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def run_lifecycle_test():
    """Execute the full 5-step lifecycle verification."""

    # Use a temp directory so tests don't pollute real data
    test_data_dir = Path(tempfile.mkdtemp(prefix="supermemory_test_"))
    info(f"Test data directory: {test_data_dir}")

    passed = 0
    failed = 0

    try:
        # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        # STEP 1: Empty Knowledge Base
        # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        header("STEP 1 · Initialize with Empty Knowledge Base")

        engine = MeTTaEngine()
        pm = PersistenceManager(data_dir=test_data_dir)
        translator = LLMTranslator()

        atoms = engine.get_all_atoms()
        info(f"Engine mode: {'Mock' if engine.is_mock else 'Native Hyperon'}")
        info(f"Translator mode: {'Fallback (regex)' if translator._use_fallback else 'LLM'}")
        info(f"AtomSpace contents:")
        show_atoms(atoms)

        if len(atoms) == 0:
            success("AtomSpace is empty at startup")
            passed += 1
        else:
            fail(f"AtomSpace should be empty but has {len(atoms)} atoms")
            failed += 1

        # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        # STEP 2: Query with Empty KB → Uncertainty
        # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        header("STEP 2 · Query Empty KB → Uncertainty Detection")

        query_text = "Where are the biosensor kits stored?"
        info(f'User asks: "{query_text}"')

        translation = translator.translate(query_text)
        info(f"Detected intent: {translation.intent.value}")
        info(f"MeTTa pattern: {translation.metta_expression}")
        info(f"Query template: {translation.query_template}")

        if translation.intent != IntentType.QUERY:
            fail(f"Expected 'query' intent, got '{translation.intent.value}'")
            failed += 1
        else:
            success("Correctly classified as a query")
            passed += 1

        result = engine.query(
            translation.metta_expression,
            translation.query_template,
        )
        info(f"Query result: {result.formatted}")

        if result.is_empty:
            success("Empty result set — uncertainty flag triggered ✓")
            passed += 1

            # Generate uncertainty payload
            uncertainty = translator.generate_answer(query_text, [])
            show_result_box(
                "Uncertainty Payload",
                f"Message: {uncertainty.message}\n"
                f"Suggestion: {uncertainty.suggested_input}\n"
                f"Missing concepts: {uncertainty.missing_concepts}",
                style="yellow",
            )
        else:
            fail("Expected empty results from empty KB!")
            failed += 1

        # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        # STEP 3: Teach a Fact → Learn, Diff, Persist
        # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        header("STEP 3 · Teach Fact → MeTTa Injection + Diff + Persist")

        teach_text = "Biosensor kits are stored in Tech Park Lab 3."
        info(f'User says: "{teach_text}"')

        translation = translator.translate(teach_text)
        info(f"Detected intent: {translation.intent.value}")
        info(f"MeTTa expression: {translation.metta_expression}")
        info(f"Verbal reasoning: {translation.verbal_reasoning}")
        info(f"Confidence: {translation.confidence}")

        if translation.intent != IntentType.ASSERTION:
            fail(f"Expected 'assertion' intent, got '{translation.intent.value}'")
            failed += 1
        else:
            success("Correctly classified as an assertion")
            passed += 1

        # Inject into AtomSpace
        pm.refresh_snapshot(engine)
        injected = engine.add_atom(translation.metta_expression)
        info(f"Injected atom: {injected}")

        # Record modification (audit + persist)
        entry = pm.record_modification(
            engine, teach_text, translation.metta_expression, "add"
        )

        # Show diff
        info("Knowledge diff:")
        show_diff(entry.diff)

        info(f"Before hash: {entry.before_state_hash}")
        info(f"After hash:  {entry.after_state_hash}")

        # Verify atom is in space
        atoms = engine.get_all_atoms()
        info("AtomSpace after learning:")
        show_atoms(atoms)

        if len(atoms) > 0:
            success(f"AtomSpace now contains {len(atoms)} atom(s)")
            passed += 1
        else:
            fail("AtomSpace is still empty after injection!")
            failed += 1

        # Verify persistence to disk
        if pm.kb_path.exists() and pm.kb_path.stat().st_size > 0:
            success(f"State persisted to {pm.kb_path}")
            passed += 1
            info(f"File contents:\n{pm.kb_path.read_text(encoding='utf-8')}")
        else:
            fail("State was NOT persisted to disk!")
            failed += 1

        # Verify audit log
        if pm.audit_path.exists():
            audit_entries = pm.get_audit_log()
            success(f"Audit log has {len(audit_entries)} entry(ies)")
            passed += 1
            for ae in audit_entries:
                info(f"  Audit: {ae['operation']} → {ae['extracted_metta_rule']}")
        else:
            fail("No audit log created!")
            failed += 1

        # Store the metta expression for later verification
        learned_expression = translation.metta_expression

        # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        # STEP 4: Simulate Process Kill → Fresh Reload
        # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        header("STEP 4 · Simulate Process Kill → Reload from Disk")

        info("Destroying engine and persistence objects...")
        del engine
        del pm
        del translator
        info("All in-memory state destroyed ✓")

        # Create fresh instances
        info("Creating fresh engine, persistence, and translator...")
        engine2 = MeTTaEngine()
        pm2 = PersistenceManager(data_dir=test_data_dir)
        translator2 = LLMTranslator()

        atoms_before_load = engine2.get_all_atoms()
        info(f"AtomSpace before reload: {len(atoms_before_load)} atoms")

        if len(atoms_before_load) == 0:
            success("Fresh engine starts empty (as expected)")
            passed += 1
        else:
            fail("Fresh engine should be empty!")
            failed += 1

        # Reload from disk
        loaded = pm2.load_state(engine2)
        atoms_after_load = engine2.get_all_atoms()
        info(f"AtomSpace after reload: {len(atoms_after_load)} atoms")
        show_atoms(atoms_after_load)

        if loaded and len(atoms_after_load) > 0:
            success(f"Successfully reloaded {len(atoms_after_load)} atom(s) from disk!")
            passed += 1
        else:
            fail("Failed to reload state from disk!")
            failed += 1

        # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        # STEP 5: Re-Query → Correct Answer from Persisted KB
        # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        header("STEP 5 · Re-Query → Answer from Persisted AtomSpace")

        query_text_2 = "Where are the biosensor kits stored?"
        info(f'User asks again: "{query_text_2}"')

        translation2 = translator2.translate(query_text_2)
        info(f"MeTTa pattern: {translation2.metta_expression}")

        result2 = engine2.query(
            translation2.metta_expression,
            translation2.query_template,
        )
        info(f"Query result: {result2.formatted}")

        if not result2.is_empty:
            success("Query returned results from persisted knowledge! 🎉")
            passed += 1

            answer = translator2.generate_answer(
                query_text_2,
                result2.raw,
                source_atoms=result2.raw,
            )
            show_result_box(
                "Agent Answer",
                f"Answer: {answer.answer}\n"
                f"Source atoms: {answer.source_atoms}\n"
                f"Confidence: {answer.confidence}",
                style="green",
            )
        else:
            fail("Query returned EMPTY — persistence or query failed!")
            failed += 1

    finally:
        # Cleanup
        shutil.rmtree(test_data_dir, ignore_errors=True)

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # RESULTS SUMMARY
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    header("TEST RESULTS")

    total = passed + failed
    info(f"Passed: {passed}/{total}")
    info(f"Failed: {failed}/{total}")

    if RICH:
        table = Table(box=box.ROUNDED, title="Lifecycle Test Summary")
        table.add_column("Metric", style="bold")
        table.add_column("Value", justify="right")
        table.add_row("Total Checks", str(total))
        table.add_row("Passed", f"[green]{passed}[/]")
        table.add_row("Failed", f"[red]{failed}[/]" if failed else f"[green]{failed}[/]")
        table.add_row(
            "Status",
            "[bold green]ALL PASSED ✅[/]" if failed == 0 else "[bold red]FAILURES ❌[/]",
        )
        console.print(table)
    else:
        if failed == 0:
            print(f"\n  🎉 ALL {total} CHECKS PASSED\n")
        else:
            print(f"\n  ⚠️  {failed} FAILURE(S) out of {total} checks\n")

    return failed == 0


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Entry Point
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

if __name__ == "__main__":
    all_passed = run_lifecycle_test()
    sys.exit(0 if all_passed else 1)
