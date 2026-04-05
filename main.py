"""
main.py — CLI for the SOP → Airtop Agent Prompt RAG system.

Commands
--------
  python main.py ingest   <path>       Ingest SOP docs from a file or folder
  python main.py generate <task>       Generate an Airtop prompt for a task
  python main.py info                  Show collection stats
  python main.py clear                 Wipe the ChromaDB collection
  python main.py demo                  Run a self-contained demo with sample text

Examples
--------
  python main.py ingest ./sop_docs
  python main.py ingest ./sops/login_procedure.pdf
  python main.py generate "Log into the portal and download the monthly report"
  python main.py info
"""

import sys
import textwrap
from pathlib import Path

from rag_pipeline import SOPAgentRAG


# ---------------------------------------------------------------------------
# Demo helpers (no real files needed)
# ---------------------------------------------------------------------------

DEMO_SOP_TEXT = """
STANDARD OPERATING PROCEDURE — Web Portal Order Export

1. PURPOSE
   This procedure defines the steps for exporting pending orders from the
   company's order management portal (https://orders.internal.acme.com).

2. SCOPE
   Applies to all members of the Operations team with Viewer or higher access.

3. PREREQUISITES
   - Valid SSO credentials (username: corp email, password: AD password).
   - VPN must be active before accessing the portal.
   - Only users with the "Reports" permission can download CSV exports.

4. PROCEDURE
   4.1  Navigate to https://orders.internal.acme.com and click "Sign In".
   4.2  Enter your corporate email and password, then complete MFA if prompted.
   4.3  From the left sidebar, select "Orders" → "Pending".
   4.4  Set the date filter to "Last 30 days" using the calendar picker.
   4.5  Click "Export" in the top-right corner and choose "CSV".
   4.6  Save the file to the shared drive at //fileserver/ops/exports/.
   4.7  Confirm the row count matches the number shown in the UI before closing.

5. ERROR HANDLING
   - If the Export button is greyed out, your account lacks the "Reports"
     permission — contact IT via help@acme.com.
   - If the portal is unreachable, verify VPN connectivity before retrying.
   - Do NOT attempt to scrape data manually if the export fails; log a ticket.

6. COMPLIANCE
   Exported files must not be stored on personal devices.  Delete local copies
   after uploading to the shared drive within 24 hours.
""".strip()


def run_demo() -> None:
    """Write a temporary SOP text file and run the full pipeline as a demo."""
    demo_file = Path("demo_sop.txt")
    demo_file.write_text(DEMO_SOP_TEXT, encoding="utf-8")
    print("Demo SOP written to demo_sop.txt\n")

    rag = SOPAgentRAG()
    rag.ingest(str(demo_file))

    task = (
        "Log into the order management portal and export all pending orders "
        "from the last 30 days as a CSV, then save the file to the shared drive."
    )

    prompt = rag.generate_prompt(task)

    print("=" * 70)
    print("GENERATED AIRTOP AGENT PROMPT")
    print("=" * 70)
    print(prompt)
    print("=" * 70)

    # Clean up demo file
    demo_file.unlink(missing_ok=True)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    args = sys.argv[1:]

    if not args or args[0] in ("-h", "--help", "help"):
        print(__doc__)
        return

    command = args[0].lower()

    if command == "demo":
        run_demo()
        return

    rag = SOPAgentRAG()

    if command == "ingest":
        if len(args) < 2:
            print("Usage: python main.py ingest <path>")
            sys.exit(1)
        rag.ingest(args[1])
        info = rag.collection_info()
        print(f"Collection now contains {info['documents']} chunks.")

    elif command == "generate":
        if len(args) < 2:
            print("Usage: python main.py generate \"<task description>\"")
            sys.exit(1)
        task = " ".join(args[1:])
        prompt = rag.generate_prompt(task)
        print("\n" + "=" * 70)
        print("GENERATED AIRTOP AGENT PROMPT")
        print("=" * 70)
        print(prompt)
        print("=" * 70)

    elif command == "info":
        info = rag.collection_info()
        print(f"Status  : {info['status']}")
        print(f"Chunks  : {info['documents']}")

    elif command == "clear":
        rag.clear_database()

    else:
        print(f"Unknown command: '{command}'")
        print("Run `python main.py --help` for usage.")
        sys.exit(1)


if __name__ == "__main__":
    main()
