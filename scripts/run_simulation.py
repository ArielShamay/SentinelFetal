"""
SentinelFetal Simulation Dashboard Launcher.

This script launches the real-time CTG simulation dashboard.

Usage:
    python scripts/run_simulation.py
    
Or directly with streamlit:
    streamlit run src/ui/simulation_app.py
"""

import subprocess
import sys
from pathlib import Path

# Fix encoding for Windows console without replacing stdout (avoids closed-file errors)
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")


def main():
    """Launch the simulation dashboard."""
    # Get paths
    project_root = Path(__file__).parent.parent
    app_path = project_root / "src" / "ui" / "simulation_app.py"
    
    if not app_path.exists():
        print(f"Error: App not found at {app_path}")
        sys.exit(1)
    
    print("=" * 60)
    print("[+] SentinelFetal Real-Time Simulator")
    print("=" * 60)
    print()
    print("Launching simulation dashboard...")
    print(f"App path: {app_path}")
    print()
    print("Dashboard will open in your browser.")
    print("Press Ctrl+C in this terminal to stop the server.")
    print()
    print("=" * 60)
    
    # Launch streamlit
    try:
        subprocess.run(
            [
                sys.executable, "-m", "streamlit", "run",
                str(app_path),
                "--server.headless", "true",
                "--browser.gatherUsageStats", "false"
            ],
            cwd=str(project_root)
        )
    except KeyboardInterrupt:
        print("\n\nSimulation stopped.")


if __name__ == "__main__":
    main()
