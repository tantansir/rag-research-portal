"""
Launch script for Personal Research Portal Phase 3
Run this script to start the Streamlit web application
"""

import subprocess
import sys
from pathlib import Path

if __name__ == "__main__":
    app_path = Path(__file__).parent / "src" / "app" / "app.py"
    
    print("=" * 60)
    print("Personal Research Portal - Phase 3")
    print("=" * 60)
    print("\nStarting Streamlit app...")
    print(f"App path: {app_path}")
    print("\nHint: Make sure GEMINI_API_KEY environment variable is set")
    print("=" * 60)
    print()
    
    try:
        subprocess.run(
            [sys.executable, "-m", "streamlit", "run", str(app_path)],
            check=True
        )
    except KeyboardInterrupt:
        print("\nApp stopped")
    except Exception as e:
        print(f"Failed to start: {e}")
        sys.exit(1)
