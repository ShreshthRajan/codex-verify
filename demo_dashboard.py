# demo_dashboard.py
"""
Quick demo launcher for CodeX-Verify Dashboard
"""

import subprocess
import sys
import os
from pathlib import Path

def main():
    """Launch the CodeX-Verify dashboard"""
    
    print("🔍 CodeX-Verify Dashboard")
    print("=" * 40)
    print("🎯 Multi-agent verification • 4 agents • Production-ready")
    print()
    
    # Use the FULL dashboard, not the simple one
    dashboard_path = Path(__file__).parent / "ui" / "streamlit_dashboard.py"
    
    if not dashboard_path.exists():
        print("❌ Dashboard file not found!")
        print(f"Expected: {dashboard_path}")
        return
    
    print(f"🚀 Launching full dashboard...")
    print(f"📁 Location: {dashboard_path}")
    print()
    print("🌐 The dashboard will open in your browser automatically")
    print("🔧 Use Ctrl+C to stop the server")
    print()
    
    try:
        # Launch Streamlit with the FULL dashboard
        subprocess.run([
            sys.executable, "-m", "streamlit", "run", 
            str(dashboard_path),
            "--server.port", "8501",
            "--server.headless", "false",
            "--browser.gatherUsageStats", "false"
        ])
    except KeyboardInterrupt:
        print("\n👋 Dashboard stopped. Thank you for using CodeX-Verify!")
    except Exception as e:
        print(f"❌ Error launching dashboard: {e}")

if __name__ == "__main__":
    main()