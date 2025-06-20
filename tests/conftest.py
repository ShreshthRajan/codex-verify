# tests/conftest.py
"""
Pytest configuration file.
Place this in tests/ directory to fix import paths.
"""

import sys
import os

# Add the project root directory to Python path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)