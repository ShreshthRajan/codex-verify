# ui/__init__.py
"""
CodeX-Verify UI Package
"""

from .streamlit_dashboard import CodeXVerifyDashboard
from .cli_interface import cli

__all__ = ['CodeXVerifyDashboard', 'cli']
