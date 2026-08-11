import sys
import os

# Ensure the project root is always in sys.path so that
# `from backend.xxx import ...` works without setting PYTHONPATH manually.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
