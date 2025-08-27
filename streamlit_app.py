"""
実行時: streamlit run streamlit_app.py
"""

import runpy
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent

# src だけ import path に追加 (demo はパッケージとして解決させる)
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

# demo をパッケージとして扱いつつ, demo/main.py を __main__ として実行
runpy.run_module("demo.main", run_name="__main__")
