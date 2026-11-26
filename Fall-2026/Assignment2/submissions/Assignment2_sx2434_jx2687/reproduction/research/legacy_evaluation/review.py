"""Rebuild the November 23 validation diagnostic with the original pre-integration model."""
import importlib.util
import os
import subprocess
from pathlib import Path
import runpy
import sys

HERE = Path(__file__).resolve().parent
source = HERE.parent / "valuation_candidate/source_code"
sys.path.insert(0, str(source))
# The final validation evaluation is as of November 30, 2023. The available
# December sessions establish that November is complete without later prices.
(HERE / "student_data").mkdir(exist_ok=True)
subprocess.run([sys.executable, str(HERE / "build_history_bc98428.py"), "--through-month", "2023-11"],
               check=True, cwd=HERE, env=dict(os.environ, PYTHONPATH=str(source)))
from shared import data_view
data_view.STUDENT_DATA = HERE / "student_data"
name = "valuation.student_valuation"
spec = importlib.util.spec_from_file_location(name, HERE / "model_bc98428.py")
module = importlib.util.module_from_spec(spec)
sys.modules[name] = module
spec.loader.exec_module(module)
runpy.run_path(str(source / "review_inputs.py"), run_name="__main__")
