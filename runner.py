"""Automated runner for Bi-weekly competition Season 1 (Walk-Forward) submissions.

Loads a submission file, runs walk-forward validation, and records results.
Do not modify this file.
"""

import argparse
import csv
import importlib
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time

import pandas as pd

from data_loader import list_periods, load_period
from evaluation import (
    backtest,
    check_cross_sectional_demean,
    returns_to_equity,
    sharpe_ratio,
)
from walkforward import walkforward_backtest, walkforward_full


def convert_notebook_to_python(notebook_path: str) -> str:
    """Convert a Jupyter notebook to a Python script.

    Extracts code cells, skipping markdown and cells with shell commands.

    Args:
        notebook_path: Path to the .ipynb file.

    Returns:
        Path to the generated .py file.
    """
    with open(notebook_path) as f:
        nb = json.load(f)

    temp_dir = tempfile.mkdtemp()
    base_name = os.path.basename(notebook_path).replace(".ipynb", ".py")
    py_path = os.path.join(temp_dir, base_name)

    import ast as _ast

    def _cell_defines_predictor_subclass(filtered_src: str) -> bool:
        """True if this cell's executable code defines a Predictor subclass.

        Earlier substring scan (``"class " in source and "(Predictor)" in source``)
        was triggered by markdown-style example code or string literals
        containing ``class Foo(Predictor)`` and could prematurely terminate
        extraction. Parse the cell as Python instead so only real
        ClassDef nodes count. Fall back to substring (best-effort) on a
        SyntaxError — partial cells happen mid-extraction.
        """
        try:
            tree = _ast.parse(filtered_src)
        except SyntaxError:
            return "class " in filtered_src and "(Predictor)" in filtered_src
        for node in _ast.walk(tree):
            if not isinstance(node, _ast.ClassDef):
                continue
            for base in node.bases:
                base_name = ""
                if isinstance(base, _ast.Name):
                    base_name = base.id
                elif isinstance(base, _ast.Attribute):
                    base_name = base.attr
                if base_name == "Predictor":
                    return True
        return False

    lines = []
    for cell in nb.get("cells", []):
        if cell["cell_type"] != "code":
            continue
        source = "".join(cell["source"])
        # Filter out magic/shell lines but keep the rest of the cell
        filtered = "\n".join(
            line for line in source.splitlines()
            if not line.lstrip().startswith(("!", "%"))
        )
        if not filtered.strip():
            continue
        lines.append(filtered)
        lines.append("\n\n")

    # Note: previously we stopped at the cell that defined the
    # Predictor subclass to skip "analysis/plotting" cells that
    # crash outside the notebook context. That dropped legitimate
    # post-class cells too — trailing imports that the class methods
    # reference would silently disappear, surfacing later as
    # AttributeError or NameError mid-train(). Now all code cells
    # are kept; the structure check in `check_data_leakage.py`
    # rejects any top-level helper / executable code outside the
    # Predictor subclass with a clear "Code Outside Class" error,
    # which is a much better feedback loop for scientists. Plotting /
    # demo code in notebook cells should be guarded with
    # `if __name__ == "__main__":`.

    with open(py_path, "w") as f:
        f.write("\n".join(lines))

    return py_path


def extract_predictor_class(python_file: str) -> tuple[str | None, str | None]:
    """Find the Predictor subclass in a Python file.

    Args:
        python_file: Path to the .py file.

    Returns:
        Tuple of (class_name, instance_name) or (None, None) if not found.
    """
    import ast

    with open(python_file) as f:
        tree = ast.parse(f.read())

    predictor_class_name = None
    predictor_instance_name = None

    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef):
            for base in node.bases:
                base_name = ""
                if isinstance(base, ast.Name):
                    base_name = base.id
                elif isinstance(base, ast.Attribute):
                    base_name = base.attr
                if base_name == "Predictor":
                    predictor_class_name = node.name
                    break

        if isinstance(node, ast.Assign) and predictor_class_name:
            if isinstance(node.value, ast.Call):
                func = node.value.func
                func_name = ""
                if isinstance(func, ast.Name):
                    func_name = func.id
                elif isinstance(func, ast.Attribute):
                    func_name = func.attr
                if func_name == predictor_class_name and node.targets:
                    if isinstance(node.targets[0], ast.Name):
                        predictor_instance_name = node.targets[0].id

    return predictor_class_name, predictor_instance_name


def install_pep723_deps_from_file(file_path: str) -> None:
    """Parse PEP 723 dependencies and pip-install them in the current process.

    Mirrors the internal `5/runner.py` helper of the same name so the
    drift check passes. Scientists who want to run a submission file
    locally can call this directly; the main entry point is
    `load_predictor_class_from_file`, which calls it on the way to
    importing the predictor class.

    Honors `COMP6_DISABLE_PIP_INSTALL`: any truthy value short-circuits.
    No-op if the file doesn't exist or has no PEP 723 block.
    """
    if os.environ.get("COMP6_DISABLE_PIP_INSTALL"):
        print("Skipping PEP 723 install (COMP6_DISABLE_PIP_INSTALL set).")
        return

    if not os.path.exists(file_path):
        return

    with open(file_path) as f:
        content = f.read()

    if "# /// script" not in content or "# dependencies" not in content:
        return

    deps_match = re.search(
        r"# /// script.*?#\s*dependencies\s*=\s*\[(.*?)\].*?# ///",
        content,
        re.DOTALL,
    )
    if not deps_match:
        return

    deps = re.findall(r"""["']([^"']+)["']""", deps_match.group(1))
    if not deps:
        return

    print(f"⚠ About to pip-install from submission metadata: {', '.join(deps)}")
    print("  (set COMP6_DISABLE_PIP_INSTALL=1 to disable this for safety)")
    uv_path = shutil.which("uv")
    if uv_path:
        result = subprocess.run(
            [uv_path, "pip", "install"] + deps,
            capture_output=True,
            text=True,
        )
    else:
        result = subprocess.run(
            [sys.executable, "-m", "pip", "install"] + deps,
            capture_output=True,
            text=True,
        )
    if result.returncode != 0:
        print(f"  Warning: install failed: {result.stderr}")


def load_predictor_class_from_file(file_path: str) -> type:
    """Load the Predictor subclass from a submission file.

    Args:
        file_path: Path to the submission file (.py or .ipynb).

    Returns:
        The Predictor subclass (not an instance).
    """
    if file_path.endswith(".ipynb"):
        python_file = convert_notebook_to_python(file_path)
    else:
        temp_dir = tempfile.mkdtemp()
        base_name = os.path.basename(file_path)
        python_file = os.path.join(temp_dir, base_name)
        shutil.copy2(file_path, python_file)

    predictor_class_name, _ = extract_predictor_class(python_file)

    if predictor_class_name is None:
        raise ValueError(f"No Predictor subclass found in {file_path}")

    # PEP 723 auto-install is convenient when you're testing your own
    # code, but it's a supply-chain risk if you ever run someone else's
    # submission file: a malicious "# dependencies = [...]" block can
    # cause pip to execute setup.py / wheel build steps that read your
    # env (AWS_*, etc.). Set COMP6_DISABLE_PIP_INSTALL=1 to short-circuit
    # this path; you can then `pip install` the listed deps yourself.
    install_pep723_deps_from_file(python_file)

    temp_dir = os.path.dirname(python_file)
    sys.path.insert(0, temp_dir)

    module_name = os.path.basename(python_file).replace(".py", "")
    try:
        module = importlib.import_module(module_name)
        predictor_class = getattr(module, predictor_class_name)
        # Register the submission module for cloudpickle by-value serialization
        # while sys.modules still has it. Defensive — most scientist workflows
        # don't pickle the trained predictor, but if they do (or if a downstream
        # tool like the internal pipeline does), this keeps the pickle
        # self-contained instead of falling back to by-reference. Mirrors the
        # internal `5/runner.py` pattern; see that file for the long rationale.
        try:
            import cloudpickle
            if getattr(module, "__file__", None):
                cloudpickle.register_pickle_by_value(module)
        except ImportError:
            pass  # cloudpickle is optional in the scientist environment
        print(f"Loaded predictor class: {predictor_class_name}")
        return predictor_class
    finally:
        if temp_dir in sys.path:
            sys.path.remove(temp_dir)
        # Evict the cached module too. importlib's cache is keyed on
        # module name, not path — without this, a scientist iterating
        # on a submission named `submission.py` in a Jupyter session
        # gets the first submission's class back on every re-run after
        # the first, silently scoring the wrong model.
        sys.modules.pop(module_name, None)
        # Clean up the tempdir holding the submission .py. Both call
        # paths above (the .py copy and the .ipynb→.py conversion)
        # create tempdirs we own; without this rmtree, every runner.py
        # invocation leaks one tempdir into /tmp.
        shutil.rmtree(temp_dir, ignore_errors=True)


def validate_submission(
    file_path: str,
    output_file: str = "results.csv",
    data_dir: str = "data/train",
    universe: str = "small",
    full: bool = False,
    gauge_fix: bool = False,
    max_train_time: float = 240.0,
    max_predict_time: float = 60.0,
    verbose: bool = True,
) -> dict:
    """Validate a walk-forward submission.

    Default mode: uses the latest period, holds out min(800, 20%) for
    validation. With full=True: walks forward through ALL periods,
    concatenating PnL across validation sets. With gauge_fix=True
    (implies full=True), also computes gauge-fixed vectors, city, IC,
    and other geometric metrics.

    Args:
        file_path: Path to the submission file (.py or .ipynb).
        output_file: Path to output CSV file.
        data_dir: Directory containing training data.
        universe: Universe type.
        full: Walk forward through all periods (default False).
        gauge_fix: Compute gauge-fixed metrics (implies full=True).
        max_train_time: Max training time in seconds (default 4 min).
        max_predict_time: Max prediction time in seconds (default 60s).
        verbose: Print progress.

    Returns:
        Dictionary with walk-forward results.
    """
    if gauge_fix:
        full = True

    if verbose:
        print(f"Processing submission: {file_path}")
        mode = "full walk-forward" if full else "latest period"
        if gauge_fix:
            mode += " + gauge-fix"
        print(f"Mode: {mode}")
        print("=" * 60)

    # Load predictor class
    try:
        predictor_class = load_predictor_class_from_file(file_path)
    except Exception as e:
        print(f"Failed to load predictor: {e}")
        return {}

    if full:
        results = walkforward_full(
            predictor_class,
            data_dir=data_dir,
            universe=universe,
            gauge_fix=gauge_fix,
            max_train_time=max_train_time,
            max_predict_time=max_predict_time,
            verbose=verbose,
        )
    else:
        results = walkforward_backtest(
            predictor_class,
            data_dir=data_dir,
            universe=universe,
            max_train_time=max_train_time,
            max_predict_time=max_predict_time,
            verbose=verbose,
        )

    # A predictor that runs cleanly but happens to produce zero PnL (e.g. a
    # constant or perfectly mean-neutral signal) returns sharpe=0.0 — that's
    # a real result, not a failure. Distinguish via the explicit error keys
    # that walkforward sets on actual failure paths.
    if "sharpe" not in results or results.get("error"):
        print(f"Validation failed: {results.get('error', 'no sharpe in results')}")
        return results

    # Write results
    file_exists = os.path.isfile(output_file)
    with open(output_file, "a", newline="") as f:
        writer = csv.writer(f)
        if not file_exists:
            header = ["file_name", "sharpe", "n_observations", "timestamp"]
            if gauge_fix:
                header.extend([
                    "ic", "ic_std", "ic_dispersion_corr",
                    "concentration", "compression_loss",
                    "city_novelty_deg", "nearest_city",
                ])
            writer.writerow(header)

        row = [
            file_path,
            f"{results['sharpe']:.6f}",
            results.get("n_observations", results.get("val_rows", "")),
            pd.Timestamp.now().isoformat(),
        ]
        if gauge_fix:
            row.extend([
                f"{results.get('ic', 0):.6f}",
                f"{results.get('ic_std', 0):.6f}",
                f"{results.get('ic_dispersion_corr', float('nan')):.6f}",
                f"{results.get('concentration', 0):.6f}",
                f"{results.get('compression_loss', 0):.6f}",
                f"{results.get('city_novelty_deg', float('nan')):.1f}",
                results.get("nearest_city", "N/A"),
            ])
        writer.writerow(row)

    if verbose:
        print(f"\nResults written to {output_file}")

    return results


def main():
    parser = argparse.ArgumentParser(
        description="Validate Bi-weekly competition Season 1 walk-forward submissions"
    )
    parser.add_argument("submission", help="Path to submission file (.py or .ipynb)")
    parser.add_argument("--output", default="results.csv", help="Output CSV file")
    parser.add_argument(
        "--data-dir", default="data/train", help="Training data directory"
    )
    parser.add_argument("--universe", default="small", help="Universe type")
    parser.add_argument(
        "--full", action="store_true",
        help="Walk forward through all periods (slower, more representative)",
    )
    parser.add_argument(
        "--gauge-fix", action="store_true",
        help="Compute gauge-fixed metrics: city, IC, concentration (implies --full)",
    )
    parser.add_argument("--quiet", action="store_true", help="Suppress output")

    args = parser.parse_args()

    validate_submission(
        args.submission,
        output_file=args.output,
        data_dir=args.data_dir,
        universe=args.universe,
        full=args.full,
        gauge_fix=args.gauge_fix,
        verbose=not args.quiet,
    )


if __name__ == "__main__":
    main()
