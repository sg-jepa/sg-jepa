from pathlib import Path

import pytest

nbformat = pytest.importorskip("nbformat")
NotebookClient = pytest.importorskip("nbclient").NotebookClient


def test_approach_notebook_executes_without_local_artifacts() -> None:
    root = Path(__file__).resolve().parents[1]
    notebook = nbformat.read(root / "notebooks/approach_rollout.ipynb", as_version=4)
    source = "\n".join(cell.source for cell in notebook.cells if cell.cell_type == "code")
    assert "os.environ" not in source
    assert "SGJEPA_" not in source
    client = NotebookClient(notebook, timeout=120, kernel_name="python3")
    client.execute(cwd=str(root))
    output = notebook.cells[1].outputs[0]["text"]
    assert "Edit the Settings cell" in output
