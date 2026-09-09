from pathlib import Path

from sg_jepa.data import load_recipes, validate_evaluation_manifest
from sg_jepa.data.recipes import TASK_ACTION_SCHEMAS


def test_main_text_recipes_cover_all_tasks() -> None:
    root = Path(__file__).resolve().parents[1]
    recipes = load_recipes(root / "data_generation/recipes/main_text.yaml")
    assert set(recipes) == set(TASK_ACTION_SCHEMAS)
    assert recipes["square"]["generation"]["train_episodes"] == 8000
    assert recipes["franka_basket"]["generation"]["test_episodes_per_gravity"] == 200


def test_planar_evaluation_manifest_is_complete_and_locked() -> None:
    root = Path(__file__).resolve().parents[1]
    report = validate_evaluation_manifest(
        root / "data_generation/manifests/planar_test_gstrat200_seed42.json"
    )
    assert report["status"] == "pass"
    assert report["source_ids"] == 5000
    assert report["sha256"] == "5677dace2341ecaa167de5a99a8cb450007d6645e7558b657537d44103859c5a"
