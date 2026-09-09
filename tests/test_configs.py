import json
from pathlib import Path

from data_generation.generate import TASKS
from sg_jepa.baselines.training import load_dino_training_config
from sg_jepa.config import load_experiment_config
from sg_jepa.control.config import load_policy_bundle_config


def test_all_public_configs_parse() -> None:
    root = Path(__file__).resolve().parents[1]
    for path in sorted((root / "configs/train").glob("*.yaml")):
        if path.name.endswith("_dino_wm.yaml"):
            load_dino_training_config(path)
        else:
            load_experiment_config(path)
    for directory in (root / "configs/control", root / "examples/configs"):
        for path in sorted(directory.glob("*.yaml")):
            load_policy_bundle_config(path)


def test_dino_paper_configs_use_locked_optimizer_recipe() -> None:
    root = Path(__file__).resolve().parents[1]
    for path in sorted((root / "configs/train").glob("*_dino_wm.yaml")):
        _name, _predictor, training, _raw = load_dino_training_config(path)
        assert training.batch_size == 32
        assert training.weight_decay == 0.01


def test_public_task_and_pretrained_manifests_are_minimal() -> None:
    root = Path(__file__).resolve().parents[1]
    assert TASKS == (
        "right_triangle",
        "square",
        "approach_ball",
        "arm_catcher_ball",
        "arm_paddle_ball",
        "franka_basket",
    )
    payload = json.loads((root / "sg_jepa/pretrained.json").read_text())
    assert len(payload["artifacts"]) == 8
    assert payload["organization"] == "sg-jepa"
    assert payload["repo_type"] == "dataset"
    assert set(payload["repository_revisions"]) == {"sg-jepa/sg-jepa"}
    assert payload["repository_revisions"]["sg-jepa/sg-jepa"] == (
        "a0345d0101f99251029fdbe7991b73c5ff763328"
    )
    assert set(payload["repository_revisions"]) == {
        record["repo_id"] for record in payload["artifacts"].values()
    }
    assert payload["dinov2"]["source_revision"] == ("85a24602099d397264d5b30461ad7f3bfd726ca1")
    assert not payload["dinov2"]["mirrored"]


def test_paper_environment_is_separate_and_pins_cuda_12_8() -> None:
    root = Path(__file__).resolve().parents[1]
    requirements = (root / "requirements-paper-cu128.txt").read_text()

    assert "https://download.pytorch.org/whl/cu128" in requirements
    assert "torch==2.11.0+cu128" in requirements
    assert "torchvision==0.26.0+cu128" in requirements
    assert "torchaudio==2.11.0+cu128" in requirements
    assert "cuda-toolkit==12.8.1" in requirements
    assert "-cu13" not in requirements
    assert "-e /" not in requirements
