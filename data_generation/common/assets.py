"""Pinned, content-addressed provisioning for MuJoCo Menagerie assets."""

from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
import tempfile
from collections.abc import Iterable
from pathlib import Path
from typing import Any

MENAGERIE_REPOSITORY = "https://github.com/google-deepmind/mujoco_menagerie.git"
MENAGERIE_REVISION = "c1a4eeb85694ae1dffe33ff1797d4e528928a133"
MENAGERIE_COMPONENTS = {
    "unitree_z1": {
        "files": 27,
        "tree_sha256": "c42c396bb085186c8bcf573c2f560438a25d3c0d410eb3fb330b88a3c2012030",
    },
    "franka_emika_panda": {
        "files": 80,
        "tree_sha256": "f449a52826e248bc67c063d80f1ef8e02e93602972263b8993aad019a41d8ec2",
    },
}


def _tree_sha256(root: Path) -> tuple[int, str]:
    files = sorted(path for path in root.rglob("*") if path.is_file())
    digest = hashlib.sha256()
    for path in files:
        digest.update(path.relative_to(root).as_posix().encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return len(files), digest.hexdigest()


def verify_menagerie_component(root: str | Path, component: str) -> dict[str, Any]:
    if component not in MENAGERIE_COMPONENTS:
        raise ValueError(f"unknown Menagerie component {component!r}")
    component_root = Path(root).expanduser().resolve() / component
    if not component_root.is_dir():
        raise FileNotFoundError(f"missing Menagerie component: {component_root}")
    file_count, tree_hash = _tree_sha256(component_root)
    expected = MENAGERIE_COMPONENTS[component]
    if file_count != expected["files"] or tree_hash != expected["tree_sha256"]:
        raise ValueError(
            f"Menagerie {component} integrity failure: files={file_count}, sha256={tree_hash}"
        )
    return {
        "component": component,
        "files": file_count,
        "tree_sha256": tree_hash,
        "status": "pass",
    }


def _fetch_checkout(destination: Path, components: tuple[str, ...]) -> None:
    subprocess.run(["git", "init", "--quiet", str(destination)], check=True)
    subprocess.run(
        ["git", "-C", str(destination), "remote", "add", "origin", MENAGERIE_REPOSITORY],
        check=True,
    )
    subprocess.run(
        [
            "git",
            "-C",
            str(destination),
            "-c",
            "protocol.version=2",
            "fetch",
            "--quiet",
            "--depth=1",
            "--filter=blob:none",
            "origin",
            MENAGERIE_REVISION,
        ],
        check=True,
    )
    subprocess.run(
        ["git", "-C", str(destination), "checkout", "--quiet", "FETCH_HEAD", "--", *components],
        check=True,
    )


def provision_menagerie_assets(
    destination: str | Path,
    *,
    components: Iterable[str] = tuple(MENAGERIE_COMPONENTS),
    source_checkout: str | Path | None = None,
) -> dict[str, Any]:
    """Copy only selected robot assets from the pinned upstream revision."""

    destination = Path(destination).expanduser().resolve()
    if destination.exists():
        raise FileExistsError(f"refusing to overwrite asset destination: {destination}")
    selected = tuple(dict.fromkeys(components))
    if not selected or any(name not in MENAGERIE_COMPONENTS for name in selected):
        raise ValueError(f"components must be selected from {sorted(MENAGERIE_COMPONENTS)}")

    if source_checkout is None:
        with tempfile.TemporaryDirectory(prefix="sg-jepa-menagerie-") as temporary:
            checkout = Path(temporary) / "checkout"
            _fetch_checkout(checkout, selected)
            destination.mkdir(parents=True)
            for component in selected:
                shutil.copytree(checkout / component, destination / component)
    else:
        checkout = Path(source_checkout).expanduser().resolve()
        destination.mkdir(parents=True)
        for component in selected:
            verify_menagerie_component(checkout, component)
            shutil.copytree(checkout / component, destination / component)

    verified = [verify_menagerie_component(destination, component) for component in selected]
    manifest = {
        "schema_version": 1,
        "repository": MENAGERIE_REPOSITORY,
        "revision": MENAGERIE_REVISION,
        "license": "Apache-2.0",
        "components": verified,
        "status": "pass",
    }
    (destination / "UPSTREAM.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


__all__ = [
    "MENAGERIE_COMPONENTS",
    "MENAGERIE_REPOSITORY",
    "MENAGERIE_REVISION",
    "provision_menagerie_assets",
    "verify_menagerie_component",
]
