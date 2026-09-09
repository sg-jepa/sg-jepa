"""Provision the exact frozen DINOv2 encoder used by the paper baselines."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
import urllib.error
import urllib.request
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn

DINOV2_SOURCE_COMMIT = "85a24602099d397264d5b30461ad7f3bfd726ca1"
DINOV2_SOURCE_URL = "https://github.com/facebookresearch/dinov2.git"
DINOV2_WEIGHTS_FILENAME = "dinov2_vits14_pretrain.pth"
DINOV2_WEIGHTS_URL = (
    "https://dl.fbaipublicfiles.com/dinov2/dinov2_vits14/dinov2_vits14_pretrain.pth"
)
DINOV2_WEIGHTS_BYTES = 88_283_115
DINOV2_WEIGHTS_SHA256 = "b938bf1bc15cd2ec0feacfe3a1bb553fe8ea9ca46a7e1d8d00217f29aef60cd9"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git(*args: str, cwd: Path | None = None) -> str:
    result = subprocess.run(
        ["git", *args],
        cwd=cwd,
        check=True,
        capture_output=True,
        text=True,
        env={**os.environ, "GIT_TERMINAL_PROMPT": "0"},
    )
    return result.stdout.strip()


def verify_dinov2_artifact(root: str | Path) -> dict[str, Any]:
    """Fail closed if the source revision or encoder payload differs."""

    root = Path(root).expanduser().resolve()
    source = root / "source"
    weights = root / DINOV2_WEIGHTS_FILENAME
    if not (source / ".git").is_dir():
        raise FileNotFoundError(f"DINOv2 source checkout is missing: {source}")
    revision = _git("rev-parse", "HEAD", cwd=source)
    if revision != DINOV2_SOURCE_COMMIT:
        raise ValueError(f"DINOv2 source revision differs: {revision}")
    if not weights.is_file() or weights.stat().st_size != DINOV2_WEIGHTS_BYTES:
        raise FileNotFoundError("DINOv2 ViT-S/14 weights are absent or have the wrong byte size")
    digest = _sha256(weights)
    if digest != DINOV2_WEIGHTS_SHA256:
        raise ValueError(f"DINOv2 ViT-S/14 weight hash differs: {digest}")
    return {
        "schema_version": 1,
        "model": "dinov2_vits14",
        "feature_key": "x_norm_patchtokens",
        "source": str(source),
        "source_revision": revision,
        "weights": str(weights),
        "weights_bytes": weights.stat().st_size,
        "weights_sha256": digest,
        "status": "pass",
    }


def _download_weights(destination: Path) -> None:
    try:
        with urllib.request.urlopen(DINOV2_WEIGHTS_URL, timeout=120) as response:
            with destination.open("wb") as handle:
                shutil.copyfileobj(response, handle, length=8 * 1024 * 1024)
    except (OSError, TimeoutError, urllib.error.URLError):
        subprocess.run(
            [
                "curl",
                "--fail",
                "--location",
                "--retry",
                "3",
                "--silent",
                "--show-error",
                "--output",
                str(destination),
                DINOV2_WEIGHTS_URL,
            ],
            check=True,
        )


def provision_dinov2_artifact(destination: str | Path) -> dict[str, Any]:
    """Atomically fetch the pinned DINOv2 source and ViT-S/14 weights."""

    destination = Path(destination).expanduser().resolve()
    if destination.exists():
        raise FileExistsError(f"refusing to overwrite DINOv2 destination: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(
        tempfile.mkdtemp(prefix=f".{destination.name}.partial-", dir=destination.parent)
    )
    try:
        source = temporary / "source"
        _git("init", "--quiet", str(source))
        _git("remote", "add", "origin", DINOV2_SOURCE_URL, cwd=source)
        _git(
            "-c",
            "protocol.version=2",
            "fetch",
            "--quiet",
            "--depth=1",
            "--filter=blob:none",
            "origin",
            DINOV2_SOURCE_COMMIT,
            cwd=source,
        )
        _git("checkout", "--quiet", "--detach", "FETCH_HEAD", cwd=source)
        _download_weights(temporary / DINOV2_WEIGHTS_FILENAME)
        report = verify_dinov2_artifact(temporary)
        report["source"] = "source"
        report["weights"] = DINOV2_WEIGHTS_FILENAME
        report["source_url"] = DINOV2_SOURCE_URL
        report["weights_url"] = DINOV2_WEIGHTS_URL
        (temporary / "artifact.json").write_text(json.dumps(report, indent=2) + "\n")
        temporary.replace(destination)
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return verify_dinov2_artifact(destination)


class Native128Preprocessor:
    """Exact native-128 image contract used by the paper DINO-WM runs."""

    image_size = 128
    encoder_image_size = 112

    @staticmethod
    def _as_bchw(pixels: Any) -> torch.Tensor:
        value = pixels if torch.is_tensor(pixels) else torch.as_tensor(pixels)
        if value.ndim == 3:
            value = value.unsqueeze(0)
        if value.ndim != 4:
            raise ValueError(f"expected a 3D/4D image tensor, got {tuple(value.shape)}")
        if value.shape[-1] in {1, 3, 4} and value.shape[1] not in {1, 3, 4}:
            value = value.permute(0, 3, 1, 2)
        if value.shape[1] == 1:
            value = value.repeat(1, 3, 1, 1)
        if value.shape[1] < 3:
            raise ValueError(f"expected RGB pixels, got {tuple(value.shape)}")
        return value[:, :3]

    def normalized_128(self, pixels: Any) -> torch.Tensor:
        value = self._as_bchw(pixels)
        integer_input = not torch.is_floating_point(value)
        value = value.float()
        if integer_input or (value.numel() and float(value.detach().max()) > 2.0):
            value = value / 255.0
        height, width = value.shape[-2:]
        if min(height, width) != self.image_size:
            scale = self.image_size / min(height, width)
            resized = (int(round(height * scale)), int(round(width * scale)))
            value = F.interpolate(
                value,
                size=resized,
                mode="bilinear",
                align_corners=False,
                antialias=True,
            )
        height, width = value.shape[-2:]
        top = (height - self.image_size) // 2
        left = (width - self.image_size) // 2
        value = value[..., top : top + self.image_size, left : left + self.image_size]
        if value.shape[-2:] != (self.image_size, self.image_size):
            raise ValueError(f"could not center-crop pixels shaped {(height, width)}")
        return value.mul(2.0).sub(1.0)

    def __call__(self, pixels: Any) -> torch.Tensor:
        return F.interpolate(
            self.normalized_128(pixels),
            size=(self.encoder_image_size, self.encoder_image_size),
            mode="bilinear",
            align_corners=False,
            antialias=True,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "input_contract": "jpeg_rgb_uint8",
            "dataset_resize": self.image_size,
            "dataset_center_crop": self.image_size,
            "normalization_mean": [0.5, 0.5, 0.5],
            "normalization_std": [0.5, 0.5, 0.5],
            "encoder_resize": self.encoder_image_size,
            "resize_mode": "bilinear",
            "align_corners": False,
            "antialias": True,
        }


@contextmanager
def _source_path(source: Path):
    source_text = str(source)
    sys.path.insert(0, source_text)
    previous = sys.dont_write_bytecode
    sys.dont_write_bytecode = True
    try:
        yield
    finally:
        sys.dont_write_bytecode = previous
        try:
            sys.path.remove(source_text)
        except ValueError:  # pragma: no cover - defensive against external mutation.
            pass


class FrozenDinoV2Encoder(nn.Module):
    """Frozen ViT-S/14 wrapper returning the 64 normalized patch tokens."""

    feature_key = "x_norm_patchtokens"
    patch_size = 14
    embedding_dim = 384

    def __init__(self, backbone: nn.Module) -> None:
        super().__init__()
        self.backbone = backbone

    def train(self, mode: bool = True):
        del mode
        super().train(False)
        self.backbone.eval()
        return self

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        tokens = self.backbone.forward_features(images)[self.feature_key]
        if tokens.ndim != 3 or tuple(tokens.shape[-2:]) != (64, self.embedding_dim):
            raise ValueError(f"expected DINO tokens [B,64,384], got {tuple(tokens.shape)}")
        return tokens


def load_dinov2_encoder(
    root: str | Path,
    *,
    device: str | torch.device = "cpu",
) -> FrozenDinoV2Encoder:
    """Construct and strictly load the verified paper DINOv2 encoder."""

    report = verify_dinov2_artifact(root)
    source = Path(report["source"])
    with _source_path(source):
        from dinov2.models.vision_transformer import vit_small

        backbone = vit_small(
            patch_size=14,
            img_size=518,
            init_values=1.0,
            block_chunks=0,
        )
    state = torch.load(report["weights"], map_location="cpu", weights_only=True)
    backbone.load_state_dict(state, strict=True)
    backbone.requires_grad_(False).eval()
    return FrozenDinoV2Encoder(backbone).to(torch.device(device)).eval()


__all__ = [
    "DINOV2_SOURCE_COMMIT",
    "DINOV2_WEIGHTS_BYTES",
    "DINOV2_WEIGHTS_FILENAME",
    "DINOV2_WEIGHTS_SHA256",
    "FrozenDinoV2Encoder",
    "Native128Preprocessor",
    "load_dinov2_encoder",
    "provision_dinov2_artifact",
    "verify_dinov2_artifact",
]
