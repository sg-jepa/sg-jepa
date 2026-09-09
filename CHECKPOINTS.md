# Checkpoint upload

The upload bundle is in the ignored `checkpoints/sg-jepa/` directory. It
targets [`sg-jepa/sg-jepa`](https://huggingface.co/datasets/sg-jepa/sg-jepa),
a Hugging Face dataset repository used only for checkpoint files. Generated
training and evaluation data are not included.

```text
checkpoints/sg-jepa/
├── README.md
├── square/
├── franka-basket/
└── arm-paddle/
```

The three task directories contain eight inference-only safetensors and eight
adjacent JSON sidecars, totaling about 1.39 GB. Each sidecar records the
architecture, source and converted SHA-256, and tensor equality check.

| Directory | Contents | Pairs |
|---|---|---:|
| `square/` | SG-GRU and its symmetry-aware probe | 2 |
| `franka-basket/` | SG-GRU encoder and SG-JEPA/DINO policies | 3 |
| `arm-paddle/` | SG-GRU encoder and SG-JEPA/DINO policies | 3 |

Published revision:

```text
a0345d0101f99251029fdbe7991b73c5ff763328
```

The revision is recorded in `sg_jepa/pretrained.json`. The remote file list,
LFS object sizes, and SHA-256 values match the local bundle and JSON sidecars.

Do not upload raw `.pt` files, optimizer or RNG state, experiment caches,
generated trajectory data, upstream DINOv2 weights, or the
`transformer_block_concat` ablation.

The bundle intentionally omits the Approach artifacts. It also does not include
the other five native 2D world models, the Square/Right-Triangle DINO-WM pairs,
or the Catcher SG world model. Their exact checkpoints are not present on this
host.
