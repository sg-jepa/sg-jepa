# Data generation

This installed package contains the complete generators for the six main-body
Semigroup-JEPA tasks. The public recipe is
`data_generation/recipes/main_text.yaml`.

```bash
python -m data_generation.generate \
  --recipe data_generation/recipes/main_text.yaml \
  --task approach_ball \
  --split train \
  --output /path/to/approach_train
```

Choose one of `right_triangle`, `square`, `approach_ball`,
`arm_catcher_ball`, `arm_paddle_ball`, or `franka_basket`. The output path must
not already exist. Omitting `--episodes` uses the paper-scale count;
`--episodes N` retains the real simulator and task constants while limiting the
run to `N` episodes. `--seed` overrides the deterministic base seed.

## Generator families

- `planar/` shares one MuJoCo implementation for Right Triangle and Square.
- `mujoco/` shares one cleaned 3D implementation for Approach, Catcher, and
  Paddle. The task-specific Catcher and Paddle changes are applied directly.
- `franka/` contains the paddle-to-basket generator and controller.
- `common/` contains the Lance writer and pinned robot-asset provisioner.

The 3D robot generators need the `unitree_z1` or `franka_emika_panda` models
from MuJoCo Menagerie. If `--menagerie-root` is omitted, the required component
is fetched at commit `c1a4eeb85694ae1dffe33ff1797d4e528928a133`, copied next
to the temporary runtime, and verified by tree hash. To reuse a checkout:

```bash
python -m data_generation.generate \
  --recipe data_generation/recipes/main_text.yaml \
  --task arm_paddle_ball --split test --episodes 2 \
  --menagerie-root /path/to/mujoco_menagerie \
  --output /path/to/paddle_validation
```

## Splits and evaluation manifests

Paper-scale generation follows the canonical training/test schedules and keeps
the original source episode IDs. The task-neutral planar evaluation manifest
`manifests/planar_test_gstrat200_seed42.json` selects 200 held-out episodes at
each of 25 gravities. The other task recipes select evaluation episodes from
their locally generated test splits.

`manifests/paddle_fps16_v2_quality_weights.npz` is a compact training input,
not generated trajectory data. It preserves the final Paddle policy launcher's
source-episode/window quality weights. Policy training verifies its pinned
SHA-256 before use and refuses a full run unless its 7,098 successful episode
IDs exactly match the generated development cohort.

All outputs use the common one-row-per-frame Lance contract described in
[`data/README.md`](../data/README.md). Generated datasets, robot assets, and
temporary shards are ignored by Git.
