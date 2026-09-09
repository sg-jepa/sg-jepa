# Semigroup-JEPA datasets

Generate the datasets locally with `data_generation`. Generated data are not
stored in Git or uploaded to Hugging Face. The task recipes define the image
resolution and action schema:

| Task ID | Resolution | Action schema |
|---|---:|---|
| `right_triangle` | 128 | `impulse_x, impulse_z, g` |
| `square` | 128 | `impulse_x, impulse_z, g` |
| `approach_ball` | 256 | `g` |
| `arm_catcher_ball` | 256 | `g, dx, dy, dz` |
| `arm_paddle_ball` | 256 | `g, dx, dy, dz, phi, theta` |
| `franka_basket` | 256 | `g, dx, dy, dz, phi, theta` |

## Lance contract

Datasets contain one row per frame. All task families expose these columns:

| Column | Meaning |
|---|---|
| `episode_idx` | contiguous local episode index |
| `source_episode_index` | stable ID in the canonical generation schedule |
| `step_idx` | zero-based frame index within the episode |
| `split_id` | `0` for train, `1` for test |
| `pixels` | JPEG-encoded RGB frame |
| `state` | task-specific simulator state |
| `action` | full action, including gravity where specified above |
| `gravity` | signed gravity scalar |
| `phys` | task-specific physical parameters |
| `task` | public task ID |
| `episode_metadata` | JSON metadata needed to reconstruct the episode |

Robot tasks add fields such as `arm_state`, `paddle_state`, `catcher_state`,
contacts, and success labels. Schema metadata records the precise state,
action, and physics coordinate names, image size, frame rate, episode length,
and split mapping.

Use `sg_jepa.data.validate_trajectory_dataset` to check a generated store
before training:

```python
from sg_jepa.data import validate_trajectory_dataset

print(validate_trajectory_dataset("/path/to/output", expected_task="approach_ball"))
```

The paper recipes use 64-frame episodes sampled at 16 Hz. Pixel hashes are not
used as a portability criterion because MuJoCo rendering can differ across GPU
and driver versions; scene parameters, simulator state, schemas, splits, and
source episode IDs are deterministic.
