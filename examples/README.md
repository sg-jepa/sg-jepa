# Examples

This directory contains all example-specific shell entry points, policy
configs, runtime logic, and optional Slurm wrappers. Generic Python utilities
are called from `scripts/`. Run commands from the repository root. Every stage
requires an NVIDIA CUDA GPU.

## Interface

```bash
RUN_MODE=smoke bash examples/square.sh data|world-model|probe|evaluate
RUN_MODE=smoke bash examples/franka_basket.sh data|world-model|policy|evaluate
RUN_MODE=smoke bash examples/arm_paddle_ball.sh data|world-model|policy|evaluate
```

Control evaluation accepts an optional seed slot from `0` through `4`. After
all five full-size control evaluations finish, run the `aggregate` stage.

Stages are independent and reuse complete outputs. World-model and policy
training resume from their latest checkpoint when an interrupted output is
present. `RUN_MODE` defaults to `smoke`; use `RUN_MODE=full` explicitly for a
full-size run.

## Environment

The entry points recognize:

- `SGJEPA_PYTHON`: Python executable for the installed environment.
- `SGJEPA_DATA_ROOT`: parent directory for generated datasets.
- `SGJEPA_OUTPUT_ROOT`: parent directory for checkpoints and reports.
- `SGJEPA_MENAGERIE_ROOT`: location of the pinned MuJoCo assets.
- `EXAMPLE_TAG`: subdirectory shared by all stages of one run.
- `SGJEPA_CPUS`: local worker count when Slurm does not set one.

Direct invocations default to `datasets/`, `output/`, and
`outputs/menagerie_assets/` inside the checkout. Select explicit storage roots
for longer runs.

## Slurm

The optional launcher submits data and compute arrays while preserving the
same stage implementation used by the direct entry points:

```bash
EXAMPLE_TAG=my-run bash examples/submit_slurm.sh smoke all
EXAMPLE_TAG=my-run bash examples/submit_slurm.sh full world-model
EXAMPLE_TAG=my-run bash examples/submit_slurm.sh full all
```

Cluster resource defaults live in `examples/slurm/*.sbatch`. Adjust the
partition and GPU resource directives for a different Slurm installation.
