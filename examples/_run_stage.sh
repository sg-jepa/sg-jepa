#!/usr/bin/env bash

set -euo pipefail

usage() {
  cat >&2 <<'EOF'
usage: bash examples/_run_stage.sh STAGE TASK [SEED_SLOT]

STAGE is one of data, world-model, downstream, probe, policy, evaluate, or
aggregate. TASK is square, franka_basket, arm_paddle_ball, or its array index
0, 1, or 2. Control evaluation SEED_SLOT is an integer from 0 through 4.
EOF
  exit 2
}

stage="${1:-}"
task_selector="${2:-}"
seed_slot="${3:-0}"
[[ -n "$stage" && -n "$task_selector" && $# -le 3 ]] || usage

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
export SGJEPA_ROOT="${SGJEPA_ROOT:-$(cd -- "$script_dir/.." && pwd -P)}"
source "$script_dir/_runtime.sh"

case "$task_selector" in
  0|square) task_index=0 ;;
  1|franka_basket) task_index=1 ;;
  2|arm_paddle_ball) task_index=2 ;;
  *) echo "unknown example task: $task_selector" >&2; usage ;;
esac
select_example_task "$task_index"

set_episode_counts() {
  if [[ "$RUN_MODE" == "smoke" ]]; then
    episodes="${SMOKE_EPISODES:-32}"
    train_episodes="$((episodes / 2))"
    test_episodes="$((episodes - train_episodes))"
  else
    episodes="$FULL_EPISODES"
    train_episodes="$FULL_TRAIN_EPISODES"
    test_episodes="$FULL_TEST_EPISODES"
  fi
}

run_data_stage() {
  set_episode_counts
  if [[ "$RUN_MODE" == "smoke" ]]; then
    generation_args=(--episodes "$episodes")
  else
    generation_args=()
  fi

  if [[ -e "$DATASET_ROOT" ]]; then
    if [[ -f "$DATASET_ROOT/generation_manifest.json" ]]; then
      verify_dataset "$episodes" "$train_episodes" "$test_episodes"
      echo "dataset already complete: $DATASET_ROOT"
      return 0
    fi
    echo "refusing to replace incomplete dataset directory: $DATASET_ROOT" >&2
    return 3
  fi

  mkdir -p "$DATA_RUN_ROOT"
  if [[ "$TASK" != "franka_basket" ]]; then
    "$PYTHON_BIN" -m data_generation.generate \
      --recipe data_generation/recipes/main_text.yaml \
      --task "$TASK" \
      --split all \
      --menagerie-root "$MENAGERIE_ROOT" \
      --output "$DATASET_ROOT" \
      "${generation_args[@]}"
    verify_dataset "$episodes" "$train_episodes" "$test_episodes"
    return 0
  fi

  if [[ "$RUN_MODE" == "full" ]]; then
    shard_starts=(0 3250 6500 9750)
    shard_counts=(3250 3250 3250 3250)
    shard_train_counts=(3250 3250 1500 0)
    shard_test_counts=(0 0 1750 3250)
  else
    if ((episodes < 4)); then
      echo "Franka smoke generation requires at least four episodes" >&2
      return 2
    fi
    train_a="$((train_episodes / 2))"
    train_b="$((train_episodes - train_a))"
    test_a="$((test_episodes / 2))"
    test_b="$((test_episodes - test_a))"
    shard_starts=(0 "$train_a" 8000 "$((8000 + test_a))")
    shard_counts=("$train_a" "$train_b" "$test_a" "$test_b")
    shard_train_counts=("$train_a" "$train_b" 0 0)
    shard_test_counts=(0 0 "$test_a" "$test_b")
  fi

  shard_root="${DATA_RUN_ROOT}/.franka_basket-shards"
  mkdir -p "$shard_root"
  available_cpus="${SLURM_CPUS_PER_TASK:-${SGJEPA_CPUS:-8}}"
  threads_per_shard="$((available_cpus / 4))"
  ((threads_per_shard > 0)) || threads_per_shard=1
  export OMP_NUM_THREADS="$threads_per_shard"
  export MKL_NUM_THREADS="$OMP_NUM_THREADS"
  stage_run_id="${SLURM_JOB_ID:-interactive-$$}"
  shard_paths=()
  shard_logs=()
  shard_pids=()

  generate_franka_shard() {
    local shard_index="$1"
    local source_start="${shard_starts[$shard_index]}"
    local shard_episodes="${shard_counts[$shard_index]}"
    local shard_train="${shard_train_counts[$shard_index]}"
    local shard_test="${shard_test_counts[$shard_index]}"
    local source_stop="$((source_start + shard_episodes))"
    local shard_path="${shard_root}/source-${source_start}-${source_stop}"
    local temporary="${shard_path}.partial-${stage_run_id}-${shard_index}"

    if [[ -e "$shard_path" ]]; then
      if [[ ! -f "$shard_path/generation_manifest.json" ]]; then
        echo "refusing to replace incomplete Franka shard: $shard_path" >&2
        return 3
      fi
      "$PYTHON_BIN" scripts/verify_examples.py data \
        --path "$shard_path" \
        --task franka_basket \
        --episodes "$shard_episodes" \
        --train-episodes "$shard_train" \
        --test-episodes "$shard_test"
      echo "Franka shard already complete: $shard_path"
      return 0
    fi
    if [[ -e "$temporary" ]]; then
      echo "refusing to replace stale Franka shard staging directory: $temporary" >&2
      return 3
    fi
    "$PYTHON_BIN" -m data_generation.generate \
      --recipe data_generation/recipes/main_text.yaml \
      --task franka_basket \
      --split all \
      --episodes "$shard_episodes" \
      --source-start "$source_start" \
      --menagerie-root "$MENAGERIE_ROOT" \
      --output "$temporary"
    "$PYTHON_BIN" scripts/verify_examples.py data \
      --path "$temporary" \
      --task franka_basket \
      --episodes "$shard_episodes" \
      --train-episodes "$shard_train" \
      --test-episodes "$shard_test"
    mv -- "$temporary" "$shard_path"
  }

  for shard_index in 0 1 2 3; do
    source_start="${shard_starts[$shard_index]}"
    source_stop="$((source_start + shard_counts[shard_index]))"
    shard_paths+=("${shard_root}/source-${source_start}-${source_stop}")
    shard_logs+=("${shard_root}/source-${source_start}-${source_stop}.log")
    generate_franka_shard "$shard_index" >"${shard_logs[$shard_index]}" 2>&1 &
    shard_pids+=("$!")
  done

  shard_failure=0
  for shard_index in 0 1 2 3; do
    if ! wait "${shard_pids[$shard_index]}"; then
      shard_failure=1
    fi
    sed "s/^/[franka-shard-${shard_index}] /" "${shard_logs[$shard_index]}"
  done
  if ((shard_failure)); then
    echo "at least one Franka generation shard failed" >&2
    return 4
  fi

  merge_args=(
    --output "$DATASET_ROOT"
    --episodes "$episodes"
    --train-episodes "$train_episodes"
    --test-episodes "$test_episodes"
    --seed 20260807
  )
  for shard_path in "${shard_paths[@]}"; do
    merge_args+=(--shard "$shard_path")
  done
  if [[ "$RUN_MODE" == "full" ]]; then
    merge_args+=(--paper-scale)
  fi
  "$PYTHON_BIN" scripts/merge_franka_shards.py "${merge_args[@]}"
  verify_dataset "$episodes" "$train_episodes" "$test_episodes"
}

run_world_model_stage() {
  set_episode_counts
  verify_dataset "$episodes" "$train_episodes" "$test_episodes"
  mkdir -p "$(dirname -- "$WM_OUTPUT")"

  if [[ "$RUN_MODE" == "smoke" ]]; then
    if [[ -f "$WM_OUTPUT/train_report.json" ]]; then
      completed_steps="$($PYTHON_BIN -c 'import json,sys; print(json.load(open(sys.argv[1]))["steps_after"])' "$WM_OUTPUT/train_report.json")"
    else
      completed_steps=0
    fi
    if ((completed_steps < 1)); then
      "$PYTHON_BIN" train.py \
        --config "$WM_CONFIG" \
        --data "$DATASET_ROOT" \
        --output-dir "$WM_OUTPUT" \
        --device cuda \
        --max-steps 1
      completed_steps=1
    fi
    if ((completed_steps < 2)); then
      "$PYTHON_BIN" train.py \
        --config "$WM_CONFIG" \
        --data "$DATASET_ROOT" \
        --output-dir "$WM_OUTPUT" \
        --device cuda \
        --resume "$WM_OUTPUT/checkpoint.pt" \
        --max-steps 1
    fi
    "$PYTHON_BIN" scripts/verify_examples.py world-model \
      --output "$WM_OUTPUT" \
      --config "$WM_CONFIG" \
      --minimum-steps 2
    return 0
  fi

  if [[ -f "$WM_OUTPUT/train_report.json" ]]; then
    completed_epochs="$($PYTHON_BIN -c 'import json,sys; print(json.load(open(sys.argv[1]))["completed_epochs"])' "$WM_OUTPUT/train_report.json")"
    if ((completed_epochs == 20)); then
      "$PYTHON_BIN" scripts/verify_examples.py world-model \
        --output "$WM_OUTPUT" \
        --config "$WM_CONFIG" \
        --completed-epochs 20
      echo "world model already complete: $WM_OUTPUT"
      return 0
    fi
  fi

  resume_args=()
  if [[ -f "$WM_OUTPUT/checkpoint.pt" ]]; then
    resume_args=(--resume "$WM_OUTPUT/checkpoint.pt")
  else
    shopt -s nullglob
    epoch_checkpoints=("$WM_OUTPUT"/checkpoint_epoch_*.pt)
    shopt -u nullglob
    if ((${#epoch_checkpoints[@]})); then
      resume_args=(--resume "${epoch_checkpoints[-1]}")
    fi
  fi
  "$PYTHON_BIN" train.py \
    --config "$WM_CONFIG" \
    --data "$DATASET_ROOT" \
    --output-dir "$WM_OUTPUT" \
    --device cuda \
    "${resume_args[@]}"
  "$PYTHON_BIN" scripts/verify_examples.py world-model \
    --output "$WM_OUTPUT" \
    --config "$WM_CONFIG" \
    --completed-epochs 20
}

run_downstream_stage() {
  set_episode_counts
  if [[ "$RUN_MODE" == "smoke" ]]; then
    "$PYTHON_BIN" scripts/verify_examples.py world-model \
      --output "$WM_OUTPUT" --config "$WM_CONFIG" --minimum-steps 2
  else
    "$PYTHON_BIN" scripts/verify_examples.py world-model \
      --output "$WM_OUTPUT" --config "$WM_CONFIG" --completed-epochs 20
  fi
  verify_dataset "$episodes" "$train_episodes" "$test_episodes"
  mkdir -p "$(dirname -- "$DOWNSTREAM_OUTPUT")"

  if [[ "$TASK" == "square" ]]; then
    if [[ -f "$DOWNSTREAM_OUTPUT/probe_summary.json" ]]; then
      "$PYTHON_BIN" scripts/verify_examples.py report \
        --path "$DOWNSTREAM_OUTPUT/probe_summary.json"
      echo "Square probe already complete: $DOWNSTREAM_OUTPUT"
      return 0
    fi
    resume_args=()
    if [[ -f "$DOWNSTREAM_OUTPUT/probe_latest.pt" ]]; then
      resume_args=(--resume "$DOWNSTREAM_OUTPUT/probe_latest.pt")
    fi
    if [[ "$RUN_MODE" == "smoke" ]]; then
      probe_args=(
        --max-train-windows 128
        --max-val-windows 64
        --feature-batch-size 8
        --batch-size 32
        --max-epochs 1
        --patience 1
        --num-workers 2
      )
    else
      probe_args=(--num-workers 8)
    fi
    "$PYTHON_BIN" train_probe.py \
      --task square \
      --config "$WM_CONFIG" \
      --checkpoint "$WM_OUTPUT/checkpoint.pt" \
      --data "$DATASET_ROOT" \
      --output-dir "$DOWNSTREAM_OUTPUT" \
      --split "$WM_OUTPUT/train_val_split.json" \
      --device cuda \
      "${probe_args[@]}" \
      "${resume_args[@]}"
    "$PYTHON_BIN" scripts/verify_examples.py report \
      --path "$DOWNSTREAM_OUTPUT/probe_summary.json"
    return 0
  fi

  if [[ -f "$DOWNSTREAM_OUTPUT/training.json" ]]; then
    completed_steps="$($PYTHON_BIN -c 'import json,sys; print(json.load(open(sys.argv[1]))["steps_after"])' "$DOWNSTREAM_OUTPUT/training.json")"
  else
    completed_steps=0
  fi
  if [[ "$RUN_MODE" == "full" ]] && ((completed_steps == 300000)); then
    "$PYTHON_BIN" scripts/verify_examples.py report \
      --path "$DOWNSTREAM_OUTPUT/training.json" --minimum-steps 300000
    echo "Diffusion policy already complete: $DOWNSTREAM_OUTPUT"
    return 0
  fi

  policy_base=(
    --config "$POLICY_CONFIG"
    --data "$DATASET_ROOT"
    --world-model-config "$WM_CONFIG"
    --world-model-checkpoint "$WM_OUTPUT/checkpoint.pt"
    --split "$WM_OUTPUT/train_val_split.json"
    --output-dir "$DOWNSTREAM_OUTPUT"
    --device cuda
  )
  if [[ "$RUN_MODE" == "smoke" ]]; then
    smoke_sampling_args=()
    if [[ "$TASK" == "arm_paddle_ball" ]]; then
      # The fixed soft-quality artifact is keyed to the 8,000-episode
      # training population. A newly generated 32-episode smoke cohort has
      # different source IDs, so use the deterministic uniform sampler while
      # preserving the configured sampler for every full run.
      smoke_sampling_args=(--batch-sampling-override uniform_epoch_without_replacement)
    fi
    if ((completed_steps < 1)); then
      "$PYTHON_BIN" train_policy.py "${policy_base[@]}" \
        "${smoke_sampling_args[@]}" --max-steps 1
      completed_steps=1
    fi
    if ((completed_steps < 2)); then
      "$PYTHON_BIN" train_policy.py "${policy_base[@]}" \
        "${smoke_sampling_args[@]}" \
        --resume "$DOWNSTREAM_OUTPUT/checkpoints/last_policy.pt" \
        --max-steps 1
    fi
    "$PYTHON_BIN" scripts/verify_examples.py report \
      --path "$DOWNSTREAM_OUTPUT/training.json" --minimum-steps 2
    return 0
  fi

  resume_args=()
  if [[ -f "$DOWNSTREAM_OUTPUT/checkpoints/last_policy.pt" ]]; then
    resume_args=(--resume "$DOWNSTREAM_OUTPUT/checkpoints/last_policy.pt")
  elif [[ -e "$DOWNSTREAM_OUTPUT" ]]; then
    echo "policy output exists without a resumable checkpoint: $DOWNSTREAM_OUTPUT" >&2
    return 3
  fi
  "$PYTHON_BIN" train_policy.py "${policy_base[@]}" "${resume_args[@]}"
  "$PYTHON_BIN" scripts/verify_examples.py report \
    --path "$DOWNSTREAM_OUTPUT/training.json" --minimum-steps 300000
}

run_evaluation_stage() {
  set_episode_counts
  verify_dataset "$episodes" "$train_episodes" "$test_episodes"

  if [[ "$TASK" == "square" ]]; then
    "$PYTHON_BIN" scripts/verify_examples.py report \
      --path "$DOWNSTREAM_OUTPUT/probe_summary.json"
    result_dir="$EVALUATION_OUTPUT"
    if [[ -f "$result_dir/evaluation.json" ]]; then
      "$PYTHON_BIN" scripts/verify_examples.py report --path "$result_dir/evaluation.json"
      return 0
    fi
    if [[ "$RUN_MODE" == "smoke" ]]; then
      cohort_args=(--max-episodes 1)
    else
      cohort_args=(
        --max-episodes 5000
        --episode-manifest data_generation/manifests/planar_test_gstrat200_seed42.json
      )
    fi
    mkdir -p "$(dirname -- "$result_dir")"
    "$PYTHON_BIN" evaluate_prediction.py \
      --task square \
      --method Semigroup-JEPA \
      --config "$WM_CONFIG" \
      --checkpoint "$WM_OUTPUT/checkpoint.pt" \
      --probe "$DOWNSTREAM_OUTPUT/probe_weights.pt" \
      --data "$DATASET_ROOT" \
      --normalization-stats "$WM_OUTPUT/normalization_stats.json" \
      --output-dir "$result_dir" \
      --device cuda \
      --horizon 44 \
      "${cohort_args[@]}"
    "$PYTHON_BIN" scripts/verify_examples.py report --path "$result_dir/evaluation.json"
    return 0
  fi

  if [[ ! "$seed_slot" =~ ^[0-4]$ ]]; then
    echo "control evaluation seed slot must be an integer from 0 through 4" >&2
    return 2
  fi
  if [[ "$TASK" == "franka_basket" ]]; then
    evaluation_seed="${EVAL_SEED:-$((${FRANKA_EVAL_BASE_SEED:-20260810} + seed_slot))}"
  else
    evaluation_seed="${EVAL_SEED:-$((${PADDLE_EVAL_BASE_SEED:-42} + seed_slot * ${PADDLE_EVAL_SEED_STRIDE:-10000019}))}"
  fi
  if [[ "$RUN_MODE" == "smoke" ]]; then
    max_episodes=1
    paper_args=()
  else
    max_episodes="$FULL_TEST_EPISODES"
    paper_args=(--paper-cohort)
  fi
  minimum_policy_steps=300000
  [[ "$RUN_MODE" == "smoke" ]] && minimum_policy_steps=2
  "$PYTHON_BIN" scripts/verify_examples.py report \
    --path "$DOWNSTREAM_OUTPUT/training.json" \
    --minimum-steps "$minimum_policy_steps"
  result_dir="${EVALUATION_OUTPUT}/seed_${evaluation_seed}"
  if [[ -f "$result_dir/evaluation.json" ]]; then
    "$PYTHON_BIN" scripts/verify_examples.py report --path "$result_dir/evaluation.json"
    return 0
  fi
  mkdir -p "$(dirname -- "$result_dir")"
  "$PYTHON_BIN" evaluate_control.py \
    --task "$TASK" \
    --method Semigroup-JEPA \
    --config "$POLICY_CONFIG" \
    --checkpoint "$DOWNSTREAM_OUTPUT/checkpoints/policy_ema_inference.pt" \
    --encoder-config "$WM_CONFIG" \
    --encoder-checkpoint "$WM_OUTPUT/checkpoint.pt" \
    --data "$DATASET_ROOT" \
    --menagerie-root "$MENAGERIE_ROOT" \
    --output-dir "$result_dir" \
    --device cuda \
    --max-episodes "$max_episodes" \
    --seed "$evaluation_seed" \
    "${paper_args[@]}"
  "$PYTHON_BIN" scripts/verify_examples.py report --path "$result_dir/evaluation.json"
}

run_aggregate_stage() {
  if [[ "$TASK" == "square" ]]; then
    echo "Square produces one prediction report and has no seed aggregate" >&2
    return 2
  fi
  "$PYTHON_BIN" scripts/aggregate_control_evaluations.py \
    --input-root "$EVALUATION_OUTPUT" \
    --output "$EVALUATION_OUTPUT/aggregate.json" \
    --expected-seeds 5
}

require_cuda
case "$stage" in
  data) run_data_stage ;;
  world-model) run_world_model_stage ;;
  downstream) run_downstream_stage ;;
  probe)
    [[ "$TASK" == "square" ]] || { echo "probe is only valid for square" >&2; exit 2; }
    run_downstream_stage
    ;;
  policy)
    [[ "$TASK" != "square" ]] || { echo "policy is only valid for control tasks" >&2; exit 2; }
    run_downstream_stage
    ;;
  evaluate) run_evaluation_stage ;;
  aggregate) run_aggregate_stage ;;
  *) echo "unknown example stage: $stage" >&2; usage ;;
esac
