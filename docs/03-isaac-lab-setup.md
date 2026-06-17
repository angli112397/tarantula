# Isaac Lab Setup

当前 Isaac Lab 入口只服务 baseline 链路：

- shared heightmap terrain smoke;
- DirectRLEnv smoke;
- lightweight PPO smoke training;
- actor export to Gazebo deployment `.npz`.

## Environment

```bash
source ~/isaac_venv/bin/activate
export OMNI_KIT_ACCEPT_EULA=Y
export PYTHONPATH=$(pwd)/src:$(pwd)/src/tarantula_control:${PYTHONPATH:-}
```

## Smoke Checks

Shared Gazebo/Isaac terrain importer:

```bash
scripts/isaac_shared_terrain_smoke.sh
```

Direct RL environment:

```bash
scripts/run_rl_env_smoke_v5.sh
```

Geometry/spawn sanity check:

```bash
scripts/isaac_geometry_check.sh
```

## Lightweight Training

```bash
NUM_ENVS=2 scripts/run_ppo_train_v5.sh \
  --max_iterations 1 \
  --terrain-dir "$(pwd)/generated/terrains/gazebo_demo/42"
```

## Export

```bash
python3 src/tarantula_isaac/export_weights_v5.py \
  --checkpoint logs/rsl_rl/tarantula_suspension/<run>/model_0.pt \
  --npz-out generated/policies/cmd_vel_actor.npz
```

The exported `.npz` is consumed by `tarantula_control.motion_control_node` when
`rl_compensation_enabled:=true`.

## Deterministic Eval

Run this before any Gazebo judgement. The `open_loop` mode is the Isaac-side
motion-control baseline; the `npz` mode evaluates the exported structured actor
on the same command sequence.

```bash
python3 src/tarantula_isaac/eval_policy_v5.py \
  --mode open_loop \
  --num-envs 16 \
  --terrain-dir "$(pwd)/generated/terrains/gazebo_demo/42" \
  --out generated/benchmarks/isaac_eval/open_loop_summary.json

python3 src/tarantula_isaac/eval_policy_v5.py \
  --mode npz \
  --policy-npz generated/policies/cmd_vel_actor.npz \
  --num-envs 16 \
  --terrain-dir "$(pwd)/generated/terrains/gazebo_demo/42" \
  --out generated/benchmarks/isaac_eval/policy_summary.json
```

Reject a policy before Gazebo if Isaac eval shows immediate spawn termination,
near-zero commanded displacement, large command-tracking error, or persistent
action saturation.
