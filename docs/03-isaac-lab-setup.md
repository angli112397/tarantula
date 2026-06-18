# Isaac Lab Active-Suspension Setup

Isaac Lab is used for the 6D active-suspension policy only. Motion stays under the classical skid-steer controller.

## Smoke

```bash
scripts/run_rl_env_smoke_v5.sh
```

The smoke test checks:

- observation shape `(N, 50)`;
- action shape `(N, 6)`;
- wheel F/T contact sensor is alive;
- no NaN in observations or rewards;
- reset origins are lifted above the shared heightmap.

## Train

```bash
source /home/ang/isaac_venv/bin/activate
PYTHONPATH=src:src/tarantula_control \
python3 src/tarantula_isaac/train_v5.py \
  --num_envs 64 \
  --max_iterations 400 \
  --terrain-dir "$(pwd)/generated/terrains/rl_curriculum/42" \
  --terrain-level-min 0 \
  --terrain-level-max 0 \
  --command-profile stage0
```

Start with level `0:0`, then increase terrain difficulty only after Gazebo posture acceptance passes.

## Export

```bash
python3 src/tarantula_isaac/export_weights_v5.py \
  --checkpoint logs/rsl_rl/tarantula_suspension/<run>/model_399.pt \
  --npz-out generated/policies/posture_actor.npz
```

The exported actor must be `50D/6D`; any other actor shape is rejected by the Gazebo runtime.
