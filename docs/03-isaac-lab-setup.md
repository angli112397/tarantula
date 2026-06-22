# Isaac Lab Active-Suspension Setup

Isaac Lab is used for the 6D active-suspension policy only. Motion stays under the classical skid-steer controller.

## Smoke

```bash
scripts/run_rl_env_smoke_v5.sh
```

The smoke test checks:

- observation shape `(N, 56)`;
- action shape `(N, 6)`;
- wheel joint reaction wrench (`ArticulationData.body_incoming_joint_wrench_b`) is alive;
- no NaN in observations or rewards;
- reset origins are lifted above the shared heightmap.

## Train

```bash
source /home/ang/isaac_venv/bin/activate
PYTHONPATH=src:src/tarantula_control \
python3 src/tarantula_isaac/train_v5.py \
  --num_envs 128 \
  --max_iterations 2000 \
  --terrain-dir "$(pwd)/generated/terrains/rl_curriculum/42" \
  --command-profile stage0 \
  --pursuit-prob 0.3
```

The original plan here was a manually-staged curriculum: start training
restricted to `--terrain-level-min/max 0:0` (easiest row only), then re-run
with the range widened once Gazebo posture acceptance passed. That widening
step never actually happened -- every run stayed pinned to the easiest 2 of
4 difficulty rows, including by default even without passing the flags
(`shared_heightmap_terrain.py` silently capped `max_init_terrain_level` to 1
regardless), so training never experienced the harder terrain types. Fixed
2026-06-20: `--terrain-level-min/max` no longer exist on `train_v5.py` --
`suspension_env.py`'s `_reset_idx` now re-rolls a random (row, col) tile
from the terrain's *full* difficulty range on every single reset, so every
run already covers the whole curriculum without a manual staging step.
(`gui_smoke.py` still accepts `--terrain-level-min/max` for restricting its
own smoke runs to easy terrain, where predictability matters more than
coverage.)

`--pursuit-prob` opts into pure-pursuit checkpoint-chasing commands
(`CommandsCfg.pursuit_prob`, default 0.0/off) on top of whichever
`--command-profile` is chosen; omit it to keep the profile's own primitive/
mission mix unchanged.

## Domain randomization

`DomainRandCfg` (`suspension_env_cfg.py`) widens training to cover the Isaac
PhysX <-> Gazebo DART sim-to-sim gap: `friction_range=(0.05,1.75)` and
`hip_stiffness_scale_range`/`hip_damping_scale_range=(0.8,1.2)` are on by
default (no flag needed); body mass and push perturbations stay opt-in by
`--command-profile` (e.g. `stage0` enables push DR) since they're more likely
to destabilize an unproven policy.

## Export

```bash
python3 src/tarantula_isaac/export_weights_v5.py \
  --checkpoint logs/rsl_rl/tarantula_suspension/<run>/model_399.pt \
  --npz-out generated/policies/posture_actor.npz
```

The exported actor must be `56D/6D`; any other actor shape is rejected by the Gazebo runtime.

## Evaluate in Gazebo

Load the exported `.npz` via `posture_policy_enabled:=true
policy_weights_npz:=...` on `sim.launch.py`, then compare against a frozen
(`posture_policy_enabled:=false`) baseline with the same seed/checkpoints
using `scripts/gazebo_pursuit_eval.py` + `scripts/gazebo_eval_compare.py` --
see docs/00-project-plan.md's Evaluation Plan section.
