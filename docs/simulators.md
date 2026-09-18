# Simulators

Show-Harness integrates ManiSkill and Isaac Lab. They serve two roles:

1. **Zero-shot evaluation** — run the deployment pipelines (the subgoal planner
   stack or the fine-tuned action model, see `docs/finetuned.md`) in sim
   with no robot attached.
2. **Sim-to-real training data** — generate demonstrations on the same action
   lattice the real robot deploys with (single-axis, ~2 cm per token), so sim
   rollouts mix directly with real teleop rollouts for fine-tuning.

Each integration keeps the deployment contracts: the same nine-token action
vocabulary and the same image transforms (`core/record/images.py`). Measure
each embodiment's step calibration before treating one token as ~2 cm of
physical travel.

## ManiSkill

Fine-tuned policy only, in ManiSkill 3's translation-only `pd_ee_delta_pos`
control mode (rotation locked — the assumption the atomic-token policy makes).

- `scripts/run_maniskill_mvtoken.py` — entry point; config `configs/robot_maniskill.yaml`.
  One config covers both protocols: it defaults to the fixed layout preset, and
  `--traj-id random --layout wide` reproduces the randomized object layout the
  training data was generated with. `--env-id` switches scenes.
- `scripts/maniskill/eval_batch.sh <config> <model> <n_episodes> [max_steps] [tag]`
  — batch eval over consecutive seeds; prints the closed-loop success rate.

ManiSkill needs its own Python environment; the
runner's remaining dependencies (numpy, requests, pyyaml, PIL, imageio) are
standard. Scenes are declared as one `SceneSpec` row each in
`core/sim/maniskill_scenes.py`. The default `BlockPAP-v1` is a real2sim replica
of the real Franka rig (table, pedestal, block + coaster, calibrated front
camera) and needs an RLinf checkout (`RLINF_ROOT`); `BlockStack-v1` is the same
rig with a stacking task. Stock tasks (`PickCube-v1`, `StackCube-v1`) run too
but with a much larger domain gap.

```bash
python scripts/run_maniskill_mvtoken.py --version v3 --model <adapter> --max-steps 60 --probe-axes
```

`--probe-axes` records the measured per-token TCP delta into `calibration.json`.

Calibration facts (load-bearing; full derivations in the yaml comments):

- `step_m: 0.026` x `sim_steps_per_decision: 2` is the commanded setting that
  achieves ~20.2 mm per decision (PD lag makes achieved < commanded) — see
  `configs/robot_maniskill.yaml`.
- `wrist_flip: both` and `agentview_square_size: 256` are training contracts;
  read the yaml comments before touching either, and regenerate data after any
  camera change.

Training-data generation lives in `scripts/trajectory/real2sim/`: a
simulator-agnostic core (`atomic_tokenizer.py` — token vocabulary, closed-loop
2 cm execution, Manhattan/RDP/chase planners, teleop-format writer) plus one
backend per simulator (`backends/maniskill.py`, `backends/robolab.py`). It
produces rollouts in exactly the real-teleop format, so sim and real data mix
without special cases. See `scripts/trajectory/real2sim/README.md`.

## RoboLab (Isaac Lab)

NVIDIA's [RoboLab](https://github.com/NVLabs/RoboLab) benchmark: 120 authored
Isaac Sim manipulation tasks with automated success predicates and photoreal
rendering.

- `scripts/run_robolab_mvtoken.py` — entry point; config `configs/robot_robolab.yaml`.
- `scripts/robolab/eval_batch.sh <model> <n_episodes> [task ...]` — batch eval,
  one process per task (Isaac Sim's cold start dominates otherwise).

The RoboLab checkout location comes from the `ROBOLAB_ROOT` environment
variable (see `robolab_root()` in `core/sim/robolab_task.py`). Isaac Sim pins
Python 3.11 with its own large dependency set, so run this repo's scripts with
the RoboLab venv's interpreter — it also satisfies everything the runner needs.
First launch requires accepting Isaac Sim's EULA (`OMNI_KIT_ACCEPT_EULA=YES`),
and `libGLU.so.1` must be loadable or Isaac Sim segfaults during stage creation
with a misleading backtrace; `launch_isaac` in `core/sim/robolab_task.py`
checks for it up front and prints the fix.

```bash
python scripts/run_robolab_mvtoken.py --list-tasks                                     # no Isaac Sim needed
python scripts/run_robolab_mvtoken.py --task RubiksCubeTask --dump-views --probe-axes --no-rollout   # calibration only, no VLM
python scripts/run_robolab_mvtoken.py --version v3 --task RubiksCubeTask --episodes 5
```

`--task` takes the task class name; `--episodes N` reuses one Isaac Sim app and
env across episodes; `--gui` shows the viewport (default headless).

Embodiment. The sim Franka wears the real rig's short yellow fingertips, not
the stock black fingers — for a policy that reads pixels, the stock finger is a
distribution shift in the middle of every frame.
`assets/robolab_franka/panda_short_finger.usda` replaces both finger visuals
and collision meshes (rebuild with
`scripts/trajectory/real2sim/robolab/make_short_finger_asset.py`; set
`ROBOLAB_PANDA_USD` to compare against the stock robot). The camera geometry
also mirrors the real rig rather than RoboLab's DROID default: Panda hand, and
the wrist camera centered between the fingers looking down the grasp axis
(`core/sim/robolab_franka.py`).

Calibration facts (load-bearing; measured tables in the yaml comments):

- RoboLab's relative differential-IK achieves a constant ~28% of any commanded
  delta, so `step_m: 0.072` commanded yields ~20.1 mm measured per decision —
  see `configs/robot_robolab.yaml`, including why settle steps do not help.
- `wrist_rotation_degrees: 270` / `wrist_flip: none` are measured for the Panda
  hand (fingertips at the top of the frame, no mirror). The camera contract
  lives only in this yaml; the runner and the data generators both read it via
  `core.config.camera_contract()` — never restate it elsewhere.

Training-data generation uses the same real2sim core. The RoboLab oracle parses
each task's own subtask declaration, so any single-object pick-and-place task
among the 120 generates without code changes (others raise `UnsupportedTask`).
Generate per task with `real2sim/robolab/record_demos.py` then
`follow_tokenize.py` (commands in
[real2sim/README.md](../scripts/trajectory/real2sim/README.md)), looping over the
task names `--list-tasks` prints to build a set. Quality gate before training:
`python scripts/robolab/check_dataset.py <dir>` (nonzero exit = do not train on
it). Prefer rotation-insensitive objects — the vocabulary has no wrist-rotation
token, so elongated objects are ungraspable.

## Single-arm Piper (Isaac Lab)

`scripts/run_piper_isaaclab_mvtoken.py` runs one AgileX Piper in a standalone
Isaac Lab 2.2 / Isaac Sim 5 scene. It does **not** use RoboLab's Franka-only
tasks or require `ROBOLAB_ROOT`. The first task places a red cube on a table;
success means the closed gripper lifts it 8 cm while keeping it near the TCP.
The scene exposes a fixed front camera and an arm-mounted wrist camera. A
translation-only differential IK controller moves the midpoint of the two
fingers; the two finger joints receive open/close position targets. Piper's
home wrist pose is singular for a fixed-orientation IK solve, so tool
orientation is free during movement.

Get AgileX's [Piper Isaac Sim asset](https://github.com/agilexrobotics/piper_isaac_sim)
and keep its `piper_description_v100_realsense_camera_v2` directory intact.
The entry USD references four companion layers in `configuration/`; a single
`.usd` file is insufficient. By default the runner expects the checkout under
`~/piper_isaac_sim`; use `--piper-usd /absolute/path/to/...usd` or `PIPER_USD`
for another location. Run with the Python interpreter that has Isaac Lab and
Isaac Sim installed, and with a working NVIDIA RTX driver:

```bash
git clone https://github.com/agilexrobotics/piper_isaac_sim.git ~/piper_isaac_sim
<isaaclab-python> scripts/run_piper_isaaclab_mvtoken.py --no-rollout --dump-views --probe-axes --probe-gripper --probe-pick
<isaaclab-python> scripts/run_piper_isaaclab_mvtoken.py --version v3 --max-steps 80
```

The first command runs without a VLM endpoint. It writes raw and
policy-transformed PNGs, a picked-cube frame and `calibration.json` under
`rollouts/piper_isaaclab/`. On the configured scene, the six move tokens were
measured at about 2 cm with about 1 mm off-axis drift, and the scripted
grasp lifted the cube beyond the 8 cm success threshold. Re-run these probes
after changing the USD, cube pose or camera setup. The sim adapter named in
the config was trained on ManiSkill and RoboLab, not this Piper scene; use it
for an integration smoke test, then collect Piper data and fine-tune before
drawing conclusions about policy task performance.

To try the hosted DeepSeek vision API, set `DEEPSEEK_API_KEY` in the gitignored
`configs/secrets.env` (or export it in the shell), then select the optional
`deepseek` profile:

```bash
<isaaclab-python> scripts/run_piper_isaaclab_mvtoken.py --vlm-backend deepseek --max-steps 80
```

Longer pick-and-place variants add a visible green target pad. They exercise
grasping, repeated lifting, horizontal transport, lowering, and release:

```bash
<isaaclab-python> scripts/run_piper_isaaclab_mvtoken.py --vlm-backend deepseek --task pick_place_left --max-steps 80
<isaaclab-python> scripts/run_piper_isaaclab_mvtoken.py --vlm-backend deepseek --task pick_place_forward --max-steps 80
```

Use `--no-rollout --probe-place` with either variant to verify the complete
physical trajectory without consuming API calls. The task-specific
`piper_deepseek_pick_place.txt` prompt receives measured TCP, cube, target, and
finger state. Successful placement requires the released cube to settle within
4 cm of the configured target center, followed by an open-gripper retreat and
return to the measured home TCP. The episode does not terminate at release.

This profile uses the current `deepseek-flash` model and
`https://api.deepseek.com/chat/completions`. The front and wrist images are sent
as base64 `image_url` parts in the user message, with thinking disabled and a
Piper-specific action prompt. The runner supplies the current TCP position,
measured finger gap, and a phase inferred from these robot sensors alongside
the two images, so this is a state-assisted
zero-shot integration, not an image-only Piper-trained policy. The prompt tells
the model when to grasp at the minimum TCP height and to lift repeatedly after contact;
the runner checks the authenticated `/models` list before launching Isaac Sim,
so an unavailable model fails early. The previous
`deepseek-v4-flash-vision-exp` name has been retired; DeepSeek routes requests
using that name to `deepseek-flash`. The deterministic `--probe-pick` above
verifies the simulation independently
of model behavior. See the [DeepSeek vision guide](https://api-docs.deepseek.com/zh-cn/guides/vision/)
for the current model and image-input contract.

For a GPT comparison through a HeyRoute Codex relay, put `HEYROUTE_API_KEY` in
the gitignored `configs/secrets.env` and select `--vlm-backend heyroute`.
This profile sends the two camera PNGs as `input_image` blocks to
`https://heyroute.ai/v1/responses` with model `gpt-5.6-sol`. It still uses the
generic MVTOKEN prompt; the DeepSeek profile selects `piper_deepseek.txt` for
pick-lift and `piper_deepseek_pick_place.txt` for the longer tasks.
Use `--prompt-profile generic` to run DeepSeek with the original shared
`mvtoken_generator_lite.txt` prompt, or `--prompt-profile deepseek` to force the
state-assisted adapter with another backend. The explicit switch supports a
controlled prompt A/B test; `auto` remains the default.

That prompt-profile switch is still a flat, planner-free MVTOKEN loop. To run the
repository's full original zero-shot stack, select `--policy-stack
original_zero_shot`. This calls `plugins/subgoal/subgoal_planner.txt` on the first
observation, then drives each visual stage with `prompts/controller.txt` plus the
enabled formal prompt plugins (coordinates, wrist frame, ego mapping,
proprioception, memory, variable step, and recovery). Stage-level `DONE` advances
to the next planned stage; it does not end the episode. The simulator also advances
an achieved stage from measured grasp width, TCP/target error, placement contact,
release state, retreat height, and home-pose error. These checks prevent a late
visual action from undoing an already verified milestone. The adapted prompts
remain available under the default `--policy-stack adapted` mode.

```bash
<isaaclab-python> scripts/run_piper_isaaclab_mvtoken.py \
  --policy-stack original_zero_shot --vlm-backend deepseek \
  --task pick_place_left --max-steps 80
```

For this formal DeepSeek path, the runner enables thinking mode with high reasoning
effort and raises the output allowance to at least 2048 tokens. Planner JSON calls
remain deterministic and do not request a reasoning monologue. Controller logs keep
both the final action and `reasoning_content` for inspection.

Each run saves the generated plans as `zero_shot_plan_*.json`, the rendered
generic controller requests under `controller_prompts/`, the per-step decisions,
and the rollout MP4.
Test one two-image request (`--max-steps 1`) before running a full episode:

```bash
<isaaclab-python> scripts/run_piper_isaaclab_mvtoken.py --vlm-backend heyroute --max-steps 1
```

HeyRoute account model permissions and image-input support must be verified with
a live request; a text-only `OK` response does not establish either. The profile
is transport-tested with a mocked Responses result, but still requires a
successful live vision request before a rollout.

To measure the task ceiling imposed by fixed-axis primitives independently of
vision or prompt quality, run the paired Piper oracle benchmark:

```bash
<isaaclab-python> scripts/run_piper_primitive_limits.py --trials 3
```

It adds two non-pick-place tasks. `precision_diagonal_reach` requires the TCP to
enter a 2 mm diagonal goal region. `narrow_diagonal_wipe` requires endpoint
completion and at least 95% of physics samples to remain inside an 8 mm-wide
diagonal strip and within 4 mm of the starting surface height. Each reset is run
once with the released fixed-length, single-axis MVTOKEN interface and once with
an arbitrary Cartesian displacement whose norm is capped at the same 1 cm. Both
controllers receive perfect simulator state, so the measured success gap isolates
the action representation rather than VLM perception or planning. Every trial
writes camera video with a top-down task map, per-step records, and a JSON result
under `rollouts/piper_primitive_limits/`.

## RealMan RMC-AIDA-L with dual RM65-B-V (Isaac Lab)

`scripts/run_realman_dual_isaaclab.py` imports the complete official RMC-AIDA-L
RM65-B-V URDF as one synchronized Isaac Lab articulation. The left arm moves a
red cube to a green pad while the right arm
moves a blue cube to a yellow pad. One front camera and two arm-mounted wrist
cameras feed the existing dual policy contract. A fourth, higher Isaac overview
camera is recorded to `images/overview/` and the analysis MP4 but is not sent to
the model, so the three-image wire order remains unchanged. Both seven-dimensional relative-IK
commands are applied before each physics step, so the arms actually move together.

The scene includes the official mobile chassis, wheel geometry, lift column,
dual-arm mounts, both RM65-B-V arms, camera housings and four-bar grippers. For
these manipulation runs the articulation root is fixed, both wheel joints are
locked, and the 900 mm lift joint is held at one configured working height. This
validates manipulation with the complete mounted geometry; navigation and dynamic
lift motion are outside the current action space.

The robot geometry comes from RealMan's official
[`Ecosystem_Cases`](https://github.com/RealManRobot/Ecosystem_Cases/tree/RMC-AIDA-L)
repository and is not copied into Show-Harness. The setup command sparse-clones
the `RMC-AIDA-L` branch and exact `Embodied lifting robot_two wheels_RM65-B-V`
package. At runtime `core/sim/realman_assets.py` resolves all package mesh paths,
corrects the exported lift mesh's stale package prefix, locks the wheel joints,
and caches the prepared full-body URDF. The official gripper linkage and collision
geometry remain in use.

```bash
bash scripts/realman/setup_sim_assets.sh

# Camera, direction and gripper diagnostics; no API call.
<isaaclab-python> scripts/run_realman_dual_isaaclab.py \
  --no-rollout --dump-views --probe-axes --probe-gripper

# Full dual-arm pick, place, release, retreat and return-home acceptance run.
# It records the normal analysis MP4 and consumes no API quota.
<isaaclab-python> scripts/run_realman_dual_isaaclab.py --scripted --dump-views

# Exercise distinct forward, inward, outward and split-depth target layouts.
for task in parallel_forward diagonal_inward diagonal_outward split_depth; do
  <isaaclab-python> scripts/run_realman_dual_isaaclab.py --scripted --task "$task"
done

# Three-image DeepSeek closed loop using the RealMan-specific prompt.
<isaaclab-python> scripts/run_realman_dual_isaaclab.py \
  --vlm-backend deepseek --max-steps 90

# Original dual-arm zero-shot planner + generic controller_dual.txt.
<isaaclab-python> scripts/run_realman_dual_isaaclab.py \
  --policy-stack original_zero_shot --vlm-backend deepseek \
  --task parallel_forward --max-steps 90
```

`--task` selects a reproducible preset from
`configs/robot_realman_dual_isaaclab.yaml`. Each preset keeps the cube and home
poses fixed while changing the two target pads, which tests forward/backward and
lateral transport without changing the controller or prompt contract. Output
folders include the task name so videos from a batch do not collide.

The acceptance predicate requires both cubes to settle on their assigned pads,
both grippers to be open, and both TCPs to return to their measured home positions.
`DONE DONE` is rejected before that predicate passes. The `--scripted` path uses
the same measured phases and synchronized token executor as the VLM path, which
makes it the physics and closure test to run after any model, camera or controller
change.

The RealMan DeepSeek prompt treats each measured phase as an authoritative lookup
key. Camera images remain in the request, but they cannot override mappings such
as `LIFT_WITH_CUBE -> MV_UP` or `CARRY_BACK -> MV_BACK`. This matters in long
rollouts: a looser visual instruction can start transport before reaching the safe
height or oscillate after overshooting a target.

The `original_zero_shot` mode instead calls
`plugins/subgoal/subgoal_planner_dual.txt` and executes the two returned tracks
with `prompts/controller_dual.txt` and the configured formal prompt plugins. The
planner must preserve explicit closure instructions as separate `RELEASE`,
`RETREAT`, and `RETURN` stages. Per-arm `DONE` advances only that arm. Measured
completion checks also advance a stage when its physical predicate is already true;
the placement guard lifts before correcting low-altitude XY drift, then descends only
after the TCP is aligned. If a stage exceeds `v0.max_subgoal_steps`, the current
images are sent back through the original planner up to `v0.max_replans` times.
This mode does not inject the RealMan phase label into the VLM prompt, but it does
render measured TCP, gripper, cube, and goal feedback through the proprioception and
coordinate plugins. It is therefore a state-assisted original-prompt zero-shot
evaluation, rather than a pure image-only baseline.
