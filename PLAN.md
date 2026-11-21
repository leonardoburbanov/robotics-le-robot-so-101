# SO-101: "VLM → robot" marker in / out — 0 → 1 plan

Task: tell the arm "put the yellow marker inside the paper" (or "outside") and have it
do it smoothly. Teleop for demos goes through a **virtual follower** (MuJoCo twin), not the leader.

## The strategy in one picture

```
 you: "put it inside"                                        ~1 call / episode
        │
 ┌──────▼───────────────────────┐   overhead image
 │ BRAIN  (VLM, e.g. Claude)    │◄──────────────── overhead cam (on leader gripper)
 │ - is the marker in or out?   │
 │ - pick the task string       │
 │ - check success, retry       │
 └──────┬───────────────────────┘
        │ task string (exactly one of the 2 training strings)
 ┌──────▼───────────────────────┐   overhead + wrist images + 6 joints   30 Hz
 │ POLICY (VLA: SmolVLA)        │◄──────────────── cameras + follower
 │ outputs 50-step action chunk │
 └──────┬───────────────────────┘
        │ RTC blends chunks (no jumps between chunks)
 ┌──────▼───────────────────────┐
 │ FOLLOWER  (STS3215 servos)   │  + servo acceleration limit
 └──────────────────────────────┘
```

Why this split: a VLM is good at "what/where/did it work" but too slow (seconds) to
drive motors. A VLA is good at 30 Hz motion but only knows the instructions it was
trained on. So the VLM picks one of **two exact strings**, the VLA executes it.

Why SmolVLA: it is native to LeRobot, takes a language instruction, is small (~450M) and is the
usual SO-101 starting point. pi0.5 / GR00T / X-VLA need 24–40 GB+ VRAM to fine-tune.
ACT is trained first as a fast baseline (fits your 6 GB GPU, no language input).

## Where smoothness comes from (in order of impact)

1. **Smooth demos.** The policy copies your teleop. Constant speed, no stop at every
   waypoint, no hesitation. Jerky demos → jerky policy, whatever the model.
2. **Chunk blending at run time.**
   - SmolVLA: `lerobot-rollout --inference.type=rtc` (Real-Time Chunking).
   - ACT: temporal ensembling (`n_action_steps=1`, `temporal_ensemble_coeff=0.01`).
     Last time `n_action_steps=100` was the cause of the jerk.
3. **Servo side.** STS3215 `Acceleration` register + a per-step joint delta clamp in the
   teleop/IK layer.

## Phase 0 — Freeze the hardware (½ day)

- Overhead cam stays on the leader gripper; leader parked with torque ON, **never**
  `SOLeader.connect()` (it drops the arm). Nothing may move it after recording starts.
- Wrist cam on follower. Name cameras `front` (overhead) and `wrist`; check indices by
  grabbing a frame every session (they swap on replug).
- Fixed lighting, paper taped down, same yellow marker.
- Re-calibrate follower if needed (`lerobot-calibrate`).

Exit: both cameras at 30 fps, 640x480, `lerobot-find-cameras` shows the right images.

## Phase 1 — Virtual-follower teleop (1–2 days)  ← the part to build

```
gamepad ──► end-effector velocity ──► IK on MuJoCo twin ──► joint targets
                                         │                       │
                                   twin renders           real follower (rate-limited)
```

- Model: `Simulation/SO101/so101_new_calib.xml` (official SO-ARM100 MJCF).
- Input: LeRobot's `gamepad` teleop (or `phone`). Gamepad = smoothest for hands: analog
  sticks give continuous velocity.
- IK: damped least squares on the MuJoCo Jacobian (`mj_jac`, ~15 lines) or `mink`.
  5-DOF arm → control x, y, z, pitch, roll; gripper on a trigger.
- Write it as **one LeRobot `Teleoperator` class** whose `get_action()` returns the joint
  dict. Then `lerobot-record` works unchanged and records action = what the follower got.
- Safety: clamp joint ranges to the real calibration (wrist_flex ±85.9°), clamp per-step
  delta, deadman button.
- Bonus later: same teleop can **take over mid-rollout** to record corrections (Phase 6).

Prior art to read (don't depend on it): ricklon/xbox_soarm_teleop (gamepad + IK + twin),
SinfonIAUniandes/lerobot_teleoperator_so101_ik (IK teleop plugin for lerobot-record),
Juxi-Technology/so101-mujoco (MuJoCo plugin for LeRobot).

Exit: you pick and place the marker by gamepad 5/5 times, motion looks smooth on video.

## Phase 2 — Record demos (1–2 days)

Two task strings, used **verbatim** everywhere (record, train, rollout, VLM prompt):

- `Pick the yellow marker outside the white paper and put it completely inside the paper.`
- `Pick the yellow marker inside the white paper and put it completely outside the paper.`

- 50 episodes per task to start (100 total), 30 fps, ~10–20 s each.
- Vary marker position and angle over the whole reachable area; outside = all 4 sides
  of the paper that the arm reaches.
- One consistent strategy (same approach height, same grasp: across the marker's narrow
  side, the marker is 11.7×2.7×1.5 cm and does not roll).
- ~10–15% of episodes include a recovery (miss, re-grasp). The policy needs to see
  how to fix mistakes.
- Check with `lerobot-dataset-viz` and delete bad episodes (`lerobot-edit-dataset`).

Tip: chaining helps. "Outside → inside" ends with the marker inside = a valid start for
"inside → outside". Alternate tasks and resets are almost free.

## Phase 3 — ACT baseline, local (1 day)

- `lerobot-train --policy.type=act`, batch 8, ~30–50k steps on the 3060.
- ACT has no language: train on both tasks anyway (the image tells in vs out) or one
  model per task. Purpose: prove data + cameras + calibration are right.
- Rollout with temporal ensembling.

Exit: ≥50% success per task on 10 trials. If not, the problem is data, not the model.

## Phase 4 — SmolVLA + RTC (1–2 days)

- Fine-tune from `lerobot/smolvla_base`, `--policy.type=smolvla`, ~20k steps.
- 6 GB is tight: try small batch + bf16 + frozen vision encoder; otherwise train on a
  rented GPU (Colab / HF Jobs / RunPod, a few hours) and run inference locally (fits 6 GB).
- Run: `lerobot-rollout --inference.type=rtc`; tune `execution_horizon`,
  `max_guidance_weight`.
- Known issue: RTC on SO-101 can oscillate far from training poses → cover the
  workspace in Phase 2.

Exit: ≥70% per task, no visible stutter between chunks.

## Phase 5 — The brain (VLM) (1 day)

- Loop: grab overhead frame → VLM returns JSON `{marker_inside: bool, task: "<exact string>"}`
  → run policy until done or timeout → VLM (or the color/homography check) verifies →
  retry once if failed.
- Accepts free speech ("take it off the paper", in any language) → maps to one of the 2 strings.
- Cheap cross-check without a VLM: yellow blob (LAB b) inside the paper polygon.
  Use it as the automatic success metric for evaluation too.

## Phase 6 — Iterate (ongoing)

- Evaluate: 20 trials per task on a fixed grid of start positions; log success and a
  jerk metric (mean |Δ²q| of the commanded joints).
- Where it fails: let the policy run, take over with the gamepad, record the correction,
  add to the dataset, fine-tune again (DAgger / human-in-the-loop).

## Not now (and when)

- Sim-to-real training data: skip. Sim is the teleop interface and safety preview only.
  Revisit if real demos become the bottleneck.
- pi0.5 / GR00T: only if SmolVLA plateaus and a ≥24–40 GB cloud GPU is available.
- Scripted real-robot expert (overhead homography → auto demos): if 100 manual demos
  aren't enough.

## Sources

- SmolVLA paper: https://arxiv.org/pdf/2506.01844
- SmolVLA on SO-101 (Arm): https://learn.arm.com/learning-paths/laptops-and-desktops/finetune-smolvla-lerobot/
- SmolVLA SO-101 fine-tune guide: https://openelab.io/blogs/learn/how-to-fine-tune-smolvla-on-so-101-with-lerobot
- SmolVLA cube pick-place notes: https://ggando.com/blog/smolvla-so101
- VLA comparison 2026: https://voxel51.com/blog/vla-models-compared
- LeRobot RTC docs: https://huggingface.co/docs/lerobot/rtc
- LeRobot rollout docs: https://huggingface.co/docs/lerobot/main/inference
- RTC oscillation report on SO-101: https://github.com/huggingface/lerobot/pull/4580
- Gamepad + IK + twin: https://github.com/ricklon/xbox_soarm_teleop
- IK teleop plugin: https://github.com/SinfonIAUniandes/lerobot_teleoperator_so101_ik
- SO-101 MuJoCo + LeRobot: https://github.com/Juxi-Technology/so101-mujoco
