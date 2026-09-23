# Autopoiesis

**[中文版](README.zh-CN.md)** | English

*Autopoiesis* (Greek: "self-creation", Maturana & Varela) — a system that
produces itself. This monorepo engineers the minimal conditions for life in
a learning model, then lets it grow itself.

Implemented according to the 24+3 module plan. Each module = an independent project
(own directory, pyproject, tests, versioning, and docs), ready to be split into
standalone Git repositories at any time. Modules communicate only through
explicitly defined interfaces.

## Direction Shift: From Agent to a "New-Species Large Model"

The positioning of this repository has been upgraded: the cognitive modules
(M01–M27) are no longer the final deliverable (a runtime agent scaffold), but
**evaluation probes and curriculum generators for a new-species large model**.
The core paradigm has three stages:

- **Stage A (in progress)** — `exp-a-homeostatic-rl`: train a small Transformer
  from scratch where the *only* reward is homeostatic deficit reduction
  (the M12 idea) + prediction-error intrinsic reward (world model). This
  validates whether "vitality" emerges without any artificial task reward.
  **Requires torch; run with `py -3.12`.**
- **Stage B (planned)**: inject homeostatic + world-model objectives into an
  open-source foundation model via post-training.
- **Stage C (planned)**: online lifelong learning; training and deployment
  become one.

```powershell
cd exp-a-homeostatic-rl
py -3.12 -m pip install -e ".[dev]"
py -3.12 -m pytest
py -3.12 -m homeo_rl.train --updates 500 --out runs/base
py -3.12 -m homeo_rl.evaluate --ckpt runs/base/checkpoint.pt
```

## Current Progress

| Phase | Modules | Status |
|---|---|---|
| 1 | M01 cog-config, M02 cog-logger, M03 cog-checkpoint | **Done** |
| 2 | M04 cog-perception, M05 cog-policy, M12 cog-drives, M25 cog-body, M26 cog-world | **Done (172 tests all passing)** |
| 3 | M06 cog-working-memory, M14 cog-workspace, M15 cog-goal-manager | **Done (264 tests all passing)** |
| 4 | M10 world model, M16 planner | To develop |
| 5 | M07/M08/M09 three memory types | To develop |
| 6 | M13 emotion, M17 metacognition, M18 self-narrative | To develop |
| 7 | M19 grounding, M20 inner speech, M21 social speech | To develop |
| 8 | M11 causal graph, M22 ethics, M23 visualization | To develop |
| 9 | M27 multi-agent society | To develop |
| 10 | M24 integration (cog-agent) | Ongoing |

## Directory Layout (identical per module)

```
cog-xxx/
├── README.md / CHANGELOG.md / docs/api.md
├── pyproject.toml
├── src/cog_xxx/{__init__.py, core.py, types.py}
├── tests/{test_core.py, test_interface.py, test_edge.py}
└── examples/demo.py
```

## Running Tests

```powershell
.\run_tests.ps1    # per-module isolated runs (preserving the "independent tests" principle)
```

## Technical Decisions

- **NumPy backend**: current Python 3.14 has no stable torch, so M04/M05 use
  self-implemented micro-networks (`nn.py`: Linear/Tanh/Sequential + hand-written
  backprop) with torch-aligned naming (`forward/parameters/zero_grad/backward/sgd_step`),
  so upper layers need zero changes when switching to torch.
- **Module dependencies**: M26 depends on cog-body (the agent *is* a Body);
  M12 directly consumes M25's effect dictionaries.

## Completed Modules at a Glance

| Module | Version | Interface | Tests |
|---|---|---|---|
| cog-config | 0.1.0 | `Config.load(path) -> Config` | 24 |
| cog-logger | 0.1.0 | `log(event)` / `metric(name, value, step)` | 14 |
| cog-checkpoint | 0.1.0 | `Checkpoint.save/load` | 21 |
| cog-perception | 0.1.0 | `Encoder(obs) -> z` (tanh-bounded, multimodal) | 22 |
| cog-policy | 0.1.0 | `Policy(z) -> (DiagGaussian, value)` | 23 |
| cog-drives | 0.1.0 | `step(effects)` / `deficit()` / `to_tensor()` | 24 |
| cog-body | 0.1.0 | `step(action) -> effects` | 22 |
| cog-world | 0.1.0 | `observe(id)` / `step(id, action) -> (obs, reward)` | 22 |
| cog-working-memory | 0.1.0 | `write(z)` / `read(query) -> z` (FIFO + cosine attention) | 24 |
| cog-workspace | 0.1.0 | `forward(perc, mem, emo, goal) -> (broadcast, weights)` | 20 |
| cog-goal-manager | 0.1.0 | `add(goal)` / `update_from_drives()` / `current()` (zero-dep) | 28 |

## Typical Data Flow (Phase 3, working end-to-end)

```
World.observe -> Encoder -> z ──┬─► WorkingMemory.write(z) ─► read(query)
                                ├─► Workspace(perc=z, mem=..., emo=..., goal=...)
                                │     └► (broadcast, weights)  # attention focus
Drives.deficit ──► GoalManager.update_from_drives ─► current()  # current goal
```

## Typical Data Flow (Phase 2, working end-to-end)

```
World.observe(id) -> obs (6,) -> Encoder -> z (128,) -> Policy -> action
World.step(id, action) -> effects -> DriveSystem.step(effects) -> deficit
```

## Next Steps (Phase 4, ~3 weeks)

1. M10 cog-world-model: `forward(z, a) -> (z', r, d)` / `imagine(z, policy, H)`
   (latent transition model + rollout, NumPy micro-network backend).
2. M16 cog-planner: `plan(z, goal) -> actions` (CEM/MPC planning on top of M10).

## Next Steps (Phase 5, preview)

M07 episodic memory (similarity retrieval), M08 semantic memory (knowledge
graph + spreading activation), M09 procedural memory (skill library).

---

For the full Stage A experiment log, goal statement, and the "cultivation
philosophy" (place the minimal conditions for life; let it grow itself),
see [`exp-a-homeostatic-rl/README.md`](exp-a-homeostatic-rl/README.md).
