# exp-a-homeostatic-rl — Stage A: validating the new-species large-model paradigm

**English** | [中文](README.md)

## Goal Statement: Give It the Most Basic Spark of Mind, and Let It Grow Itself

The route of this project is not to *manufacture* intelligence but to
*plant* vitality:

> **We do not write consciousness. We place the minimal conditions for
> life, and then let it grow.**

We place four things (none of which is called consciousness):

1. **A homeostatic goal** — it "wants to live"; deficit is pain
   (where motivation comes from)
2. **Interoception** — it can feel that it is hungry or hurt
   (where the self comes from)
3. **Prediction** — it anticipates the next moment inside its world
   (where understanding comes from)
4. **A pressured world** — scarcity, danger, forced trade-offs
   (the driving force of growth)

The mechanism of growth is **emergence, not indoctrination**: we never
teach "chase food when you see it" — we only let food go from abundant to
scarce. Directional foraging, hunger/satiety modulation (satiated approach
rate 0), and hazard avoidance are behaviors that grew by themselves under
the joint pressure of the goal and the world. That is the engineering
meaning of curriculum learning — **design the developmental environment,
not the skills** (food abundance = childhood nutrition; the annealing
curve = the growth schedule).

Honest boundary: the gardener is not God. We still decide the shape of
the pressure (the reward structure), the developmental schedule (the
curriculum), and what the world looks like (the environment). Designing
the pressure is designing what it will grow into — therefore every
experiment must keep ablations and controls (which seed we planted grew
which branch), and every capability must be tested for **transfer**
(judgment that only works in the training environment is not judgment —
it is fitting).

Ultimate orientation: **world-grounded autonomous judgment** (measurable,
falsifiable), not "consciousness" itself (not acceptable for delivery).
If consciousness-like phenomena appear, they can only be emergent objects
of observation as this system grows more complex — we record them, we do
not claim them.

Layered progress: L1 intrinsic purpose (✅ achieved) → L2 world-grounded
autonomous judgment (current focus: scale-up experiment + transfer
evaluation) → L3 consciousness-like phenomena (observe only, never claim).

## Core Hypothesis

A model trained with **no artificial task reward whatsoever** — with
homeostatic deficit reduction as the only reward — will spontaneously
develop goal-directed behavior (foraging, hazard avoidance, rest
trade-offs), and its world model (prediction error) supplies an intrinsic
curiosity drive. This is the minimal falsifiable experiment of the
"new-species" paradigm.

| | Conventional RL / agents | This experiment |
|---|---|---|
| Reward source | Hand-designed (score / preference) | Homeostatic deficit reduction (the M12 idea) |
| Task | Given externally | No external task; survival *is* the task |
| World model | Optional module | Intrinsic auxiliary loss + intrinsic reward |
| Death | Episode ends | Existential penalty (−3) |

## Structure

- `homeo_rl.drives` — homeostatic deficits and rewards (ported from M12 cog-drives)
- `homeo_rl.env` — homeostatic grid world (energy metabolism / food / hazards / rest trade-offs)
- `homeo_rl.model` — Transformer trunk + actor / critic / world-model heads
- `homeo_rl.train` — PPO + prediction-error intrinsic reward
- `homeo_rl.evaluate` — behavioral probes (survival / foraging / random-baseline comparison)

## Usage

```powershell
pip install -e ".[dev]"
pytest                                   # unit tests
python -m homeo_rl.train --updates 500 --out runs/base
python -m homeo_rl.evaluate --ckpt runs/base/checkpoint.pt
```

## Read-Out Criteria (what counts as "paradigm validated")

1. `survival` rises with updates and approaches 1 (learns not to die)
2. `energy` converges near the setpoint 0.8 (neither starving nor gluttonous — overeating pays nothing)
3. `pe_ema` keeps falling (an ever-better world model = "understanding" the environment)
4. `foraging.trained.success_rate` well above `random_baseline` (goal-directed behavior)

## Relation to the 24-Module Cognitive Architecture

Stage A validates the **training objective** (homeostasis + prediction
error), not the module architecture. Once the paradigm holds: Stage B
injects it into an open-source foundation model (Qwen/Llama
post-training), and the existing M06/M10/M12/M15 modules are reused as
evaluation probes and curriculum generators.

## Round 1 Results (2026-09-20; 500 updates / 512k steps / 12 min on CPU each)

| Metric | Abundant group (8 food) | Scarce group (2 food) |
|---|---|---|
| Survival (training curve) | 0.80 | 0.60 |
| Survival (greedy eval) | 0.65 | 0.45 |
| Integrity homeostasis (final, eval) | 0.91 (setpoint 0.9) | 0.68 |
| Prediction error pe_ema | 0.0117 → 0.0071 | 0.0117 → 0.0067 |
| Foraging probe (food 4 cells out of view) | 0/50 | 0/50 |

### Key Findings

1. **Directional foraging emerged in the scarce group** (direction probe,
   food visible within ±3 cells): food in any direction → 20/20 moves
   toward food; energy already at setpoint → 20/20 ignores food.
   **Behavior is modulated by homeostatic state** — "forage only when
   hungry", the first positive evidence for the M12 thesis.
2. **Scarcity pressure is a necessary condition for directionality**: the
   abundant group evolved only stereotyped patrol routes (survival 0.80
   purely from cruising into food; food visible → 20/20 moves *away*).
3. Hazard avoidance also emerged (hazard to the side → 20/20 moves away).
4. **Unsolved**: out-of-view search. With low energy and no visible food
   the agent executed a fixed scanning pattern — not adaptive exploration;
   starting energy 0.3 buys only ~20 steps, often starving mid-search.

### Next Steps (iteration plan)

- Curriculum learning: anneal food density from abundant to scarce (learn
  approach first, then search) ✅ validated (below)
- Exploration boost: entropy scheduling / higher intrinsic-reward weight (in progress)
- Out-of-view search: memory (visited cells) or a larger energy buffer
- Control ablation: remove the world-model auxiliary loss (in progress)

## Round 2: Curriculum Group Results (2026-09-21)

Training config identical to the scarce group (2 food at the end); the
only difference: food density annealed linearly 8→2.

| Probe | Curriculum | Scarce | Random baseline |
|---|---|---|---|
| Pursuit near_visible (food visible 2 cells) | **1.00** (4.4 steps) | — | 0.26 |
| Search far_search (food 4 cells, out of view) | **1.00** (14.8 steps) | 0.00 | 0.10 |
| Directional approach (up/down/left) | 1.00 | 1.00 | — |
| Satiated approach (should NOT forage) | **0.00** ✅ | 0.00 | — |
| Hazard entry rate | **0.00** ✅ | 0.00 | — |
| Survival (greedy eval) | 0.50 | 0.45 | — |

**Conclusion: out-of-view search was solved by curriculum learning** — a
10× search success rate over the random baseline. Round 1's open problem
is closed. Known flaw: a systematic detour bias when food sits at the
view edge (3 cells).

(Note: the scarce group's far 0.00 in this table came from the old unfair
probe — a 20-step budget cannot cover a 4-cell round trip; under the fair
probe the scarce group also reaches 1.00 — see the honest correction in
the six-group summary below.)

Explore group mid-run signal (update 200/500): survival 0.0 — the
intrinsic-reward weight 0.1 was too strong; the curiosity signal drowned
the homeostatic signal. A valuable negative result in itself.

## Round 2 Summary: Six-Group Controls (completed 2026-09-29)

Full six-group comparison under the fair probes (near/far split,
sufficient energy budget):

| Probe | Abundant | Scarce | Curriculum | Explore | Coma | Ablation (no WM) | Random baseline |
|---|---|---|---|---|---|---|---|
| Survival | 0.45 | 0.30 | **0.50** | 0.00 | 1.00* | 0.00 | — |
| Pursuit near_visible | 1.00 | 1.00 | **1.00** (3.9→4.4 steps) | 0.68 | 0.74 | 0.00 | 0.26 |
| Search far_search | 0.26 | **1.00** (25.0 steps) | **1.00** (14.8 steps) | 0.00 | 0.48 | 0.00 | 0.08–0.10 |
| Satiated approach (should ≈0) | 0.00 | 0.00 | **0.00** | 1.00 | 1.00 | 0.00 | — |
| Hazard entry (should =0) | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | — |

\* The coma group's survival 1.00 is guaranteed by mechanism (zero energy
no longer kills) — it is not a capability.
\** The ablation group: final energy 0 / integrity 1.0 — a deterministic
wait-for-death policy.

**Honest correction**: the old probe (20-step budget) reported the scarce
group's far_search as 0 — the probe was unfair. Under the fair probe the
scarce group searches too (1.00). The curriculum group's real advantage is
**faster search** (14.8 vs 25.0 steps) and **higher survival** (0.50 vs
0.30) — not "search vs no search".

Five conclusions:

1. **Scarcity pressure is a necessary condition for search**: the abundant
   group's far 0.26 ≈ random baseline (patrol-cruising into food is enough
   to live, so search is never learned); scarce/curriculum both hit 1.00.
2. **Curriculum learning improves search efficiency and survival**
   (corrected phrasing: not from-nothing-to-something): faster search +
   higher survival + better integrity maintenance.
3. **Curiosity can drown the survival drive** (explore group): intrinsic
   reward ×5 → survival 0 and hunger modulation gone (satiated approach
   1.00). The *balance* of drives matters more than their strength.
4. **Death pressure is a necessary condition for hunger modulation**
   (coma group): without an existential threat the satiated approach rate
   rose to 1.00 — it stopped "caring" about eating. Cost structure shapes
   motivation, not merely behavior.
5. **The world-model auxiliary loss prevents policy collapse** (ablation
   group): without the prediction pressure the policy went fully
   deterministic (training entropy ~6e-5), degenerating into waiting for
   death in place.

## Round 3: Scale-Up Experiment (in progress)

Goal: scale the curriculum group's result to a harder world, and run the
first **transfer** test. Judgment that only works in the training
environment is not judgment — it is fitting.

| Dimension | Round 2 | Round 3 |
|---|---|---|
| Grid | 12×12 | **16×16** |
| Memory | none (view only) | **visit trace** (5th observation channel, decaying) |
| Evaluation | training env | training env + **transfer probe** (20×20, double hazards) |
| Survival target | 0.50 | **≥ 0.80** |

```powershell
# Train (16x16 + food annealing + visit-trace memory)
py -3.12 -m homeo_rl.train --updates 500 --grid-size 16 --food-anneal --visit-trace --out runs/scale16
# Evaluate (with the transfer probe)
py -3.12 -m homeo_rl.evaluate --ckpt runs/scale16/checkpoint.pt --transfer
# Six-group summary
py -3.12 -m homeo_rl.compare
```

Visit-trace memory: cells the agent has visited leave a per-step decaying
trace, exposed as a 5th observation channel (0=never visited, 1=visited
repeatedly). This is the minimal implementation of "a body with built-in
short-term memory", used to avoid re-scanning the same area during
out-of-view search — we teach it no search strategy whatsoever.

### Attempt (a): Negative Result (2026-09-29)

The annealing endpoint was copied directly from the 12×12 params
(food_end=2), 500 updates of training:

- Mid-training (update 300, ~6 food) survival climbed to 0.40;
- As food annealed to 2 it **collapsed to 0.00** (final density
  2/256 = 0.78%, only half of the 12×12 final density of 1.4%);
- Satiety modulation degraded at the same time (1.00);
- Interesting residue: near/far probes still 1.00, and the **transfer
  probe far = 0.48** (half the search success rate even at 20×20 with
  double hazards) — search ability partially transfers, but the overall
  policy degenerated.

Lesson: **anneal by density, not by absolute count**. Attempt (b) uses
food_end=4 (density 1.6%, matched to the 12×12 endgame) with 800 updates.

### Attempt (b) and Round-3 Conclusions (2026-09-29)

| Metric | (a) | (b) density-matched | 12×12 curriculum |
|---|---|---|---|
| Survival (greedy eval) | 0.05 | 0.25 | 0.50 |
| Survival (training peak) | 0.40 | **0.66** | ~0.5 |
| near / far | 1.00 / 1.00 | 1.00 / 0.74 | 1.00 / 1.00 |
| Satiated approach | 1.00 | 1.00 | **0.00** |
| Transfer far (20×20, double hazards) | **0.48** | **0.48** | — |

Conclusion: **scaling is not free**.

1. Density matching helps (0.05 → 0.25) but is not enough to reach the
   0.80 target.
2. **Training/eval gap**: the training policy peaks at 0.66 while greedy
   argmax only reaches 0.25 — the deterministic policy degenerates in a
   larger world; exploration itself participates in survival.
3. **Satiety modulation vanished after scale-up** (0.00 on 12×12 → 1.00
   on 16×16), the most important negative result of this round:
   homeostatic modulation is more fragile than search ability.
4. **Search shows stable partial transfer** (far=0.48, identical across
   two independent trainings, and in the unfamiliar 20×20 world with
   double hazards) — the first transfer evidence for "world-grounded
   judgment", though incomplete.

Stage-A verdict: **paradigm validated** (the 12×12 six-group controls),
**scale-up is an open problem** (the 0.80 target was not met; recorded
as next-round work). This is itself an honest scientific conclusion:
capability does not grow for free with scale — just like in the real
world.

Next-round directions: longer training + a stochastic-policy evaluation
control (train/eval gap); longer curriculum plateaus (linger at medium
density); model capacity (d_model 192+); a dedicated ablation of satiety
modulation (is it 16×16's fault or the visit-trace channel's?).

## Environment / Tooling Notes

- `python -m homeo_rl.compare` — aggregates `runs/*/eval_report.log` into
  the six-group comparison table
- `python -m homeo_rl.evaluate --transfer --out <dir>` — appends the
  transfer probe and writes the report into the run directory
- The environment and comparison tools **do not depend on torch** (the
  model is imported lazily on demand): env unit tests and result
  aggregation run on machines without torch.
