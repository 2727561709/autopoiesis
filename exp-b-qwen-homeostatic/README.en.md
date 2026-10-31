# exp-b-qwen-homeostatic — Stage B: injecting homeostasis into a language model

**English** | [中文](README.md)

## Goal

Stage A validated the paradigm (a small Transformer whose only reward is
homeostatic deficit reduction; six-group controls hold). Stage B asks the
next question:

> **Can a language model *embody* a homeostatic policy? Is its behavior
> modulated by interoception?**

If Qwen-0.5B learns to "stay alive" while its behavior changes with its
body state (energy / integrity) — foraging when hungry, ignoring food
when full — then the "new-species large model" paradigm has crossed its
second step: not memorizing a policy, but installing homeostatic
judgment into language representations.

## Method (v1, falsifiable)

```
exp-a curriculum policy (teacher, 12x12, satiated=0 / far=1.00)
        | trajectory replay
        v
text observation -> action     SFT dataset (~20k examples)
        | QLoRA (4-bit, RTX 3050 4GB)
        v
Qwen2.5-0.5B-Instruct (student)
        | behavioral probes
        v
1. agreement with the teacher   — did it learn the policy
2. survival                     — can it stay alive
3. interoceptive modulation ★   — same view, different body states:
                                   do the actions change?
   (the dividing line between fitting a policy and embodying
   homeostatic judgment)
```

Honest boundary: SFT only proves "injectable", not "emergent". If the
modulation probe holds, follow-ups (DPO on homeostatic preference pairs,
RL with environmental reward directly into the policy) test deeper
injection.

## Structure

- `homeo_lm.textenv` — text-rendered homeostatic grid world (reuses the exp-a env)
- `homeo_lm.gen_data` — teacher trajectories -> SFT dataset
- `homeo_lm.sft` — QLoRA fine-tuning (snapshots + `--resume`)
- `homeo_lm.probe` — behavioral probes (agreement / survival / modulation)

## Status (v1, 2026-10-01 -> 10-02)

- [x] Environment: CUDA torch (RTX 3050 4GB), Qwen2.5-0.5B-Instruct cached locally
- [x] Teacher data: 40 episodes / **13,729 examples** (teacher death rate 0.325,
      mean episode length 343.2)
- [x] LoRA SFT: **done** (1 epoch, batch 2 x accum 16; the bitsandbytes DLL is
      blocked by Smart App Control, WinError 4551 -> automatic bf16 LoRA
      fallback; loss 0.65 -> 0.19)
- [x] Three probes: **done** — see below (`runs/sft-lora/probe_report.json`)

## Results (v1) — negative, recorded honestly

| Probe | Result | Criterion | Verdict |
|---|---|---|---|
| agreement | **0.485** (chance ~0.2, illegal rate 0) | > 0.7 | ✗ |
| survival | **0.00** (mean 77.7 steps/episode, illegal rate 0) | well above random | ✗ |
| modulation_gap ★ | **0.00** (hungry/satiated both 0% toward food) | > 0.5 | ✗ |

**Diagnosis**: greedy decoding outputs `down` in 100% of 60 constructed
states (food 3 cells right, alternating hungry/satiated) — the policy
collapsed to a single action. Yet the teacher's action distribution in the
dataset is rest 50% / right 15% / left 14% / up 11% / down 10%: the model
collapsed to a *minority* class, so 1 epoch achieved **format learning**
(emit a legal action word) but not **policy binding** (observation ->
action).

**Signals still worth noting**: illegal-output rate 0 (the format was
learned); agreement 0.485 is well above chance 0.2 (some policy signal
exists, but not enough to survive).

## v2 Plan

1. Scale data to 200 episodes (~70k examples); train 2-3 epochs
2. Per-action stratified balancing (rest is 50% of the data — class
   imbalance must be handled, e.g. downsampling rest)
3. Monitor the per-action prediction distribution during training;
   detect collapse and stop early
4. Wider LoRA coverage (add MLP layers) or larger r
5. Read-out logic unchanged: if v2 reaches agreement + survival while
   modulation_gap stays 0, that is precisely the predicted watershed
   between "fitting a policy" and "embodying homeostatic judgment" —
   escalate to DPO (homeostatic preference pairs)

## v2 Results (2026-10-10) — also negative, but the ceiling is exposed

| Probe | Result | Criterion | Verdict |
|---|---|---|---|
| agreement | **0.585** (illegal rate 0) | > 0.7 | △ progress, still short |
| survival | **0.00** (mean 110.6 steps/episode) | well above random | ✗ |
| modulation_gap ★ | **-1.00** (hungry 0% / satiated 100% toward food) | > 0.5 | ✗ **inverted** |

**v2 recipe**: 200 episodes / 70,300 SFT examples + per-action stratified
balancing (rest cap 11867) + 2 epochs + per-step distribution monitor
(checked every 200 steps; would alarm at top > 0.6; observed top stayed
at 0.31-0.34 throughout — **no collapse** — the monitor itself works).

**Key diagnosis**: v2 crossed two thresholds that v1 failed at —
- v1 collapsed to a single action (`down` × 100%); v2 stays distributed (top 0.33)
- v1 modulation gap = 0 (no modulation); v2 modulation gap = -1.0
  (**modulation is now present, but pointing the wrong way**)

This is the SFT ceiling: **pure behavior cloning can learn the action
distribution and a weak "interoception -> action" association, but it
cannot learn the causal direction**. The 0.5B model over-memorised the
minority "hungry + left" combinations in 70k examples and locked in a
reversed mapping.

## Conclusion and Next Step

**SFT has reached its ceiling** (v1 collapse -> v2 inverted modulation).
To go further, the gradient must come from **homeostatic consequences**
(alive / dead) rather than from behavior cloning:

- **DPO**: construct homeostatic preference pairs ("when hungry, go to
  food" chosen vs. "when hungry, walk away" rejected) so the model
  learns the interoception -> correct-action causal direction directly.
- **PPO / RL**: environment reward (Δenergy, survival steps) goes
  straight into the policy, bypassing behavior cloning.
- **Both routes need the student to be able to step the env** — already
  possible today: `TextEnv` runs at 0.5B.

## v3: DPO (built and trained 2026-10-15)

DPO is enabled after v2. The preference pairs are **not teacher labels**
but same-state exhaustive contrasts ranked by the environment's true
homeostatic consequences.

### Recipe (three decisions confirmed with the user)

| Decision | Choice |
|---|---|
| Pair source | same-state exhaustive contrast: try all 5 actions one step ahead at each state, rank by homeostatic consequence |
| Starting point | the v2 adapter serves as both π_θ and π_ref (`disable_adapter` reuses one base model; 1.2GB VRAM) |
| β | 0.1 (TRL default scale) |

Implementation: `homeo_lm.gen_dpo_data` / `homeo_lm.dpo` (hand-written DPO
loss, no new dependencies; assistant-token-only logprobs; reuses the SFT
collapse monitor).

### Three label-level pitfalls (each found and fixed in the smoke test)

A naive one-step contrast yields **wrong labels** in a homeostatic
environment — this is v3's most important lesson:

1. **Sparse-reward trap**: with food 3 cells away, walking toward it costs
   exactly the same -0.015 as walking anywhere, and `rest` (-0.005) always
   wins -> naive pairs would train an "always rest" collapse.
   **Fix**: potential-based shaping (PBRS) — `shaped = r + w·[φ(s') - φ(s)]`,
   with `φ = -dist(food)` gated by hunger and `+dist(hazard)` gated by
   injury. PBRS provably preserves the optimal policy, and the
   interoceptive modulation is built directly into the gate.

2. **Eat-respawn trap**: food respawns elsewhere after being eaten, so
   `φ(s')` computed from the respawned food yields a large negative
   shaping that drowns the +0.33 true eating gain.
   **Fix**: eat steps skip shaping (the raw reward is already a strong
   signal).

3. **View-discontinuity trap**: baking visibility straight into `φ` lets
   one outward step push food out of view, snapping φ back to 0 for a
   **positive** shaping.
   **Fix**: gates are decided once at the sampled state; all 5
   counterfactuals within a state use continuous omniscient distances.

After the fixes every label verifies correct: hungry + food 3 cells right
-> `right`; satiated -> `rest` (the antidote to v2's reversal); injured +
hazard-adjacent -> rejects stepping into the hazard; hungry + food-adjacent
-> eats.

### Training results

4,002 pairs (half teacher / half random rollouts) × 2 epochs = 500
optimizer steps:

| Metric | Smoke (200 pairs) | Full (500 steps) |
|---|---|---|
| loss | 0.52 | **0.12** |
| margin (how much the policy prefers chosen) | +0.44 | **+4.23** |
| acc (fraction with margin > 0) | 0.82 | **0.97** |
| collapse monitor top | 0.33 | **0.41** (healthy) |

The margin grew from +0.44 to +4.23: the policy now strongly prefers the
homeostatically better action — the DPO loop itself works.

### Probes: **pending** (blocked by system memory)

The `runs/dpo-v3` adapter is saved, but the three probes have not run yet.
On 2026-10-21 four consecutive attempts crashed inside `safe_open` with
`OSError 1455 (page file too small)` — the system commit charge was
38.8 / 39.9 GB (1.1GB left), and even importing numpy/torch failed.

**Unblock with any one of**:
1. Enlarge the page file (currently only 8.4GB; needs admin + reboot)
2. Close high-commit apps (pyrefly 1.6GB / ChatGPT 1.0GB / WorkBuddy 1.0GB)
3. Reboot the machine and run the probe first thing

Run with: `py -3.12 -m homeo_lm.probe --ckpt runs/dpo-v3 --teacher ../exp-a-homeostatic-rl/runs/curriculum/checkpoint.pt`

**What to read**: whether `modulation_gap` flips from v2's **-1.0 to
positive** (> 0.5 passes). That single number is v3's verdict.

## RSI Phase 0: fast measurement + controller skeleton (2026-10-25)

Full design in [`docs/RSI-PLAN.md`](../../docs/RSI-PLAN.md). What is
built so far:

### Probe speed-up (RSI's premise: measurement must be fast)

Every action used to go through `model.generate(max_new_tokens=4)` —
**4 forward passes per decision, one sample at a time**. Now it is
**batched first-token classification**: the action is one word and the
5 action words have distinct first tokens (asserted by
`action_token_ids`), so "greedy-decode an action" == "argmax over the 5
action first tokens" — ONE forward pass per batch. The survival probe
advances all 20 episodes in parallel.

- new `homeo_lm.acthead` (action head shared by probes and PPO):
  `act_batch`, `greedy_with_legality`, `batch_logprobs`
- **the illegal-rate signal survives**: restricting the decision to 5
  tokens would make the model "always legal", so we also look at the
  FULL-VOCAB argmax — anything outside the 5 action tokens is illegal
- equivalence is testable: `--verify-batching N` compares the batched
  path against per-sample `generate` (< 0.99 prints
  `BATCHING MISMATCH` and the fast path must not be used)

### `homeo_lm.rsi` (controller skeleton: measure + record only)

1. **three-way seed split** train / select / holdout (frozen; no
   drilling on the same exam)
2. **probe lock**: the probe config hash goes into the ledger; if it
   changes, this round's adoptions are void
3. **significance**: the bootstrap 95% CI of Δ must be entirely > 0 —
   not "looks higher"
4. **evolution ledger** `runs/rsi_ledger.jsonl`: `(tag, parent, change,
   metrics, probe_hash, guardrails)` per line — traceable, rollback-able
5. **guardrails** (report-only in this version): illegal rate over
   threshold, modulation gap < 0 (the v2 failure mode)

Run: `py -3.12 -m homeo_lm.rsi --ckpt runs/dpo-v3 --tag baseline-A`

### L0 self-tuning (2026-10-25, code ready)

`homeo_lm.rsi_l0` + `rsi.py --cycle`: **dry-run by default** — it prints
what to try next and the matching training commands, and executes
nothing (a trial means real training plus the full probe suite).

| Part | Behaviour |
|---|---|
| `KNOBS` | five bounded knobs: `shape_w` / `gate_margin` / `teacher_mix` / `entropy_coef` / `curriculum` |
| `FORBIDDEN_KEYS` | homeostatic objective, env physics, probes, the gate itself — **rejected structurally, metrics never consulted** |
| `ucb_propose` | UCB1: untried arms first, then mean + c·sqrt(ln N / n) |
| `Budget` | trials per cycle capped; repeated failure -> circuit breaker, hand to a human |
| `oscillating` | zig-zag detection: a knob flipping back and forth means spinning in noise — stop |
| `decide` | three-way: guardrail / regression -> **rollback**; significant gain -> adopt; else reject |

Run: `py -3.12 -m homeo_lm.rsi --cycle --out runs/rsi_ledger.jsonl`

### L1 self-generated training signal (2026-10-25, code ready)

L0 tunes knobs; **L1 tunes the data itself**: the student walks the
environment and we run the exhaustive contrast at the states where it is
CURRENTLY wrong — pairs grow where the mistakes are, and the
distribution moves with the policy (that is what separates iterated DPO
from one-shot DPO).

| Part | What it does |
|---|---|
| `gen_dpo_data --policy <adapter> --policy-frac 0.34` | a third data source: the student's OWN on-policy rollouts; `PolicyDriver` drives with acthead **sampling** (not greedy — greedy collapses trajectories onto a few deterministic routes and narrows coverage) |
| `homeo_lm.selftrain` | one round = generate -> train -> several independent measurements -> decide -> record; only an **adopt** advances `current`, otherwise the parent is kept. Dry-run by default |
| `probe --seed-offset` | independent samples for the significance test (same offset = reproducible) |
| `ppo --probe-every N` | a light in-training modulation probe (the model is already in memory, so it is nearly free); trace goes to `probe_trace.json` |

**Key engineering decision**: every L1 step runs in a **subprocess**;
models are never stacked in one process. Generation loads the teacher,
training loads policy+reference, measurement loads the student — two
0.5B models in one process would blow up on this 4GB machine.

Run: `py -3.12 -m homeo_lm.selftrain --start runs/dpo-v3 --rounds 3`
(add `--run` to actually execute)

### L2 metacognition + gated change of the learning rule (2026-10-25, code ready)

`homeo_lm.metacog` + `rsi.py --level 2`. L2 changes **the learning rule
itself** (LoRA rank / lr / consolidation), so it REQUIRES hypotheses —
otherwise it is random search, and random search yields no attribution
and therefore no accumulation.

**Every hypothesis must carry three parts** (enforced by
`is_falsifiable()`; a missing prediction blocks any action): `claim` —
a falsifiable assertion; `prediction` — which number moves which way if
it is true; `knob` — what would refute it.

| Hypothesis | Evidence (>= 2 conditions at once) | Refuting knob |
|---|---|---|
| `capacity_deficit` | high train_acc · gap<=0 · curve not rising | `lora_rank` |
| `optimization_underfit` | low margin · high loss · gap<0.5 | `lr` |
| `distribution_mismatch` | high agreement · low survival · holdout much worse | `teacher_mix` |
| `shaping_misspecified` | gap~0 · low illegal rate · low survival | `shape_w` |
| `catastrophic_forgetting` | rose then fell · training loss still falling | `replay_frac` |
| `collapse` | top share >= 0.6 · entropy < 0.2 | `entropy_coef` |

- **a missing metric is UNKNOWN and never fires a rule** (no hunches
  allowed as evidence)
- **historical hit rates feed ranking** (`hit_rates` + UCB): hypothesis
  classes that the environment used to confirm get priority — that is
  what makes it "meta"
- **`l2_gate` is stricter than L0**: holdout verified · >= 2 seeds ·
  **single variable** (one knob at a time, or there is no attribution) ·
  L2-only knobs
- load-bearing principle unchanged: **the system proposes, the
  environment decides**

Run: `py -3.12 -m homeo_lm.rsi --level 2 --metrics-file m.json --holdout --seeds 3`

### P4: merging into Stage C — training and deployment as one (2026-10-25, code ready)

The first three levels are all **offline**; P4 attaches the loop to a
**running** policy. Unification brings a new risk: one bad update would
**replace the live policy**. So the load-bearing parts change:

| Rule | Mechanism |
|---|---|
| the live policy is never replaced in place | the candidate runs in shadow and is promoted only if the **canary** passes (`promote`) |
| any metric regressing blocks promotion | no trading one metric for another (all three monitored metrics checked) |
| holdout delta above the threshold | live up but holdout flat = memorising the current stream |
| long-run health | no degradation, no oscillation over >10 cycles; K cycles without a new best -> **plateau alarm** |

`lifelong.py`: `promote` / `drift` / `throttle` / `health` /
`LifelongLoop`. The throttle keeps improvement from hogging live compute
(drift may start a cycle early, but a time floor still applies).

### Meta-experiment: is RSI worth it (`metaexp.py`)?

> Not "are the metrics high?" but **at equal compute, does RSI beat just
> training longer?**

Five groups A-E (A control / B L0 / C L1 / D L0+L1 / E everything);
`equal_compute()` verifies that compute is **genuinely equal** across
groups (otherwise the whole round is void). Criterion:
**superadditivity** — `D > A` and `D > B` and `D > C`.

The verdict carries an **evidence floor**, `MIN_MARGIN = 0.02`: D beating
A by 0.01 is not evidence (written and auditable, not a knob to fiddle
with). `D ≈ A` is honestly reported as `no_evidence`: **a negative result
must be recorded, not hidden**.

Run: `py -3.12 -m homeo_lm.metaexp --groups ABCDE`

## The real-run pipeline: `scripts/` (2026-11-01)

All four RSI levels are code-complete but **entirely dry-run**. Running
them for real has a mandatory order, and the scripts enforce it (any
step failing stops the chain — see `scripts/run_all.ps1`):

| Step | Script | Why this order |
|---|---|---|
| 1 | `01_verify_batching.ps1` | prove the batched fast path == per-sample `generate`. **Fail this and every later number is void** |
| 2 | `02_probe_v3.ps1` | full measurement of v3: has `modulation_gap` flipped from v2's -1.0 to positive (> 0.5 to pass)? |
| 3 | `03_baseline_A.ps1` | record the measurement in the ledger as **the zero point of control group A** — without it, "improved by how much" is empty |

One shot: `powershell -ExecutionPolicy Bypass -File scripts\run_all.ps1`
Step by step (when memory is tight): run the script directly; all accept
`-Ckpt runs\xxx`.

## v4 route: PPO (implemented, not yet trained)

`homeo_lm.ppo` — the environment reward goes **straight into the
policy**, bypassing behavior cloning entirely:

    rollout: the student plays TextEnv itself -> homeostatic reward
    update : PPO-clip on the LoRA; the token probability of
             high-reward action words rises

| Design | Choice | Rationale |
|---|---|---|
| Action space | FIRST TOKEN of the 5 action words; logits renormalised over those 5 | RL over a discrete action set; entropy and sampling stay well-behaved |
| Advantage | batch-normalised returns (no critic) | 4GB VRAM leaves no room for a value net; higher variance, but a single model trains |
| Reward shaping | reuses the DPO PBRS (same gates, eat steps skipped) | the homeostatic reward is too sparse to learn from raw |
| KL penalty | none (would need a second reference model) | stability comes from the entropy bonus and a small lr |

Training logs: return, loss, entropy, clip fraction, **survival rate /
mean episode length**, per-action distribution (alarm at top > 0.9).
Supports `--resume` and periodic snapshots, same convention as SFT/DPO.

Run: `py -3.12 -m homeo_lm.ppo --ckpt runs/dpo-v3 --out runs/ppo-v4`

## Extended read-out: `modulation-sweep`

A single gap is one difference; `--only modulation-sweep` sweeps an
energy gradient (0.2 -> 1.0) and yields a **dose-response curve** that
separates three failure modes:

| Curve shape | Verdict |
|---|---|
| flat | no modulation |
| monotonically decreasing | inverted modulation (exactly v2) |
| **monotonically increasing** | **correct modulation — the hungrier, the more it heads for food** |

Run: `py -3.12 -m homeo_lm.probe --ckpt <adapter> --only modulation-sweep`

## Usage

```powershell
py -3.12 -m pip install -e ../exp-a-homeostatic-rl   # teacher env
py -3.12 -m pip install -e ".[dev]"
py -3.12 -m homeo_lm.gen_data --ckpt ../exp-a-homeostatic-rl/runs/curriculum/checkpoint.pt --episodes 200
py -3.12 -m homeo_lm.sft --data data/sft.jsonl
py -3.12 -m homeo_lm.probe --ckpt runs/sft-lora --teacher ../exp-a-homeostatic-rl/runs/curriculum/checkpoint.pt
```

Model download (China mirror): `$env:HF_ENDPOINT="https://hf-mirror.com"`

## Read-Out Criteria

1. `agreement` > 0.7 — the action distribution matches the teacher
   (the policy was learned)
2. `survival` well above the random baseline — it stays alive
3. **`modulation_gap` > 0.5** — action distributions differ significantly
   between low-energy and satiated states (homeostatic judgment was
   installed into the language model, not just a vision->action mapping)
