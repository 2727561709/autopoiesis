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
