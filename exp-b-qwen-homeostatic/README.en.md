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
- `homeo_lm.sft` — QLoRA fine-tuning
- `homeo_lm.probe` — behavioral probes (agreement / survival / modulation)

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
