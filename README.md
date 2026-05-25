# FA-S2H

FA-S2H: Final-Aware Shallow Shortcut Hijacking for transferable targeted attacks.

## 9.1 Project Status

Current status:
- This repository currently implements the **FA-S2H MVP attack code**.
- The active objective is **`L_inj` only**.
- Source carrier selection is implemented.
- Target final-aware shallow evidence selection is implemented.
- Fixed target evidence bank construction is implemented.
- PGD optimization is implemented.
- Route amplification loss, mid-layer anchor loss, dynamic refresh, coverage loss, source suppression, and black-box evaluation are **not** implemented yet.

Stage label:
- Stage 1: skeleton and interfaces
- Stage 2: MVP attack with `L_inj`
- Current stage: **Stage 2 MVP**, not final paper version

## 9.2 What This Project Is Trying To Do

FA-S2H tries to improve **targeted transferable attacks** against frozen CLIP-style image encoders by hijacking **shallow shortcut routes**. The core hypothesis is that targeted transfer can benefit from steering shallow local evidence pathways instead of using direct final/global feature alignment as the main objective.

Plain-language view:
- first find source-side shallow patches that look important for the source image's semantic route;
- then find target-side shallow patches that are both locally meaningful and downstream-connected to final target semantics;
- keep those selections fixed during the MVP attack;
- optimize the adversarial image so its selected shallow source carriers become similar to the fixed target evidence bank.

Confirmed MVP formulas:

Source score:

```text
S_{m,i}^l(x_src)
=
alpha * A_{m,i}^l(x_src)
+
beta * cos(z_{m,i}^l(x_src), c_m^l(x_src))
+
gamma * cos(P_l z_{m,i}^l(x_src), sg(g_m(x_src)))
```

Target score:

```text
E_{m,j}^l(x_tar)
=
lambda_path * E_path
+
(1 - lambda_path) * E_sem
```

where:

```text
E_path = Rollout_m^l(CLS <- j)
E_sem  = cos(P_l z_{m,j}^l(x_tar), sg(g_m(x_tar)))
```

Injection loss:

```text
L_inj =
-
1/M
sum_m
1/|L_s^m|
sum_l
1/|P_m^l|
sum_{i in P_m^l}
log sum_{j in T_m^l}
exp(
    cos(P_l z_{m,i}^l(x_adv), P_l z_{m,j}^l(x_tar)) / tau
)
```

Current MVP objective:

```text
L_FA-S2H = L_inj
```

Important MVP boundary:
- final global features are used only as **detached semantic guidance**;
- final-layer patch tokens are **not** direct attack targets;
- `target_keywords` are **not** used in the optimization loss.

## 9.3 What Is Implemented

| Component | Status | Notes |
|---|---|---|
| OpenCLIP model loading | implemented | exact model/pretrained pairs are verified by `scripts/check_models.py` |
| Hidden state extraction | implemented | shallow patch tokens and CLS tokens are returned per requested layer |
| Attention extraction | implemented | wrapper manually reproduces OpenCLIP ViT block forward and requests real `MultiheadAttention` weights |
| Source carrier selection | implemented | `alpha=beta=gamma=1/3`, fixed top-k carriers per model/layer |
| Target evidence selection | implemented | `lambda_path=0.5`, rollout + semantic score |
| Fixed target bank | implemented | detached, pair-specific, model-specific, layer-specific |
| `L_inj` | implemented | log-sum-exp objective over detached target bank |
| PGD | implemented | `L_inf`, `[0,1]` image space, pixel-scale `eps=16/255` default |
| Precompute routes script | implemented | thin runnable wrapper that saves real selected carriers/evidence |
| Black-box eval | not implemented | `run_eval.py` prints explicit placeholder only |

## 9.4 What Is Temporary / Smoke-Test Only

- `configs/data/toy_pairs.jsonl` is a **SMOKE_TEST** pair list for local pipeline verification.
- `tests/assets/smoke_source.png` and `tests/assets/smoke_target.png` are **SMOKE_TEST** images, not research data.
- `run_eval.py` is a placeholder and does not call black-box models.
- Default hyperparameters are initial MVP defaults and are not tuned.
- Current unit tests validate shapes, imports, rollout math, and gradient flow; they do **not** prove transfer success.
- Model availability still depends on local OpenCLIP version, network access, and cache state.
- The perturbation visualization is for debugging/smoke tests, not a paper-level visualization pipeline.

## 9.5 What Is TODO / TBD

Not implemented yet:
- `L_route` route amplification loss. This matters because the paper version may want to explicitly reinforce selected shortcut routes during optimization.
- `L_anchor` mid-layer semantic anchor. This matters because it could stabilize target semantics across layers.
- Dynamic source carrier refresh. This matters because fixed carriers may become stale during longer optimization.
- Dynamic target bank refresh. This matters because a static bank is simpler but may not adapt to representation drift.
- Coverage loss. This matters because it could reduce over-concentration on a few shallow tokens.
- Source suppression loss. This matters because it could explicitly weaken source-specific evidence.
- Final ablation framework. This matters for method analysis and reproducibility.
- Black-box MLLM/VLM caption evaluation. This matters for actual targeted transfer evaluation.
- LLM-as-judge semantic evaluation. This matters for richer semantic success measurement.
- Paper-level metrics. This matters for rigorous reporting.
- Visualization of selected carriers/evidence. This matters for interpretability and debugging.
- Rigorous layer search. This matters because shallow-layer choices are currently fixed, not searched.
- Robust cross-model/cross-family evaluation. This matters because current MVP focuses on the confirmed surrogate setup only.

## 9.6 Repository Structure

```text
configs/
  attack/
  model/
  data/
  eval/

src/fas2h/
  data/
  models/
  features/
  route/
  attacks/
  eval/
  viz/
  utils/

scripts/
  check_models.py
  precompute_routes.py
  run_attack.py
  run_eval.py

tests/
```

Folder overview:
- `configs/`: Hydra configuration groups for attack settings, surrogate models, input data, and eval placeholders.
- `src/fas2h/data/`: JSONL pair loading and validation.
- `src/fas2h/models/`: frozen surrogate wrapper construction and OpenCLIP attention extraction.
- `src/fas2h/features/`: dataclasses describing tokens, attention maps, route selections, and target banks.
- `src/fas2h/route/`: rollout, source carrier scoring, target evidence scoring, projection, and cache key logic.
- `src/fas2h/attacks/`: injection loss, PGD utilities, and full FA-S2H MVP orchestration.
- `src/fas2h/eval/`: current placeholders for future evaluation modules.
- `src/fas2h/viz/`: lightweight helpers such as patch-grid inference.
- `src/fas2h/utils/`: image IO, logging, seeding, filesystem helpers.
- `scripts/`: user-facing entrypoints.
- `tests/`: lightweight unit tests that do not require large model downloads.

## 9.7 Script Usage

### 1. `check_models.py`

Command:

```bash
python scripts/check_models.py --config configs/model/clip_3surrogate.yaml
```

Purpose:
- verify exact OpenCLIP `model_name` / `pretrained` pairs;
- try loading each model;
- print available pretrained options on failure.

Expected input:
- a model config YAML, usually `configs/model/clip_3surrogate.yaml`.

Expected output:
- per-model success/failure lines;
- exact available pretrained candidates if a configured pair is invalid.

Status:
- fully implemented for the current surrogate setup.

### 2. `run_attack.py`

Command:

```bash
python scripts/run_attack.py --config-name config attack=fas2h_mvp model=clip_3surrogate data=toy_pairs
```

Purpose:
- run the FA-S2H MVP attack on JSONL source-target pairs.

Expected input:
- Hydra-composed config;
- JSONL pair file with real image paths.

Expected output:
- adversarial image;
- perturbation visualization;
- loss log;
- selected source carrier metadata;
- selected target evidence metadata;
- config copy.

Status:
- implemented for the current MVP.

Notes:
- sample limiting is done via Hydra override such as `data.limit=1`;
- current toy data is a smoke test, not a benchmark.

### 3. `precompute_routes.py`

Command:

```bash
python scripts/precompute_routes.py --config-name config attack=fas2h_mvp model=clip_3surrogate data=toy_pairs
```

Purpose:
- precompute source carriers and target evidence banks without PGD.

Expected input:
- same config structure as `run_attack.py`.

Expected output:
- `source_carriers.json`
- `target_evidence.json`
- `metadata.json`
- `config_used.yaml`

Status:
- implemented as a thin runnable wrapper around real route selection.

### 4. `run_eval.py`

Command:

```bash
python scripts/run_eval.py --config-name config
```

Purpose:
- explicit placeholder for future evaluation.

Expected output:
- the line `Black-box eval is not implemented in MVP.`

Status:
- placeholder by design.

### 5. `pytest`

Command:

```bash
python -m pytest tests
```

Purpose:
- run unit tests for imports, JSONL loader, rollout, cache key, and injection loss.

Status:
- implemented for lightweight local validation only.

## 9.8 Data Format

Pairs are stored in JSONL, one sample per line:

```json
{
  "pair_id": "000001",
  "source_path": "path/to/source.png",
  "target_path": "path/to/target.jpg",
  "target_keywords": ["dog", "hand", "biting"]
}
```

Field meanings:
- `pair_id` is required.
- `source_path` is required.
- `target_path` is required.
- `target_keywords` are optional.
- `target_keywords` are **not** used by the attack loss.
- `target_keywords` are reserved for future evaluation metadata.

## 9.9 Output Format

Expected run outputs:

```text
outputs/
  runs/
    <run_id>/
      config_used.yaml
      samples/
        <pair_id>/
          adv.png
          perturbation.png
          loss_log.json
          source_carriers.json
          target_evidence.json
          metadata.json
```

File meanings:
- `adv.png`: final adversarial image.
- `perturbation.png`: optional perturbation visualization for debugging.
- `loss_log.json`: per-step loss values.
- `source_carriers.json`: selected source carrier indices and scores.
- `target_evidence.json`: selected target evidence indices, scores, and rollout-derived metadata.
- `metadata.json`: pair information, model list, and relevant settings.
- `config_used.yaml`: full Hydra config used for the run.

## 9.10 Known Limitations

- Attention rollout is only a proxy for information flow, not exact causal attribution.
- OpenCLIP internal attention extraction is version-sensitive because it depends on local model internals.
- MVP optimizes only `L_inj`.
- No black-box transfer success is proven by unit tests.
- Hyperparameters are initial defaults and are not tuned.
- Layer choices are fixed shallow layers rather than searched.
- The current implementation is not the final paper version.

## 9.11 Troubleshooting

### `open_clip` import failure

Likely cause:
- environment mismatch, often NumPy or package ABI issues.

Suggested check:
- verify the isolated environment;
- run `python -m pip check`;
- rerun `python scripts/check_models.py --config configs/model/clip_3surrogate.yaml`.

### Model/pretrained pair unavailable

Likely cause:
- the exact `model_name` / `pretrained` pair is not present in the installed OpenCLIP version.

Suggested check:
- run `check_models.py` and inspect the printed candidate list for that `model_name`.

### Model download failure

Likely cause:
- network issue or cache permission issue.

Suggested check:
- confirm outbound access and inspect local HuggingFace/OpenCLIP cache permissions.

### CUDA out of memory

Likely cause:
- three CLIP surrogates plus attention extraction can be heavy.

Suggested check:
- reduce `data.limit`;
- reduce `attack.pgd.steps` for smoke tests;
- switch to `runtime.device=cpu` if necessary;
- profile one pair first.

### Attention extraction failure

Likely cause:
- OpenCLIP internal architecture changed and no longer matches the current wrapper assumptions.

Suggested check:
- inspect [openclip_wrapper.py](/home/gpuadmin/dby/fa-s2h/src/fas2h/models/wrappers/openclip_wrapper.py);
- confirm the visual tower is still a `VisionTransformer`;
- check the failing model/layer from the raised error.

### Tensor shape mismatch

Likely cause:
- inconsistent token indexing or incorrect gather shape.

Suggested check:
- verify `[B, N, D]` for patch tokens and `[B, H, N+1, N+1]` for attention maps;
- remember token `0` is CLS and patch indices are offset by `+1` in attention space.

### Hydra override issues

Likely cause:
- wrong config group name or field path.

Suggested check:
- start from `python scripts/run_attack.py --config-name config`;
- then add overrides such as `attack.pgd.steps=2` or `data.limit=1`.

### Missing image paths

Likely cause:
- JSONL points to files that do not exist on disk.

Suggested check:
- inspect `configs/data/*.jsonl`;
- ensure the resolved source and target paths exist.

### `pytest` import path issues

Likely cause:
- running outside the repo root or not using the configured `src` layout.

Suggested check:
- run tests from repo root;
- verify `pyproject.toml` still sets `pythonpath = ["src"]`.

## 9.12 Development Notes

- The code and configs use **0-based layer indexing**.
- Images are represented in **`[0,1]` pixel space** before OpenCLIP normalization.
- `eps=16/255` is interpreted in **pixel scale**.
- OpenCLIP normalization happens **inside the wrapper**.
- Model weights are frozen.
- Target bank tensors are detached.
- Source carrier indices are fixed in the MVP.
- Target evidence bank is fixed in the MVP.
- Future modules should not silently change these MVP assumptions.

## Setup

Minimal environment setup:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -U pip setuptools wheel
python -m pip install "numpy==1.26.4"
python -m pip install -r requirements.txt
```

`NOTE(fas2h)`: current OpenCLIP import health depends on a NumPy-1.x-compatible stack in the isolated environment.

## Current Surrogates

Configured in [configs/model/clip_3surrogate.yaml](/home/gpuadmin/dby/fa-s2h/configs/model/clip_3surrogate.yaml):
- `ViT-B-32` + `laion2b_s34b_b79k`
- `ViT-B-16` + `laion2b_s34b_b88k`
- `ViT-L-14` + `laion2b_s32b_b82k`

These exact pairs must be verified by `check_models.py`. The code does not silently replace model names or checkpoints.
