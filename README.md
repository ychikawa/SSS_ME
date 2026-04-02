# Mitigating Token Homogenization in Token Merging via Source Selection Switching and Merge Embedding

The Vision Transformers (ViTs) in this repository incorporate Token Merging (ToMe) and the proposed Source Selection Switching (SSS) and Merge Embedding (ME). This repo contains files required to train, fine-tune, and benchmark DeiT/LV-ViT/MAE-style models with ToMe, DiffRate, and related variants.

---

## 1. Setup
1. Clone the repository.
2. Create the Conda environment:
   ```bash
   conda env create --file env.yml
   conda activate sss_me
   ```

---

## 2. Directory Overview
- `main.py`: entry point aggregating all train/eval logic.
- `engine.py`: per-epoch training, evaluation, calibration routines.
- `utils.py`: distributed helpers plus FLOPs/throughput utilities.
- `models/`: DeiT/LV-ViT/MAE implementations and `models/algo` housing ToMe/SSS/SSS+ME/etc.
- `data/`: ImageNet dataset loader and preprocessing code.
- `run/`: ready-to-use shell scripts for common training/eval jobs.
- `compression_rate.json`: DiffRate layer-wise token-keeping schedules.
- `output/` (generated): checkpoint directory; inference logs append to `result/result.txt`.

---

## 3. Dataset Preparation
- Expect ImageNet layout (`{DATA_PATH}/train`, `{DATA_PATH}/val`).
- Use `--sampling_ratio` / `--sampling_ratio_test` for subsampling.
- Update the `DATA_PATH` variable (line 3) inside every script under `run/` before executing.

---

## 4. Typical Usage

### 4.1 Common CLI Flags
- `--model`: choose from `deit_tiny`, `deit_small`, `deit_base`, `augreg_*`, `sam_base`, `lvvit_*`, `vit_*_mae`, etc.
- `--algo`: `default`, `diffrate`, `dtem`, `mctf`, `ppt`, `sss_me`, `sss`, `tofu`, `tome`.
- `--r`: tokens removed per layer for ToMe-style algorithms.
- `--benchmark`: store FLOPs and throughput inside `log_stats`.
- `--modular`: update only ME architecture parameters (used when training ME).
- `--resume --resume-file ./output/checkpoint.pth`: resume training or run evaluation.

### 4.2 Training Scripts
- **SSS+ME (distributed across 4 GPUs via torchrun)**

```./run/eval_sssme.sh
torchrun --nproc_per_node=4 -- main.py \
    --model $MODEL \
    --data_path $DATA_PATH \
    --batch-size 256 --epochs 1 \
    --output_dir ./output \
    --task_type [1,0,0] \
    --task_weight [1,0,0] \
    --algo $ALGO \
    --r $R \
    --modular \
    >> "${RESULT_DIR}/${MODEL}_R${R}.txt"
```

- **SSS (single-GPU benchmark example)**

```./run/eval_sss.sh
python main.py \
	--eval \
	--model $MODEL \
	--data_path  $DATA_PATH \
	--task_type [1,0,0] \
	--algo $ALGO \
	--r $R \
	--benchmark \
	>> "${RESULT_DIR}/${MODEL}_R${R}.txt"
```

Each script stores stdout under `RESULT_DIR` and writes the latest weights to `output/checkpoint.pth`.

### 4.3 Evaluation Scripts
- `eval_sss.sh`: evaluate ToMe+SSS (`--resume` reads `output/checkpoint.pth`).
- `eval_tome.sh`: evaluate plain ToMe with SSS.
- `eval_diffrate.sh`: DiffRate compression using `compression_rate.json` (`--tgt_flops` selects the schedule).

Adjust `MODEL`, `ALGO`, `R`, and `TGT_FLOPS` as needed.

---

## 5. Compression Schedules
- ToMe-style: specifying `--r` triggers `main.py`’s `r2schedule`, which produces layer-wise token counts.
- DiffRate: `--load_schedule --tgt_flops <value>` pulls `merge_kept_num` / `prune_kept_num` from `compression_rate.json`.

---

## 6. Benchmarking and Logging
- `--benchmark` compiles the model with `torch.compile(mode="reduce-overhead")`, times 200 runs, and appends FLOPs/throughput/Acc@1 to `result/result.txt` (see `main.py`).
- `run/*.sh` scripts tee stdout to `results_*/{MODEL}_R{R}.txt`.
- Checkpoints in `--output_dir` are saved as `checkpoint.pth` and can be reloaded with `--resume`.
