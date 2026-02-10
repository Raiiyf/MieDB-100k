# Omnigen2-MIE

Our model is built based on [OmniGen2](https://github.com/VectorSpaceLab/OmniGen2), please refer to original repo for detailed introduction.

## Training

1. Data Preparation
Download `dataTrain` and put as `/path/to/MieDB-100k/dataTrain/`, then run following command to generate training data jsonl file.
```bash
python Miedb_data_generator.py
```

2. Training Configuration
Training Configurations are listed in `options/ft_miedb.yml`, you can modify the hyperparameters to meet with your experiment setting.

3. Launch Training

```bash
bash scripts/train/ft.sh
```
The distributed training configurations are set in the `scripts/train/ft.sh`, and default is using 8 GPUs. Feel free to modify it according to your experiment environment.

## Inference

1. Checkpoint Conversion

After training finished, run following scripts to convert the checkpoint:
```bash
python convert_ckpt_to_hf_format.py \
  --config_path experiments/ft_miedb/ft_miedb.yml \
  --model_path experiments/ft_miedb/checkpoint-20000/pytorch_model_fsdp.bin \
  --save_path experiments/ft_miedb/checkpoint-20000/transformer
```

2. Run Inference

```bash
bash Miedb_test.sh
```

You can change the number of processes by altering the `NUM_GPU` parameter inside `Miedb_test.sh` (default is 4).

P.S. If you want to use original Omnigen2 model, just delete the `--transformer_path` line.