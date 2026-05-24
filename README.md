# StateMoE

Experimental code for the State-aware MoE multimodal recommendation model.

## Structure

- `src/`: model, data loader, training, and evaluation code
- `run_clothing.sh`: robustness experiment script for the Clothing dataset

## Run

```bash
bash run_clothing.sh <gpu_id> <dataset_dir> "<condition_list>"
```

Example:

```bash
bash run_clothing.sh 1 dataset/clothing "mixed missing noisy"
```

Logs are saved under `src/log/clothing/`.

## Data

The `dataset_dir` should contain the default feature files:

- `image_feat.npy`
- `text_feat.npy`
