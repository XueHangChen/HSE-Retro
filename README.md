# HSE-Retro

Code for HSE-Retro.

## Setups

```bash
pip install -r requirements.txt
```

Configure your own LLM API endpoint before running experiments.

## Data

Datasets, template databases, model checkpoints, and generated outputs are not included.
Place local resources under `data/` and `models/`.

## Experiments

USPTO-190:

```bash
python run_aot_parallel.py --targets data/test_sets/USPTO-190.smi
```

Pistachio Hard:

```bash
python run_pistachio_parallel.py hard
```

Pistachio Reachable:

```bash
python run_pistachio_parallel.py reachable
```
