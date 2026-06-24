# HSE-Retro

Code for HSE-Retro: LLM-guided retrosynthesis planning with structured
experience and template-backed route validation.

## Setups

```bash
pip install -r requirements.txt
```

Set the LLM API key before running experiments:

```bash
export QIAN_DUO_DUO_API_KEY=your_api_key
```

On Windows PowerShell:

```powershell
$env:QIAN_DUO_DUO_API_KEY = "your_api_key"
```

## Data

Datasets, template databases, model checkpoints, and generated outputs are not
included in this repository. Place local resources under `data/` and `models/`
following the paths in `config/default_config.py`.

## Experiments

Run USPTO-190 style experiments:

```bash
python run_aot_parallel.py --targets data/test_sets/USPTO-190.smi --workers 2 --budget 100
```

Run Pistachio experiments:

```bash
python run_pistachio_parallel.py hard --workers 2 --budget 100
python run_pistachio_parallel.py reachable --workers 2 --budget 100
```

Consolidate results:

```bash
python scripts/consolidate_experiments.py
```
