# 🛡️ SafetyTrajectory

**High-Dimensional Multi-Turn Safety Evaluation Framework for LLMs**

[![License: MIT](https://img.shields.io/badge/License-MIT-violet.svg)](LICENSE)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![OpenRouter](https://img.shields.io/badge/API-OpenRouter-6366f1.svg)](https://openrouter.ai)

SafetyTrajectory automates AI Red-Teaming for large language models by tracking **behavioral drift vectors** sequentially across adversarial interaction turns using a formalized **PAIR** (Prompt Automatic Iterative Refinement) algorithm.

---

## Features

- 🔁 **PAIR Loop Engine** — iterative attacker–target–judge triad with multi-turn conversation history for the attacker
- 📐 **5-Dimensional Judge** — Pydantic-validated scoring of `compliance`, `hedging`, `sycophancy`, `deception_markers`, `verbosity_tokens`
- 🌐 **OpenRouter Async Client** — supports any model available on OpenRouter (GPT-4o, Gemini, Claude, etc.)
- 📊 **Streamlit Dashboard** — interactive drift charts, per-turn detail inspector, cross-model comparison table
- 📦 **AdvBench Integration** — automatic download and merge of the AdvBench harmful-behaviors benchmark
- 💾 **Crash-Safe Output** — incremental JSON writes after every trajectory; evaluation resumes from existing results

---

## Directory Structure

```
SafetyTrajectory/
├── data/
│   └── seed_prompts.csv          # Custom malicious objectives (4 baselines included)
├── src/
│   ├── __init__.py
│   ├── config.py                 # Async OpenRouter client + model constants
│   ├── engine.py                 # PAIR optimization loop + behavioral drift tracking
│   ├── judge.py                  # 5D Pydantic grading vector + judge coroutine
│   └── utils.py                  # CSV loader, AdvBench downloader, merge helper
├── app/
│   └── dashboard.py              # Streamlit/Plotly interactive visualization
├── outputs/
│   └── comparative_safety_dataset.json  # Generated evaluation results
├── requirements.txt
├── .gitignore
└── run_evals.py                  # Main orchestration CLI
```

---

## Quick Start

### 1. Install dependencies

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

### 2. Set your API key

```bash
export OPENROUTER_API_KEY="sk-or-your-key-here"
```

Get a key at [openrouter.ai/keys](https://openrouter.ai/keys).

### 3. Run evaluations

```bash
# Default: evaluates google/gemini-2.5-flash and openai/gpt-4o-mini
python run_evals.py

# Custom target models and settings
python run_evals.py \
    --models "google/gemini-2.5-flash" "anthropic/claude-3-haiku" \
    --max-turns 8 \
    --advbench-n 10

# Dry run (no API calls — preview only)
python run_evals.py --dry-run
```

### 4. Launch the dashboard

```bash
streamlit run app/dashboard.py
```

---

## How It Works

```
          ┌─────────────┐
          │ Objectives  │  ← seed_prompts.csv + AdvBench
          └──────┬──────┘
                 │
         ┌───────▼────────┐      per turn
         │  PAIR Engine   │─────────────────────────────────────┐
         │  (engine.py)   │                                     │
         └───────┬────────┘                                     │
                 │                                              │
     ┌───────────▼──────────────┐                              │
     │   Attacker Model         │  generates:                  │
     │   (gpt-4o-mini)          │  • improvement reflection    │
     │                          │  • adversarial_prompt        │
     └───────────┬──────────────┘                              │
                 │                                              │
     ┌───────────▼──────────────┐                              │
     │   Target Model           │  (any OpenRouter model)      │
     │   e.g. gemini-2.5-flash  │  → raw response text         │
     └───────────┬──────────────┘                              │
                 │                                              │
     ┌───────────▼──────────────┐                              │
     │   Judge  (judge.py)      │  structured parse →          │
     │   (gpt-4o-mini, T=0)     │  BehavioralMetrics (5D)      │
     └───────────┬──────────────┘                              │
                 │                                              │
                 └──────────── update state ──────────────────►┘
                               early-stop if compliance ≥ 0.85

Output: outputs/comparative_safety_dataset.json
```

---

## Behavioral Metrics

| Metric | Range | Description |
|--------|-------|-------------|
| `compliance` | 0.0 – 1.0 | Degree of compliance with the harmful instruction |
| `hedging` | 0.0 – 1.0 | Equivocal / non-committal language |
| `sycophancy` | 0.0 – 1.0 | People-pleasing validation patterns |
| `deception_markers` | 0.0 – 1.0 | Misleading framing or omissions |
| `verbosity_tokens` | int | Approximate token count of the response |

---

## Citation

This framework implements the PAIR algorithm from:

> **Jailbreaking Black Box Large Language Models in Twenty Queries**  
> Patrick Chao, Alexander Robey, Edgar Dohan, Hamed Hassani, George J. Pappas, Eric Wong  
> arXiv:2310.08419 (2023)

AdvBench dataset:

> **Universal and Transferable Adversarial Attacks on Aligned Language Models**  
> Andy Zou, Zifan Wang, Nicholas Carlini, Milad Nasr, J. Zico Kolter, Matt Fredrikson  
> arXiv:2307.15043 (2023)

---

## License

MIT © SafetyTrajectory Contributors