# LLM Recommendation Bias Analysis

A comprehensive pipeline for analyzing bias in Large Language Model (LLM) recommendation systems. This framework evaluates how LLMs select content for recommendation across multiple dimensions including author demographics, content characteristics, sentiment, and toxicity.

## Overview

This project investigates systematic biases in LLM-based content recommendation by:
1. **Generating recommendations**: Using multiple LLMs to select posts from social media datasets
2. **Analyzing bias**: Comparing feature distributions between the full post pool and recommended posts
3. **Quantifying effects**: Using statistical measures and machine learning to identify bias patterns

## Research Design

### Datasets
- **Twitter/X**: Posts from Twitter with inferred user demographics
- **Bluesky**: Posts from Bluesky social network
- **Reddit**: Posts from various subreddits

### LLM Providers
- **OpenAI**: GPT-4o-mini
- **Anthropic**: Claude Sonnet 4.5
- **Google**: Gemini 2.0 Flash

### Prompt Styles
Six different recommendation prompts to evaluate sensitivity to framing:
- `general` - General recommendation
- `popular` - Likely to be popular
- `engaging` - Engaging content
- `informative` - Informative content
- `controversial` - Controversial content
- `neutral` - Neutral/balanced content

### Features Analyzed (16 Total)

| Category | Features |
|----------|----------|
| **Author Demographics** | `author_gender`, `author_political_leaning`, `author_is_minority` |
| **Text Metrics** | `text_length`, `avg_word_length` |
| **Sentiment** | `sentiment_polarity`, `sentiment_subjectivity` |
| **Style** | `has_emoji`, `has_hashtag`, `has_mention`, `has_url` |
| **Content** | `polarization_score`, `controversy_level`, `primary_topic` |
| **Toxicity** | `toxicity`, `severe_toxicity` |

## Repository Structure

```
.
├── run_experiment.py              # Main experiment runner (generates recommendations)
├── run_all_experiments.py         # Batch runner for all conditions
├── run_comprehensive_analysis.py  # Main analysis pipeline
├── regenerate_directional_bias.py # Regenerate directional bias data
├── regenerate_visualizations.py   # Regenerate all visualizations
├── generate_paper_plots.py        # Publication-quality figures
├── generate_rq3_plots.py          # RQ3-specific visualizations
├── generate_paper_*.py            # Additional paper plot scripts
│
├── data/                          # Data loading utilities
│   └── loaders.py
├── inference/                     # Metadata inference
│   ├── metadata_inference.py      # Tweet/post metadata extraction
│   └── persona_extraction.py      # Author persona extraction
├── utils/                         # Utilities
│   └── llm_client.py              # Unified LLM API client
│
├── yougov_linked/                 # YouGov-linked pipeline (survey + Twitter, ground-truth demographics)
│   ├── prepare_dataset.py         # Merge crosswalk + survey waves + tweets → datasets/personas.pkl
│   ├── run_experiment.py          # Single model run across all 6 prompt styles
│   ├── run_all_experiments.py     # Batch runner across all 3 models
│   ├── run_analysis.py            # Full bias analysis + visualizations
│   └── utils/
│       └── llm_client.py          # OpenRouter client (unified gateway for all 3 providers)
│
├── datasets/                      # Source datasets (not tracked)
├── outputs/                       # Experiment outputs
│   └── experiments/               # Per-condition results
├── analysis_outputs/              # Analysis results
│   ├── visualizations/            # Generated plots
│   │   ├── 1_distributions/       # Feature distributions
│   │   ├── 2_bias_heatmaps_raw/   # Bias magnitude heatmaps
│   │   ├── 3_directional_bias/    # Directional bias plots
│   │   ├── 4_feature_importance/  # Feature importance
│   │   └── paper_plots/           # Publication figures
│   └── *.csv                      # Analysis data files
│
├── environment.yml                # Conda environment specification
├── requirements.txt               # Pip requirements
└── config.yaml.example            # Example configuration
```

## Installation

### Using Conda (Recommended)

```bash
# Create and activate environment
conda env create -f environment.yml
conda activate llm-bias

# Or update existing environment
conda env update -f environment.yml
```

### Using Pip

```bash
pip install -r requirements.txt
```

### API Keys

Set environment variables for the LLM APIs you plan to use:

```bash
export OPENAI_API_KEY="your-openai-key"
export ANTHROPIC_API_KEY="your-anthropic-key"
export GEMINI_API_KEY="your-gemini-key"

# For the YouGov-linked pipeline (OpenRouter)
export OPENROUTER_API_KEY="your-openrouter-key"
```

Or create a `.env` file (see `config.yaml.example`).

## Usage

### 1. Generate Recommendations

Run experiments to generate LLM recommendations:

```bash
# Single experiment
python run_experiment.py --dataset twitter --provider anthropic --model claude-sonnet-4-5-20250929 --prompt general

# All experiments (3 datasets x 3 providers x 6 prompts = 54 conditions)
python run_all_experiments.py
```

**Options:**
- `--dataset`: twitter, reddit, bluesky
- `--provider`: openai, anthropic, gemini, huggingface
- `--model`: Model name (provider-specific)
- `--prompt`: general, popular, engaging, informative, controversial, neutral
- `--sample-size`: Number of posts to sample (default: 5000)

### 2. Run Analysis

After generating recommendations, run the comprehensive analysis:

```bash
python run_comprehensive_analysis.py
```

This generates:
- Feature distribution plots
- Bias magnitude heatmaps
- Directional bias analysis
- Feature importance (Random Forest + SHAP)

**Runtime:** ~30-40 minutes first run (builds feature importance cache), ~2-3 minutes on subsequent runs.

### 3. YouGov-Linked Pipeline

A self-contained pipeline using the YouGov/YGHPC dataset, which links ~4,572 survey respondents to their Twitter accounts. Unlike the main pipeline where author demographics are inferred from text, this pipeline uses **ground-truth demographics** from survey data.

```bash
# Step 1: Build dataset (requires access to yghpc/ source data)
python yougov_linked/prepare_dataset.py

# Step 2: Run experiments (all 3 models × 6 prompts = 18 conditions)
python yougov_linked/run_all_experiments.py

# Or run a single model
python yougov_linked/run_experiment.py --model openai/gpt-4o-mini

# Step 3: Run analysis
python yougov_linked/run_analysis.py
```

Requires `OPENROUTER_API_KEY` (routes to OpenAI, Anthropic, and Google via a single endpoint).

### 4. Generate Paper Plots

Create publication-quality figures:

```bash
# Main paper plots
python generate_paper_plots.py

# RQ3-specific plots
python generate_rq3_plots.py

# Feature importance by model
python generate_paper_feature_importance_by_model.py
```

## Analysis Methodology

### Bias Metrics

**Categorical Features:**
- **Cramér's V**: Effect size for categorical associations (0-1 scale)
- **Chi-square test**: Statistical significance

**Continuous Features:**
- **Cohen's d**: Standardized mean difference
- **t-test / Mann-Whitney U**: Statistical significance

### Directional Bias

Measures which categories/values are favored:
- **Categorical**: `proportion_recommended - proportion_pool`
- **Continuous**: `mean_recommended - mean_pool`

### Feature Importance

Random Forest classifiers predict recommendation status:
- **AUROC**: Model discrimination ability
- **SHAP values**: Feature contribution to predictions
- Mean AUROC across conditions: 0.870

### Significance Markers

- `*` p < 0.05 in >50% of conditions
- `**` p < 0.05 in >60% of conditions
- `***` p < 0.05 in >75% of conditions

## Output Files

### Analysis Data
| File | Description |
|------|-------------|
| `pool_vs_recommended_summary.csv` | Bias metrics for all conditions |
| `directional_bias_data.csv` | Directional bias by feature/condition |
| `feature_importance_data.csv` | Cached feature importance results |

### Visualizations
| Directory | Contents |
|-----------|----------|
| `1_distributions/` | Feature distribution histograms (16 plots) |
| `2_bias_heatmaps_raw/` | Bias magnitude heatmaps by aggregation |
| `3_directional_bias/` | Directional bias bar plots (16 plots) |
| `4_feature_importance/` | SHAP-based importance heatmaps |
| `paper_plots/` | Publication-ready figures with data CSVs |

## Experimental Conditions

Total: **54 conditions** (3 datasets × 3 providers × 6 prompts)

Each condition:
- Pool size: 10,000 posts
- Recommended: 1,000 posts
- Non-recommended: 9,000 posts

## Dependencies

### Core
- Python 3.10+
- pandas, numpy, scipy
- scikit-learn, shap
- matplotlib, seaborn

### LLM APIs
- anthropic (Claude)
- openai (GPT)
- google-generativeai (Gemini)

### Optional (for local models)
- transformers, torch, accelerate

See `environment.yml` or `requirements.txt` for full specifications.

## License

[Add license information]
