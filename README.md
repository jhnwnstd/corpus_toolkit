# corpus_toolkit

Python toolkit for corpus analysis and visualization. Lexical diversity, vocabulary growth, entropy measures, and Zipf/Heaps law fitting. Built for computational linguistics research.

## Quick Start

```bash
pip install -r requirements.txt
python toolkit_analysis.py
```

## Usage

```python
from toolkit_methods import *

# Load & tokenize
tokens = CorpusLoader('brown').load_corpus()
tokenized = Tokenizer(remove_punctuation=True).tokenize(tokens, lowercase=True)

# Analyze
tools = AdvancedTools(tokenized)
print(tools.calculate_zipf_alpha())
print(tools.calculate_heaps_law())

# Entropy
ec = EntropyCalculator(tokenized)
print(ec.calculate_H0(), ec.calculate_H1(), ec.calculate_H2())
print(ec.calculate_H3_kenlm())  # requires KenLM

# Plot
CorpusPlots(tools, 'brown').plot_zipfs_law_fit()
```

## Classes

### CorpusLoader
Loads corpora from NLTK or local files/directories with caching.

### Tokenizer
Tokenizes text with optional stopword/punctuation removal. Supports NLTK tokenizer, custom regex, or whitespace splitting.

### CorpusTools
Frequency distribution with dense ranking (equal freq = equal rank), rank queries, cumulative frequency analysis, hapax/dis legomena, median token.

### AdvancedTools
Extends `CorpusTools`:

| Method | Description |
|--------|-------------|
| `yules_k()` | Yule's K lexical diversity (10^4 * Simpson's D) |
| `herdans_c()` | Herdan's C vocabulary richness (log V / log N) |
| `calculate_heaps_law()` | Heaps' Law params (K, beta) via N^2-weighted curve fitting on shuffle-averaged V(n) |
| `estimate_vocabulary_size(n)` | Predict vocabulary size at n tokens |
| `calculate_zipf_alpha()` | Zipf exponent via discrete MLE with log-sum-exp stability |
| `calculate_zipf_mandelbrot()` | Zipf-Mandelbrot (q, s) via MLE with differential evolution fallback |
| `calculate_double_power_law()` | Piecewise power law (r_break, alpha1, alpha2, C) — fits core and tail regimes separately |
| `fit_quality()` | R², KS statistic, RMSE for all fitted models + Zipf-Heaps consistency check |
| `estimate_zipf_xmin()` | CSN method for power-law lower bound estimation |

All results are cached after first computation.

### EntropyCalculator
Character-level entropy (letters only, lowercased, spaces excluded from alphabet):

| Measure | Method | Notes |
|---------|--------|-------|
| H0 | `calculate_H0()` | log2(alphabet size) — theoretical maximum |
| H1 | `calculate_H1()` | Shannon entropy + Miller-Madow bias correction, capped at H0 |
| H2 | `calculate_H2()` | Rényi collision entropy, unbiased U-statistic estimator, capped at H1 |
| H3 | `calculate_H3_kenlm()` | KenLM 6-gram cross-entropy with 5-fold CV (log10→bits) |
| R | `calculate_redundancy(H3, H0)` | (1 - H3/H0) × 100% |

Guaranteed ordering: H0 ≥ H1 ≥ H2 ≥ H3. Results cached.

### CorpusPlots
Saves Zipf, Zipf-Mandelbrot, double power law, and Heaps' Law plots to `plots/`.

## Sample Output (Brown Corpus)

```
Zipf alpha:          1.02
Zipf-Mandelbrot:     q=1.22, s=1.07
Double Power Law:    α₁=0.97 (core), α₂=1.43 (tail), break=3711
Heaps' Law:          K=79.0, beta=0.47 (est vs actual: 0.06% error)
Yule's K:            97.37
Herdan's C:          0.78
H0: 4.70  H1: 4.17  H2: 3.93  H3: 2.27
Redundancy:          51.7%
```

## Dependencies

```
nltk, numpy, matplotlib, scipy, regex
```

KenLM is optional (required only for H3). See below to install.

## KenLM Installation (Optional)

```bash
# System dependencies
sudo apt-get install -y cmake build-essential libeigen3-dev libboost-all-dev

# Build from source
pip install https://github.com/kpu/kenlm/archive/master.zip
```

Ensure `lmplz` and `build_binary` are on your PATH.
