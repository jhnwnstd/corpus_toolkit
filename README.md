# corpus_toolkit
Python toolkit for textual analysis and visualization. Features include lexical diversity calculation, vocabulary growth prediction, entropy measures, and Zipf/Heaps law visualizations. Designed for computational linguistics research.

## Modules and Classes

### CorpusLoader
- **Purpose**: Loads a text corpus from NLTK or local files/directories. It supports optional downloading if the corpus is not available.
- **Key Methods**:
  - `load_corpus()`: Loads and caches the corpus. If the corpus is not locally available, it downloads it if `allow_download` is True.
  - `is_corpus_available()`: Checks if the corpus is locally available or in NLTK.
- **Initialization Parameters**:
  - `corpus_source`: Path to the corpus or NLTK corpus name.
  - `allow_download`: Boolean to allow downloading the corpus from NLTK.
  - `custom_download_dir`: Directory to store downloaded corpus.

### Tokenizer
- **Purpose**: Tokenizes text with options to remove stopwords and punctuation.
- **Key Methods**:
  - `tokenize(text, lowercase)`: Tokenizes the input text. Converts to lowercase if `lowercase` is True.
  - `set_custom_regex(pattern)`: Sets a custom regex pattern for tokenization.
- **Initialization Parameters**:
  - `remove_stopwords`: Boolean to remove stopwords.
  - `remove_punctuation`: Boolean to remove punctuation.
  - `use_nltk_tokenizer`: Boolean to use NLTK's tokenizer.
  - `stopwords_language`: Language for stopwords.

### CorpusTools
- **Purpose**: Provides basic analysis tools for a corpus, including frequency distribution, token querying, and lexical diversity measures.
- **Key Methods**:
  - `find_median_token()`: Identifies the median token based on frequency.
  - `mean_token_frequency()`: Calculates the average frequency of tokens throughout the corpus.
  - `query_by_token(token)`: Retrieves detailed information (frequency and rank) for a specific token.
  - `query_by_rank(rank)`: Retrieves the token and its frequency for a specific rank in the frequency distribution.
  - `cumulative_frequency_analysis(lower_percent, upper_percent)`: Analyzes tokens within a specified cumulative frequency range.
  - `list_tokens_in_rank_range(start_rank, end_rank)`: Lists tokens within a specific range of ranks.
  - `x_legomena(x)`: Lists tokens that occur exactly `x` times in the corpus.
  - `vocabulary()`: Returns a set of all distinct tokens in the corpus.
- **Initialization Parameters**:
  - `tokens`: List of tokens to be analyzed.
  - `shuffle_tokens`: Boolean to shuffle tokens before analysis.

### AdvancedTools
- **Purpose**: Extends `CorpusTools` with advanced linguistic metrics and statistical law fittings.
- **Key Methods**:
  - `yules_k()`: Calculates Yule's K measure for lexical diversity.
  - `herdans_c()`: Calculates Herdan's C measure for vocabulary richness.
  - `calculate_heaps_law()`: Estimates parameters for Heaps' Law.
  - `estimate_vocabulary_size(total_tokens)`: Estimates vocabulary size using Heaps' Law.
  - `calculate_zipf_alpha()`: Calculates alpha for Zipf's Law.
  - `calculate_zipf_mandelbrot()`: Fits the Zipf-Mandelbrot distribution to the corpus.
- **Inheritance**: Inherits from `CorpusTools`.

### EntropyCalculator
- **Purpose**: Calculates various entropy measures for the letters in a text corpus. Designed around character-level entropy, it provides insights into the predictability and structure of the text at different levels of complexity.
- **Key Methods**:
  - `calculate_H0()`: Computes the zeroth-order entropy (maximum entropy) for the corpus.
    - Calculation: H0 = log2(alphabet size)
    - Interpretation: Assumes a uniform distribution of characters, representing the theoretical maximum entropy.
  - `calculate_H1()`: Calculates first-order entropy based on individual character frequencies.
    - Calculation: H1 = -Σ(p(i) * log2(p(i))), where p(i) is the probability of character i
    - Interpretation: Considers the predictability of characters based on their frequency in the text.
  - `calculate_H2()`: Calculates second-order (Rényi) entropy, also known as collision entropy.
    - Calculation: H2 = -log2(Σ(p(i)^2))
    - Interpretation: Considers the probability of encountering the same character twice when sampling randomly. Less sensitive to rare events compared to H1.
  - `calculate_H3_kenlm()`: Utilizes KenLM models to estimate higher-order entropy.
    - Calculation: Based on n-gram language models (where n is specified by q_grams)
    - Interpretation: Captures linguistic patterns and context, providing deeper insights into text structure and predictability.
  - `calculate_redundancy()`: Assesses the redundancy in the text.
    - Calculation: Redundancy = (1 - H3/H0) * 100%
    - Interpretation: Measures the proportion of the text that is predictable based on linguistic structure.
- **Entropy Progression**:
  - Typically, H0 > H1 > H2 > H3
  - Each successive measure captures more linguistic structure and context
  - Lower values indicate more predictability and structure in the text
- **Inheritance**: Inherits from `CorpusTools`, leveraging its functionalities for preprocessing and token management to facilitate entropy calculations.

### CorpusPlots
- **Purpose**: Creates and saves plots related to corpus analysis.
- **Key Methods**:
  - `plot_zipfs_law_fit()`: Plots the rank-frequency distribution using Zipf's Law.
  - `plot_heaps_law()`: Plots the relationship between unique words and total words (Heap's Law).
  - `plot_zipf_mandelbrot_fit()`: Plots the Zipf-Mandelbrot distribution fit.
- **Initialization Parameters**:
  - `analyzer`: Instance of `AdvancedTools` or `CorpusTools`.
  - `corpus_name`: Name of the corpus for labeling plots.
  - `plots_dir`: Directory to save plots.

## Requirements
- Python 3.x
- NLTK package
- NumPy package
- Matplotlib package
- SciPy package
- KenLM (optional, for q-gram entropy calculations)

## Installation
Install the required packages using pip:
```bash
pip install nltk numpy matplotlib scipy
```

### KenLM Installation (Optional)

First, ensure you have the necessary system packages installed. This can be done from a terminal or included in a script that runs shell commands.

```bash
sudo apt-get update
sudo apt-get install -y cmake build-essential libeigen3-dev libboost-all-dev
```
  
Then, you can use the following Python script to download and compile KenLM:

```python
from pathlib import Path
import subprocess
import os
import urllib.request

def system_command(command):
    """Execute a system command with subprocess."""
    subprocess.run(command, shell=True, check=True)

def download_file(url, local_filename=None):
    """Download a file from a URL using urllib."""
    if not local_filename:
        local_filename = url.split('/')[-1]
    with urllib.request.urlopen(url) as response, open(local_filename, 'wb') as out_file:
        out_file.write(response.read())
    return local_filename

def compile_kenlm(max_order=12):
    """Compile KenLM with the specified maximum order."""
    url = "https://kheafield.com/code/kenlm.tar.gz"
    kenlm_tar = download_file(url)

    # Extract KenLM archive
    system_command(f'tar -xvzf {kenlm_tar}')

    # Setup KenLM directory paths using pathlib
    kenlm_dir = Path('kenlm')
    build_dir = kenlm_dir / 'build'
    build_dir.mkdir(parents=True, exist_ok=True)

    # Compile KenLM
    os.chdir(build_dir)
    system_command(f'cmake .. -DKENLM_MAX_ORDER={max_order}')
    system_command('make -j 4')
    os.chdir('../../')

    # Clean up downloaded files
    Path(kenlm_tar).unlink()

if __name__ == "__main__":
    compile_kenlm(max_order=8)
```

## Example Use (See toolkit_brown_analysis.py for more expansive use demonstration)
```python
# Load a corpus
loader = CorpusLoader('nltk_corpus_name')
corpus = loader.load_corpus()

# Tokenize
tokenizer = Tokenizer(remove_stopwords=True, remove_punctuation=True)
tokens = tokenizer.tokenize(corpus)

# Basic Analysis
corpus_analyzer = CorpusTools(tokens)
median_token = corpus_analyzer.find_median_token()

# Advanced Analysis
advanced_analyzer = AdvancedTools(tokens)
zipf_params = advanced_analyzer.calculate_zipf_params()

# Visualization
plotter = CorpusPlots(advanced_analyzer, 'Corpus_Name')
plotter.plot_zipfs_law_fit()
```

## Detailed Functionalities

- **Corpus Loading**: Handles local directories and NLTK datasets. Supports conditional downloading and caching for performance optimization.
- **Tokenization**: Offers customizable tokenization, including NLTK's tokenizer, custom regex, and options to remove stopwords and punctuation. Handles text input as strings or lists.
- **Basic Analysis**: Provides frequency distribution, median token, mean token frequency, specific token queries, rank-based queries, cumulative frequency analysis, and hapax legomena count.
- **Advanced Analysis**: Implements Zipf's Law, Heaps' Law, and Zipf-Mandelbrot distribution, including parameter estimation and fitting. Provides methods for lexical diversity (Yule's K) and vocabulary richness (Herdan's C).
- **Entropy Calculation**: Supports character-level entropy calculation, including first-order entropy and higher-order entropy using KenLM models. Also provides redundancy estimation.
- **Visualization**: Supports plotting for visual representation of Zipf's Law, Heaps' Law, and the Zipf-Mandelbrot distribution, enhancing the understanding of corpus characteristics. Plots are saved to a specified directory.
