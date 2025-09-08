# corpus_toolkit
Python toolkit for textual analysis and visualization. Features include lexical diversity calculation, vocabulary growth prediction, entropy measures, and Zipf/Heaps law visualizations. Designed for computational linguistics research.

## Modules and Classes

### CorpusLoader
- **Purpose**: Loads a text corpus from NLTK or local files/directories. It supports optional downloading if the corpus is not available and provides caching for performance optimization.
- **Key Methods**:
  - `load_corpus()`: Loads and caches the corpus. If the corpus is not locally available, it downloads it if `allow_download` is True.
  - `is_corpus_available()`: Checks if the corpus is locally available or in NLTK.
  - `_download_corpus()`: (private) Downloads the corpus from NLTK if not available locally.
  - `_load_corpus_from_path(path)`: (private) Loads the corpus from a local file or directory.
  - `_load_corpus_from_nltk()`: (private) Loads the corpus from NLTK.
  - `_load_corpus()`: (private) Determines and executes the appropriate loading method based on the corpus source.
- **Initialization Parameters**:
  - `corpus_source`: Path to the corpus or NLTK corpus name.
  - `allow_download`: Boolean to allow downloading the corpus from NLTK (default: True).
  - `custom_download_dir`: Directory to store downloaded corpus (optional).
- **Attributes**:
  - `corpus_cache`: Stores the loaded corpus to avoid reloading.

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
pip install -r requirements.txt
```

### KenLM Installation (Optional)

First, ensure you have the necessary system packages installed. This can be done from a terminal or included in a script that runs shell commands.

```bash
sudo apt-get update
sudo apt-get install -y cmake build-essential libeigen3-dev libboost-all-dev
```
  
Then, you can use the following Python script to download and compile KenLM:

```python
#!/usr/bin/env python3
"""
Simple KenLM setup script for Linux environments.
Downloads, compiles, and installs KenLM with optimal settings.

Usage:
    sudo apt-get update
    sudo apt-get install -y cmake build-essential libeigen3-dev libboost-all-dev
    python3 setup_kenlm.py
"""

from pathlib import Path
import subprocess
import tempfile
import urllib.request
import tarfile
import multiprocessing
import shutil

def run_cmd(cmd, cwd=None):
    """Run command with error handling."""
    try:
        subprocess.run(cmd, check=True, cwd=cwd, shell=isinstance(cmd, str))
    except subprocess.CalledProcessError as e:
        print(f"Error: Command failed: {cmd}")
        exit(1)

def download_and_extract(url, extract_to):
    """Download and extract tarball."""
    print("Downloading KenLM...")
    with urllib.request.urlopen(url) as response:
        with tarfile.open(fileobj=response, mode="r|gz") as tar:
            # Security check - prevent path traversal
            def safe_extract(members):
                for member in members:
                    if member.path.startswith('/') or '..' in member.path:
                        continue
                    yield member
            tar.extractall(path=extract_to, members=safe_extract(tar))

def setup_kenlm():
    """Main setup function."""
    # Check dependencies
    deps = ['cmake', 'make', 'g++']
    for dep in deps:
        if shutil.which(dep) is None:
            print(f"Error: {dep} not found. Install dependencies first:")
            print("sudo apt-get update")
            print("sudo apt-get install -y cmake build-essential libeigen3-dev libboost-all-dev")
            exit(1)
    
    # Configuration
    url = "https://kheafield.com/code/kenlm.tar.gz"
    install_dir = Path.home() / ".local"
    jobs = multiprocessing.cpu_count()
    max_order = 12  # Good default for most use cases
    
    print(f"Setting up KenLM with max order {max_order} using {jobs} cores...")
    
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        
        # Download and extract
        download_and_extract(url, tmpdir)
        
        # Find KenLM directory
        kenlm_dir = next(tmpdir.rglob("*kenlm*"), None) or next(tmpdir.iterdir())
        build_dir = kenlm_dir / "build"
        build_dir.mkdir(exist_ok=True)
        
        print("Configuring...")
        run_cmd([
            "cmake", "..",
            f"-DKENLM_MAX_ORDER={max_order}",
            f"-DCMAKE_INSTALL_PREFIX={install_dir}",
            "-DCMAKE_BUILD_TYPE=Release",
            "-DCMAKE_CXX_FLAGS=-O3 -DNDEBUG -march=native",
            "-DFORCE_STATIC=OFF"
        ], cwd=build_dir)
        
        print(f"Building with {jobs} jobs...")
        run_cmd(["make", f"-j{jobs}"], cwd=build_dir)
        
        # Install
        print("Installing...")
        install_dir.mkdir(parents=True, exist_ok=True)
        run_cmd(["make", "install"], cwd=build_dir)
        
        # Add to PATH if needed
        bin_dir = install_dir / "bin"
        bashrc = Path.home() / ".bashrc"
        path_line = f'export PATH="$PATH:{bin_dir}"'
        
        if bashrc.exists():
            content = bashrc.read_text()
            if str(bin_dir) not in content:
                print("Adding to PATH...")
                with open(bashrc, "a") as f:
                    f.write(f"\n# KenLM\n{path_line}\n")
                print(f"Added {bin_dir} to PATH in ~/.bashrc")
                print("Run 'source ~/.bashrc' or restart your shell")

        print(f"\nKenLM installed successfully!")
        print(f"Install location: {install_dir}")
        print(f"Binaries available: {', '.join(b.name for b in bin_dir.glob('*') if b.is_file())}")
        print(f"Test with: {bin_dir}/lmplz --help")
        print("\nIf you get library errors, you may need:")
        print("   sudo apt-get install -y zlib1g-dev libbz2-dev liblzma-dev")

if __name__ == "__main__":
    setup_kenlm()
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
