import math
import shutil
import string
import subprocess
import sys
import tempfile
from collections import Counter
from pathlib import Path

import kenlm
import matplotlib.pyplot as plt
import nltk
import numpy as np
import regex as reg
from nltk.corpus import PlaintextCorpusReader, stopwords
from nltk.tokenize import word_tokenize
from scipy import optimize
from scipy.optimize import curve_fit, differential_evolution, minimize

class CorpusLoader:
    """
    Load a corpus from NLTK or a local file/directory, optimized for performance.
    """
    def __init__(self, corpus_source, allow_download=True, custom_download_dir=None):
        # Source of the corpus (either a local path or an NLTK corpus name)
        self.corpus_source = corpus_source
        # Flag to allow automatic download from NLTK if the corpus isn't available locally
        self.allow_download = allow_download
        # Custom directory for downloading corpora (if needed)
        self.custom_download_dir = custom_download_dir
        # Cache for loaded corpus to avoid reloading
        self.corpus_cache = None

    def _download_corpus(self):
        """Download the corpus from NLTK."""
        # Append custom download directory to NLTK's data path if provided
        if self.custom_download_dir and self.custom_download_dir not in nltk.data.path:
            nltk.data.path.append(self.custom_download_dir)
        # Download corpus using NLTK's download utility
        nltk.download(self.corpus_source, download_dir=self.custom_download_dir, quiet=True)

    def _load_corpus_from_path(self, path):
        """Load the corpus from a local file or directory."""
        # Use PlaintextCorpusReader to read tokens from the local path
        corpus_reader = PlaintextCorpusReader(str(path), '.*')
        # Return all tokens from the corpus
        return [token for fileid in corpus_reader.fileids() for token in corpus_reader.words(fileid)]

    def _load_corpus_from_nltk(self):
        """Load the corpus from NLTK."""
        # Access corpus from NLTK using the corpus name
        corpus_reader = getattr(nltk.corpus, self.corpus_source)
        # Return all tokens from the corpus
        return [token for fileid in corpus_reader.fileids() for token in corpus_reader.words(fileid)]

    def _load_corpus(self):
        """Load the corpus into memory."""
        # Determine if the source is a local path or an NLTK corpus name
        path = Path(self.corpus_source)
        if path.is_file() or path.is_dir():
            return self._load_corpus_from_path(path)
        else:
            return self._load_corpus_from_nltk()

    def load_corpus(self):
        """Get the corpus, either from cache or by loading it."""
        # Return cached corpus if available
        if self.corpus_cache is not None:
            return self.corpus_cache

        # Download corpus if not locally available and downloading is allowed
        if not self.is_corpus_available() and self.allow_download:
            self._download_corpus()

        # Load corpus into cache and return
        self.corpus_cache = self._load_corpus()
        return self.corpus_cache

    def is_corpus_available(self):
        """Check if the corpus is available locally or through NLTK."""
        try:
            # Check if the corpus can be found by NLTK
            nltk.data.find(self.corpus_source)
            return True
        except LookupError:
            return False
        
class Tokenizer:
    """
    Tokenize text into individual words with configurable options.
    """
    def __init__(self, remove_stopwords=False, remove_punctuation=False, use_nltk_tokenizer=True, stopwords_language='english'):
        # Configuration options for tokenization
        self.remove_stopwords = remove_stopwords # 'True' to remove stopwords
        self.remove_punctuation = remove_punctuation
        self.use_nltk_tokenizer = use_nltk_tokenizer
        self.stopwords_language = stopwords_language # Language for stopwords removal
        self.custom_regex = None
        # Set to store unwanted tokens (stopwords, punctuation) for removal
        self._unwanted_tokens = set()
        self._initialized = False

    def _initialize(self):
        """Initialize the tokenizer by ensuring NLTK resources are available and loading unwanted tokens."""
        if not self._initialized:
            self._ensure_nltk_resources()
            self._load_unwanted_tokens()
            self._initialized = True

    def _ensure_nltk_resources(self):
        """Ensure that NLTK resources are available."""
        if self.remove_stopwords:
            # Download NLTK stopwords if they are not already available
            try:
                nltk.data.find(f'corpora/stopwords/{self.stopwords_language}')
            except LookupError:
                nltk.download('stopwords', quiet=True)
        if self.use_nltk_tokenizer:
            # Ensure NLTK punkt tokenizer is available
            try:
                nltk.data.find('tokenizers/punkt')
            except LookupError:
                nltk.download('punkt', quiet=True)

    def _load_unwanted_tokens(self):
        """Load stopwords and punctuation sets for efficient access."""
        if self.remove_stopwords:
            # Update the set with stopwords for the specified language
            self._unwanted_tokens.update(stopwords.words(self.stopwords_language))
        if self.remove_punctuation:
            # Update the set with string punctuation characters
            self._unwanted_tokens.update(string.punctuation)

    def set_custom_regex(self, pattern):
        """Set a custom regex pattern for tokenization."""
        # Compile regex pattern for custom tokenization
        try:
            self.custom_regex = reg.compile(pattern)
        except reg.error as e:
            raise ValueError(f"Invalid regex pattern: {e}")

    def _remove_unwanted_tokens(self, tokens):
        """Remove unwanted tokens (stopwords, punctuation) from a list of tokens."""
        # Filter out tokens present in the unwanted tokens set
        return [token for token in tokens if token not in self._unwanted_tokens and not token.startswith('``')]

    def tokenize(self, text, lowercase=False):
        """
        Tokenize text into individual words based on the selected method.
        
        Args:
            text (str or list): The input text to tokenize.
            lowercase (bool): Whether to convert the text to lowercase before tokenization.
        
        Returns:
            list: A list of tokenized words.
        """
        if isinstance(text, list):
            # If input is a list, join it into a single string
            text = ' '.join(text)

        if lowercase:
            # Convert text to lowercase if specified
            text = text.lower()

        self._initialize()  # Ensure resources are loaded and unwanted tokens are set

        # Perform tokenization based on the selected method
        if self.custom_regex:
            # Tokenization using the custom regex pattern
            tokens = self.custom_regex.findall(text)
        elif self.use_nltk_tokenizer:
            # Tokenization using NLTK's word tokenizer
            tokens = word_tokenize(text)
        else:
            # Basic whitespace tokenization
            tokens = text.split()

        # Remove unwanted tokens from the result
        return self._remove_unwanted_tokens(tokens)

class CorpusTools:
    """
    Provides basic corpus analysis tools like frequency distribution and querying capabilities.
    """

    def __init__(self, tokens, shuffle_tokens=False):
        """
        Initialize the CorpusTools object with a list of tokens.
        
        :param tokens: List of tokens (words) in the corpus.
        :param shuffle_tokens: Boolean indicating whether to shuffle the tokens. Useful for unbiased analysis.
        """
        # Validate that all elements in tokens are strings
        if not all(isinstance(token, str) for token in tokens):
            raise ValueError("All tokens must be strings.")

        # Shuffle the tokens if required
        if shuffle_tokens:
            np.random.shuffle(tokens)

        # Store tokens and calculate frequency distribution
        self.tokens = tokens
        self.frequency = Counter(tokens)
        self._total_token_count = sum(self.frequency.values())

        # Initialize token details with frequency and rank
        self.token_details = {token: {'frequency': freq, 'rank': rank}
                              for rank, (token, freq) in enumerate(self.frequency.most_common(), 1)}

    @property
    def total_token_count(self) -> int:
        """
        Return the total number of tokens in the corpus.
        """
        return self._total_token_count

    def find_median_token(self) -> dict:
        """
        Find the median token based on frequency. 
        The median token is the token in the middle of the sorted frequency distribution.

        :return: Dictionary with the median token and its frequency.
        """
        median_index = self._total_token_count / 2
        cumulative = 0
        # Iterate over tokens in order of decreasing frequency
        for token, freq in self.frequency.most_common():
            cumulative += freq
            # Return the token once the cumulative count crosses the median index
            if cumulative >= median_index:
                return {'token': token, 'frequency': freq}

    def mean_token_frequency(self) -> float:
        """
        Calculate the mean frequency of tokens in the corpus.

        :return: Float representing the mean token frequency.
        """
        return self.total_token_count / len(self.token_details)

    def query_by_token(self, token) -> dict:
        """
        Retrieve frequency and rank details for a specific token.

        :param token: Token to query.
        :return: Dictionary with token details (frequency and rank).
        :raises ValueError: If the token is not found in the corpus.
        """
        token = token.lower()
        details = self.token_details.get(token)
        if details:
            return {'token': token, **details}
        else:
            raise ValueError(f"Token '{token}' not found in the corpus.")

    def query_by_rank(self, rank) -> dict:
        """
        Retrieve token details for a specific rank in the frequency distribution.

        :param rank: Rank to query.
        :return: Dictionary with token details for the given rank.
        :raises ValueError: If the rank is out of range.
        """
        # Validate rank range
        if rank < 1 or rank > len(self.token_details):
            raise ValueError(f"Rank {rank} is out of range. Valid ranks are from 1 to {len(self.token_details)}.")
        
        # Find the token with the specified rank
        token = next((t for t, d in self.token_details.items() if d['rank'] == rank), None)
        if token:
            return {'token': token, 'rank': rank, **self.token_details[token]}
        else:
            raise ValueError(f"Token with rank {rank} is not found in the corpus.")

    def cumulative_frequency_analysis(self, lower_percent=0, upper_percent=100) -> list:
        """
        Analyze tokens within a specific cumulative frequency range. 
        Useful for understanding the distribution of common vs. rare tokens.

        :param lower_percent: Lower bound of the cumulative frequency range (in percentage).
        :param upper_percent: Upper bound of the cumulative frequency range (in percentage).
        :return: List of dictionaries with token details in the specified range.
        :raises ValueError: If the provided percentages are out of bounds.
        """
        # Validate percentage inputs
        if not 0 <= lower_percent <= 100 or not 0 <= upper_percent <= 100:
            raise ValueError("Percentages must be between 0 and 100.")

        # Calculate the numeric thresholds based on percentages
        lower_threshold = self._total_token_count * (lower_percent / 100)
        upper_threshold = self._total_token_count * (upper_percent / 100)

        # Extract tokens within the specified frequency range
        return [{'token': token, **details}
                for token, details in self.token_details.items()
                if lower_threshold <= details['frequency'] <= upper_threshold]

    def list_tokens_in_rank_range(self, start_rank, end_rank) -> list:
        """
        List tokens within a specific rank range. 
        Useful for examining the most/least frequent subsets of tokens.

        :param start_rank: Starting rank of the range.
        :param end_rank: Ending rank of the range.
        :return: List of dictionaries with token details within the specified rank range.
        :raises ValueError: If the rank range is out of valid bounds.
        """
        # Validate rank range inputs
        if not (1 <= start_rank <= end_rank <= len(self.token_details)):
            raise ValueError(f"Rank range is out of valid bounds. Valid ranks are from 1 to {len(self.token_details)}.")

        # Extract tokens within the specified rank range
        return [{'token': token, **details}
                for token, details in self.token_details.items()
                if start_rank <= details['rank'] <= end_rank]

    def x_legomena(self, x) -> set:
        """
        List tokens that occur exactly x times in the corpus.
        Useful for finding specific frequency occurrences like hapax legomena.

        :param x: Number of occurrences to filter tokens by.
        :return: Set of tokens occurring exactly x times.
        :raises ValueError: If x is not a positive integer.
        """
        if not isinstance(x, int) or x < 1:
            raise ValueError("x must be a positive integer.")
        return {token for token, details in self.token_details.items() if details['frequency'] == x}

    def vocabulary(self) -> set:
        """
        Return the set of distinct tokens in the corpus.
        """
        return set(self.frequency.keys())

class AdvancedTools(CorpusTools):
    """
    Advanced analysis on a corpus of text, extending CorpusTools.
    Implements advanced linguistic metrics and statistical law calculations.
    """

    def __init__(self, tokens):
        super().__init__(tokens)
        # Caches to store precomputed parameters for efficient reuse
        self._zipf_alpha = None  # For Zipf's Law
        self._zipf_c = None  # For Zipf's Law
        self._heaps_params = None  # For Heaps' Law
        self._zipf_mandelbrot_params = None  # For Zipf-Mandelbrot Law
        self.herdans_c_value = None  # For Herdan's C measure

    def yules_k(self) -> float:
        """
        Calculate Yule's K measure, indicating lexical diversity. Higher values suggest greater diversity.
        """
        # Using numpy for efficient array operations on token frequencies
        freqs = np.array([details['frequency'] for details in self.token_details.values()])
        N = np.sum(freqs)  # Total token count
        sum_fi_fi_minus_1 = np.sum(freqs * (freqs - 1))  # Sum of f_i * (f_i - 1) across all frequencies
        # Yule's K equation
        K = 10**4 * (sum_fi_fi_minus_1 / (N * (N - 1))) if N > 1 else 0
        return K

    def herdans_c(self) -> float:
        """
        Compute Herdan's C measure, reflecting vocabulary richness relative to corpus size.
        More comprehensive version of type-token ratio (TTR). Herdan's C is defined as the
        logarithm of the number of distinct tokens (V) divided by the logarithm of the total
        token count (N). It provides a normalized measure of vocabulary richness.
        Handles very large values of V and N by scaling them down to avoid precision issues.
        """
        if self.herdans_c_value is not None:
            return self.herdans_c_value  # Use cached value if available
        
        # Utilize properties from CorpusTools
        V = len(self.vocabulary())  # Distinct token count (Vocabulary size)
        N = self.total_token_count  # Total token count (Corpus size)

        # Check for edge cases to prevent division by zero or logarithm of zero
        if V == 0 or N == 0:
            raise ValueError("Vocabulary size (V) or total token count (N) cannot be zero.")

        # Handling very large values of V and N
        MAX_FLOAT = sys.float_info.max  # Maximum float value in the environment
        if V > MAX_FLOAT or N > MAX_FLOAT:
            # Apply scaling to reduce the values
            scaling_factor = max(V, N) / MAX_FLOAT
            V /= scaling_factor
            N /= scaling_factor

        # Calculating Herdan's C with error handling
        try:
            self.herdans_c_value = math.log(V) / math.log(N)
        except ValueError as e:
            # Handle potential math domain errors
            raise ValueError(f"Error in calculating Herdan's C: {e}")

        return self.herdans_c_value

    @staticmethod
    def heaps_law(N, k, beta):
        """Heap's Law function."""
        return k * N**beta

    def generate_corpus_sizes(self, corpus_size):
        """Generate a range of corpus sizes for sampling."""
        min_size = min(100, max(10, corpus_size // 1000))
        log_samples = np.logspace(np.log10(min_size), np.log10(corpus_size), num=60)
        critical_points = 10 ** np.arange(2, int(np.log10(corpus_size)) + 1)
        combined_samples = np.unique(np.concatenate([log_samples, critical_points, [corpus_size]]))
        return combined_samples.astype(int)

    def calculate_vocab_sizes(self, corpus_sizes):
        """Efficiently calculate vocabulary sizes for given corpus sizes."""
        vocab_sizes = []
        unique_words = set()
        for size in corpus_sizes:
            unique_words.update(self.tokens[len(unique_words):size])
            vocab_sizes.append(len(unique_words))
        return vocab_sizes

    def calculate_heaps_law(self):
        """
        Estimate parameters for Heaps' Law using weighted non-linear curve fitting.
        """
        if self._heaps_params is not None:
            return self._heaps_params

        corpus_size = self.total_token_count
        
        # Generate corpus sizes
        corpus_sizes = self.generate_corpus_sizes(corpus_size)

        # Calculate vocabulary sizes for each corpus size
        vocab_sizes = self.calculate_vocab_sizes(corpus_sizes)

        # Define weights for curve fitting
        weights = np.sqrt(corpus_sizes)

        try:
            # Estimate parameters using weighted non-linear curve fitting
            popt, _ = curve_fit(self.heaps_law, corpus_sizes, vocab_sizes, p0=[1, 0.5], sigma=1/weights)
            
            self._heaps_params = tuple(popt)
            return self._heaps_params
        except RuntimeError:
            return None

    def estimate_vocabulary_size(self, total_tokens):
        """
        Estimate the vocabulary size for a given number of tokens using Heaps' Law.
        """
        if self._heaps_params is None:
            self._heaps_params = self.calculate_heaps_law()

        k, beta = self._heaps_params
        estimated_vocab_size = self.heaps_law(total_tokens, k, beta)

        return int(round(estimated_vocab_size))
    
    def estimate_vocabulary_size(self, total_tokens) -> int:
        """
        Estimate the vocabulary size for a given number of tokens using Heaps' Law.
        Ensures that the output is an integer, as vocabulary size cannot be fractional.
        Use to evaluate Heaps' law fit by comparing with actual vocabulary size.
        """
        if self._heaps_params is None:
            self._heaps_params = self.calculate_heaps_law() # Calculate parameters if not already cached

        K, beta = self._heaps_params # Retrieve parameters
        estimated_vocab_size = K * (total_tokens ** beta) # Calculate vocabulary size

        # Round the estimated vocabulary size to the nearest integer
        return int(round(estimated_vocab_size))

    def calculate_zipf_alpha(self):
        """
        Calculate the alpha parameter for Zipf's Law using Maximum Likelihood Estimation
        with improved numerical stability.
        """
        if self._zipf_alpha is not None:
            return self._zipf_alpha

        frequencies = np.array([freq for _, freq in self.frequency.most_common()])
        n = len(frequencies)
        
        # Normalize frequencies to probabilities
        probabilities = frequencies / np.sum(frequencies)

        def log_likelihood(alpha):
            # Use log-sum-exp trick for numerical stability
            log_ranks = np.log(np.arange(1, n + 1))
            log_probs = -alpha * log_ranks
            max_log_prob = np.max(log_probs)
            return (np.log(np.sum(np.exp(log_probs - max_log_prob))) + max_log_prob 
                    + np.sum(probabilities * alpha * log_ranks))

        # Use Brent's method to find the alpha that minimizes the negative log-likelihood
        result = optimize.minimize_scalar(log_likelihood, bracket=(1, 3), method='brent')
        
        if result.success:
            self._zipf_alpha = result.x
        else:
            raise RuntimeError("Failed to estimate Zipf's alpha parameter")

        return self._zipf_alpha

    def calculate_zipf_mandelbrot(self, verbose=False):
        """
        Fit the Zipf-Mandelbrot distribution to the corpus and find parameters q and s.
        """
        if self._zipf_mandelbrot_params is not None:
            return self._zipf_mandelbrot_params
        
        frequencies = np.array([details['frequency'] for details in self.token_details.values()])
        ranks = np.array([details['rank'] for details in self.token_details.values()])

        # Normalize frequencies
        max_freq = frequencies.max()
        normalized_freqs = frequencies / max_freq

        def zipf_mandelbrot_vectorized(ranks, q, s):
            return 1 / np.power(ranks + q, s)

        def objective_function(params):
            q, s = params
            predicted = zipf_mandelbrot_vectorized(ranks, q, s)
            normalized_predicted = predicted / np.max(predicted)
            return np.sum((normalized_freqs - normalized_predicted) ** 2)

        # Use differential evolution for initial guess
        bounds = [(1.0, 10.0), (0.1, 3.0)]
        result = differential_evolution(objective_function, bounds)
        if result.success:
            initial_q, initial_s = result.x
        else:
            raise RuntimeError("Differential evolution failed to converge")

        # Refine the estimate using local optimization
        refined_result = minimize(objective_function, [initial_q, initial_s], method='L-BFGS-B', bounds=bounds, options={'disp': verbose})
        if refined_result.success:
            q, s = refined_result.x
            if verbose:
                print(f"Optimization successful. Fitted parameters: q = {q}, s = {s}")
            self._zipf_mandelbrot_params = q, s
        else:
            raise RuntimeError("Local optimization failed to converge")

        return self._zipf_mandelbrot_params

class EntropyCalculator(CorpusTools):
    def __init__(self, tokens, q_grams=6):
        # Preprocess tokens similarly to the original method
        cleaned_tokens = [' '.join(reg.sub(r'[^a-zA-Z]', '', token).lower()) for token in tokens if len(token) >= 2]
        super().__init__(cleaned_tokens)
        self.q_grams = q_grams

    def calculate_H0(self):
        """Calculate the zeroth-order entropy (H0)."""
        # Assuming alphabet consists of only letters, case ignored
        alphabet = set(''.join(self.tokens).replace(' ', ''))
        alphabet_size = len(alphabet)
        return math.log2(alphabet_size)

    def calculate_H1(self):
        """Calculate the first-order entropy (H1)."""
        letter_freq = Counter(''.join(self.tokens).replace(' ', ''))
        total_letters = sum(letter_freq.values())
        return -sum((freq / total_letters) * math.log2(freq / total_letters) for freq in letter_freq.values())

    def calculate_H2(self):
        """
        Calculate the Rényi entropy of order 2 (H2).
        This is also known as collision entropy.
        """
        # Join all tokens and remove spaces to consider character distribution
        text = ''.join(self.tokens).replace(' ', '')
        
        # Count character frequencies
        char_freq = Counter(text)
        total_chars = len(text)
        
        # Calculate probabilities
        probabilities = np.array([count / total_chars for count in char_freq.values()])
        
        # Calculate H2
        H2 = -np.log2(np.sum(probabilities**2))
        
        return H2

    def train_kenlm_model(self, text):
        """Train a KenLM model with the given text and return the model path, without immediate cleanup."""
        tempdir_path = Path(tempfile.mkdtemp())  # Create temporary directory
        text_file_path = tempdir_path / "corpus.txt"
        
        with text_file_path.open('w') as text_file:
            text_file.write('\n'.join(self.tokens))
        
        model_file_path = tempdir_path / "model.klm"
        arpa_file_path = tempdir_path / "model.arpa"
        
        # Run lmplz with output redirected to suppress terminal output
        subprocess.run(f"lmplz -o {self.q_grams} --discount_fallback < {text_file_path} > {arpa_file_path}", shell=True, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        
        # Run build_binary to convert ARPA file to binary model
        subprocess.run(f"build_binary {arpa_file_path} {model_file_path}", shell=True, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        
        # Return both model file path and temp directory to manage cleanup later
        return model_file_path, tempdir_path

    def calculate_H3_kenlm(self):
        model_path, tempdir_path = self.train_kenlm_model(' '.join(self.tokens))
        model = kenlm.Model(str(model_path))
        
        prepared_text = ' '.join(self.tokens)
        log_prob = model.score(prepared_text, bos=False, eos=False) / math.log(2)
        num_tokens = len(prepared_text.split()) - self.q_grams + 1
        H3 = -log_prob / num_tokens
        
        # Cleanup: remove the temporary directory after the model has been used
        shutil.rmtree(tempdir_path)
        
        return H3

    def calculate_redundancy(self, H3, H0):
        """Calculate redundancy based on H3 and H0."""
        return (1 - H3 / H0) * 100

    def cleanup_temp_directory(self):
        """Remove the temporary directory used for KenLM model training."""
        if hasattr(self, 'tempdir_path') and self.tempdir_path.exists():
            shutil.rmtree(self.tempdir_path)
            print("Temporary directory cleaned up.")

class CorpusPlots:
    def __init__(self, analyzer, corpus_name, plots_dir='plots'):
        self.analyzer = analyzer
        self.corpus_name = corpus_name
        self.plots_dir = Path(plots_dir)
        self.plots_dir.mkdir(exist_ok=True)

    def _setup_plot(self, title, xlabel, ylabel, figsize=(12, 8)):
        """Set up a new plot with common elements."""
        plt.figure(figsize=figsize)
        plt.title(f"{title} for {self.corpus_name} Corpus")
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.grid(True, which="both", ls="--", alpha=0.5)

    def _save_plot(self, filename):
        """Save the current plot and close the figure."""
        plt.tight_layout()
        plt.savefig(self.plots_dir / f'{filename}_{self.corpus_name}.png', dpi=300)
        plt.close()

    def plot_zipfs_law_fit(self):
        alpha = self.analyzer.calculate_zipf_alpha()
        if alpha is None:
            raise ValueError("Alpha calculation failed, cannot plot Zipf's Law fit.")

        ranks = np.arange(1, len(self.analyzer.frequency) + 1)
        frequencies = np.array([freq for _, freq in self.analyzer.frequency.most_common()])
        
        self._setup_plot("Zipf's Law Fit", "Rank", "Normalized Frequency")
        
        plt.loglog(ranks, frequencies / frequencies.max(), 'o', label='Actual Frequencies', markersize=3, alpha=0.5)
        plt.loglog(ranks, (1 / np.power(ranks, alpha)) / (1 / np.power(ranks[0], alpha)), 
                   label=f'Zipf\'s Law Fit (α={alpha:.3f})', color='red', linewidth=2)
        
        plt.legend()
        self._save_plot('zipfs_law_fit')

    def plot_zipf_mandelbrot_fit(self):
        params = self.analyzer.calculate_zipf_mandelbrot()
        if params is None:
            raise ValueError("Zipf-Mandelbrot parameter calculation failed.")
        q, s = params

        ranks = np.array([details['rank'] for details in self.analyzer.token_details.values()])
        frequencies = np.array([details['frequency'] for details in self.analyzer.token_details.values()])

        def zipf_mandelbrot(k, q, s):
            return 1 / np.power(k + q, s)

        self._setup_plot("Zipf-Mandelbrot Fit", "Rank", "Normalized Frequency")

        plt.loglog(ranks, frequencies / frequencies.max(), 'o', label='Actual Frequencies', 
                   markersize=3, alpha=0.5, color='navy')  # Changed to a darker blue
        plt.loglog(ranks, zipf_mandelbrot(ranks, q, s) / zipf_mandelbrot(ranks[0], q, s), 
                   label=f'Zipf-Mandelbrot Fit (q={q:.3f}, s={s:.3f})', color='red', linewidth=2)

        plt.legend()
        self._save_plot('zipf_mandelbrot_fit')

    def plot_heaps_law(self):
        params = self.analyzer.calculate_heaps_law()
        if params is None:
            raise ValueError("Heaps' Law parameters calculation failed.")
        K, beta = params

        # Generate corpus sizes for sampling
        corpus_size = len(self.analyzer.tokens)
        corpus_sizes = self.analyzer.generate_corpus_sizes(corpus_size)

        # Calculate vocabulary sizes for each corpus size
        vocab_sizes = self.analyzer.calculate_vocab_sizes(corpus_sizes)

        self._setup_plot("Heaps' Law Analysis", "Token Count", "Type Count")

        plt.plot(corpus_sizes, vocab_sizes, label='Empirical Data', color='blue', alpha=0.7)
        plt.plot(corpus_sizes, K * np.power(corpus_sizes, beta), '--', 
                 label=f"Heaps' Law Fit: K={K:.2f}, β={beta:.3f}", color='red', linewidth=2)

        plt.legend()
        self._save_plot('heaps_law')