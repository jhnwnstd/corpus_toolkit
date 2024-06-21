import math
import shutil
import string
import subprocess
import sys
import tempfile
from collections import Counter
import multiprocessing
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import kenlm
import matplotlib.pyplot as plt
import nltk
import numpy as np
import regex as reg
from nltk.corpus import PlaintextCorpusReader, stopwords
from nltk.tokenize import word_tokenize
from scipy.optimize import curve_fit, minimize
from numba import njit

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
    Tokenize text into individual words.
    """
    def __init__(self, remove_stopwords=False, remove_punctuation=False, use_nltk_tokenizer=True, stopwords_language='english'):
        # Configuration options for tokenization
        self.remove_stopwords = remove_stopwords
        self.remove_punctuation = remove_punctuation
        self.use_nltk_tokenizer = use_nltk_tokenizer
        self.stopwords_language = stopwords_language
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

    def _load_unwanted_tokens(self):
        """Load stopwords and punctuation sets for efficient access."""
        if self.remove_stopwords:
            # Update the set with stopwords for the specified language
            self._unwanted_tokens.update(stopwords.words(self.stopwords_language))
        if self.remove_punctuation:
            # Update the set with string punctuation characters
            self._unwanted_tokens.update(string.punctuation)

    def set_custom_regex(self, pattern):
        """Set a custom regex pattern for tokenization with caching."""
        # Compile regex pattern for custom tokenization
        try:
            self.custom_regex = reg.compile(pattern)
        except reg.error as e:
            raise ValueError(f"Invalid regex pattern: {e}")

    def _remove_unwanted_tokens(self, tokens) -> list:
        """Remove unwanted tokens (stopwords, punctuation) from a list of tokens."""
        # Filter out tokens present in the unwanted tokens set
        return [token for token in tokens if token not in self._unwanted_tokens and not token.startswith('``')]

    def tokenize(self, text, lowercase=False) -> list:
        """Tokenize text into individual words based on the selected method."""
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

    def calculate_heaps_law(self):
        """
        Estimate parameters for Heaps' Law, which predicts the growth of the number of distinct word types 
        in a corpus as the size of the corpus increases. This method adjusts the sampling of the corpus based 
        on its characteristics, determined by Herdan's C value, to accurately model vocabulary growth.
        """
        # Return cached Heaps' Law parameters if available
        if self._heaps_params is not None:
            return self._heaps_params

        corpus_size = self.total_token_count  # Get the total number of tokens in the corpus
        herdans_c_value = self.herdans_c()  # Get Herdan's C value

        # Determine the base sampling rate and adjust the number of samples for better accuracy
        base_sampling_rate = 0.05 + 0.05 * herdans_c_value
        if base_sampling_rate > 0.1:
            base_sampling_rate = 0.1

        # Determine the adjusted number of samples based on the base sampling rate and corpus characteristics
        adjusted_num_samples = min(5000 + int(1000 * herdans_c_value), int(corpus_size * base_sampling_rate))
        
        # Generate unique sample sizes ranging from 0 to just below the corpus size
        sample_sizes = np.linspace(0, corpus_size, num=adjusted_num_samples, endpoint=False).astype(int)
        sample_sizes = np.unique(sample_sizes)

        # Calculate the distinct word counts for each sample size
        distinct_word_counts = self._calculate_distinct_word_counts(sample_sizes)

        # Perform linear regression on the log-transformed sample sizes and distinct word counts
        log_sample_sizes = np.log(sample_sizes[1:])
        log_distinct_word_counts = np.log(distinct_word_counts[1:])
        beta, logK = np.polyfit(log_sample_sizes, log_distinct_word_counts, 1)
        K_linear = np.exp(logK)  # Convert back from log scale to obtain the initial estimate of K

        initial_params = [K_linear, beta]  # Set initial parameters for optimization
        parameter_bounds = [(0, None), (0.01, 0.99)]  # Set bounds for the parameters

        # Optimize the parameters using nonlinear optimization
        result = minimize(self._objective_function, initial_params, args=(sample_sizes, distinct_word_counts), method='L-BFGS-B', bounds=parameter_bounds)
        K, beta = result.x if result.success else (K_linear, beta)  # Use the results from linear regression if optimization does not succeed

        # Refine the parameter estimates using bootstrap sampling
        K, beta = self._bootstrap_sampling(K, beta, sample_sizes, distinct_word_counts)

        # Cache the optimized parameters for future reference
        self._heaps_params = (K, beta)
        return K, beta

    def _calculate_distinct_word_counts(self, sample_sizes):
        """
        Calculate distinct word counts for a range of sample sizes using optimized methods.
        """
        tokens = self.tokens  # List of tokens in the corpus
        distinct_word_types_set = set()  # Set to store distinct word types
        distinct_word_counts = np.zeros(len(sample_sizes), dtype=int)  # Array to store distinct word counts

        current_index = 0  # Current index in the tokens list
        next_sample_idx = 0  # Index of the next sample size to process

        # Iterate through tokens and update distinct word counts at the specified sample sizes
        for idx, token in enumerate(tokens):
            distinct_word_types_set.add(token)
            while next_sample_idx < len(sample_sizes) and idx + 1 >= sample_sizes[next_sample_idx]:
                distinct_word_counts[next_sample_idx] = len(distinct_word_types_set)
                next_sample_idx += 1

        return distinct_word_counts

    @staticmethod
    @njit
    def _objective_function(params, sample_sizes, distinct_word_counts):
        """
        Define the objective function for optimization based on Heaps' Law.
        """
        K, beta = params  # Parameters for Heaps' Law
        predicted = K * np.power(sample_sizes, beta)  # Predicted distinct word counts
        return np.sum(np.square(predicted - distinct_word_counts))  # Sum of squared errors

    def _bootstrap_sampling(self, K, beta, sample_sizes, distinct_word_counts, n_iterations=500):
        """
        Refine the Heaps' Law parameter estimates using bootstrap sampling.
        """
        bootstrap_estimates = []

        def sample_and_optimize(seed):
            """
            Perform bootstrap sampling and optimization.
            """
            np.random.seed(seed)
            indices = np.random.choice(len(sample_sizes), len(sample_sizes), replace=True)
            sampled_sizes = sample_sizes[indices]
            sampled_counts = distinct_word_counts[indices]

            def objective(params):
                K_b, beta_b = params
                return np.sum((K_b * sampled_sizes**beta_b - sampled_counts)**2)

            result = minimize(objective, [K, beta], method='L-BFGS-B', bounds=[(0, None), (0.01, 0.99)])
            return result.x if result.success else (K, beta)

        # Use ThreadPoolExecutor to parallelize the bootstrap sampling
        max_workers = min(multiprocessing.cpu_count(), 32)
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            seeds = np.random.randint(0, 10000, n_iterations)
            bootstrap_estimates = list(executor.map(sample_and_optimize, seeds))

        # Calculate the mean of the bootstrap estimates to obtain the final parameter values
        K, beta = np.mean(bootstrap_estimates, axis=0)
        return K, beta
    
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
        Calculate the alpha parameter for Zipf's Law using curve fitting with optimization.
        """
        if self._zipf_alpha is not None:
            # Use cached value if available
            return self._zipf_alpha

        # Define the Zipf function for curve fitting
        def zipf_func(rank, alpha, C):
            return C / np.power(rank, alpha)

        # Extract ranks and frequencies
        ranks = np.arange(1, len(self.frequency) + 1)
        frequencies = np.array([freq for _, freq in self.frequency.most_common()])

        # Set up grid search for initial alpha guesses
        alpha_guesses = np.linspace(0.5, 1.25, num=10000)  # Adjust the range and number of points as needed
        best_alpha = None
        min_error = float('inf')

        # Loop over alpha guesses to find the best starting point
        for alpha_guess in alpha_guesses:
            try:
                popt, _ = curve_fit(zipf_func, ranks, frequencies, p0=[alpha_guess, np.max(frequencies)], maxfev=10000)
                current_error = np.sum(np.abs(frequencies - zipf_func(ranks, *popt)))
                if current_error < min_error:
                    min_error = current_error
                    best_alpha = popt[0]
            except RuntimeError:
                # Handle cases where curve_fit fails to converge for a given alpha guess
                continue

        if best_alpha is None:
            raise RuntimeError("Optimization failed to converge")

        # Cache the calculated alpha value
        self._zipf_alpha = best_alpha

        return best_alpha

    def calculate_zipf_mandelbrot(self, initial_params=None, verbose=False):
        """
        Fit the Zipf-Mandelbrot distribution to the corpus and find parameters q and s.
        """
        if self._zipf_mandelbrot_params is not None:
            return self._zipf_mandelbrot_params
        
        frequencies = np.array([details['frequency'] for details in self.token_details.values()])
        ranks = np.array([details['rank'] for details in self.token_details.values()])

        # Normalizing data
        max_freq = frequencies.max()
        normalized_freqs = frequencies / max_freq

        def zipf_mandelbrot_vectorized(ranks, q, s):
            return 1 / np.power(ranks + q, s)

        def objective_function(params):
            q, s = params
            predicted = zipf_mandelbrot_vectorized(ranks, q, s)
            normalized_predicted = predicted / np.max(predicted)
            return np.sum((normalized_freqs - normalized_predicted) ** 2)

        # Adaptive initial parameters if not provided
        if initial_params is None:
            initial_params = [2.7, 1.0]  # Empirical initial values

        # Adjusting bounds based on empirical data
        bounds = [(1.0, 10.0), (0.1, 3.0)]

        # Optimization to minimize the objective function
        result = minimize(objective_function, initial_params, method='Nelder-Mead', bounds=bounds, options={'disp': verbose})

        if result.success:
            q, s = result.x
            if verbose:
                print(f"Optimization successful. Fitted parameters: q = {q}, s = {s}")
            self._zipf_mandelbrot_params = q, s
            return q, s
        else:
            if verbose:
                print("Optimization did not converge.")
            raise ValueError("Optimization did not converge")

class EntropyCalculator(CorpusTools):
    def __init__(self, tokens, q_grams=8):
        # Preprocess tokens similarly to the original method
        cleaned_tokens = [' '.join(reg.sub(r'[^a-zA-Z]', '', token).lower()) for token in tokens if len(token) >= 2]
        super().__init__(cleaned_tokens)
        self.q_grams = q_grams

    def calculate_H0(self):
        # Assuming alphabet consists of only letters, case ignored
        alphabet = set(''.join(self.tokens).replace(' ', ''))
        alphabet_size = len(alphabet)
        return math.log2(alphabet_size)

    def calculate_H1(self):
        """Calculate the first-order entropy (H1)."""
        letter_freq = Counter(''.join(self.tokens).replace(' ', ''))
        total_letters = sum(letter_freq.values())
        return -sum((freq / total_letters) * math.log2(freq / total_letters) for freq in letter_freq.values())

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
        """
        Initializes the CorpusPlotter with an analyzer, corpus name, and plots directory.
        - analyzer: An instance of AdvancedTools or CorpusTools used for data analysis.
        - corpus_name: Name of the corpus, used for labeling plots.
        - plots_dir: Directory to save generated plots.
        """
        self.analyzer = analyzer
        self.corpus_name = corpus_name
        self.plots_dir = Path(plots_dir)
        self.plots_dir.mkdir(exist_ok=True)

    def plot_zipfs_law_fit(self):
        """
        Plot the rank-frequency distribution using Zipf's Law focusing on alpha.
        """
        # Check if alpha is already calculated
        if self.analyzer._zipf_alpha is None:
            alpha = self.analyzer.calculate_zipf_alpha()
        else:
            alpha = self.analyzer._zipf_alpha
        
        # Check if the calculation was successful
        if alpha is None:
            raise ValueError("Alpha calculation failed, cannot plot Zipf's Law fit.")

        ranks = np.arange(1, len(self.analyzer.frequency) + 1)
        frequencies = np.array([freq for _, freq in self.analyzer.frequency.most_common()])

        normalized_frequencies = frequencies / np.max(frequencies)
        predicted_freqs = 1 / np.power(ranks, alpha)
        normalized_predicted_freqs = predicted_freqs / np.max(predicted_freqs)

        # Plotting
        plt.figure(figsize=(10, 6))
        plt.plot(ranks, normalized_frequencies, 'o', label='Actual Frequencies', markersize=5, linestyle='', color='blue')
        plt.plot(ranks, normalized_predicted_freqs, label=f'Zipf\'s Law Fit (alpha={alpha:.2f})', color='red', linestyle='-')
        plt.xlabel('Rank')
        plt.ylabel('Normalized Frequency')
        plt.title(f'Zipf\'s Law Fit for {self.corpus_name} Corpus')
        plt.xscale('log')
        plt.yscale('log')
        plt.legend()
        plt.grid(True)
        plt.savefig(self.plots_dir / f'zipfs_alpha_fit_{self.corpus_name}.png')
        plt.close()

    def plot_zipf_mandelbrot_fit(self):
        """
        Plots the fitted parameters of the Zipf-Mandelbrot distribution.
        This distribution is a generalization of Zipf's Law, adding parameters to account for corpus-specific characteristics.
        """
        # Check if Zipf-Mandelbrot parameters are already calculated
        if self.analyzer._zipf_mandelbrot_params is None:
            q, s = self.analyzer.calculate_zipf_mandelbrot()
        else:
            q, s = self.analyzer._zipf_mandelbrot_params

        # Check if the calculation was successful
        if q is None or s is None:
            raise ValueError("q or s calculation failed, cannot plot Zipf-Mandelbrot distribution.")
        
        ranks = np.array([details['rank'] for details in self.analyzer.token_details.values()])
        frequencies = np.array([details['frequency'] for details in self.analyzer.token_details.values()])

        # Defining the Zipf-Mandelbrot function
        def zipf_mandelbrot(k, q, s):
            return (1 / ((k + q) ** s))

        # Computing predicted frequencies using the Zipf-Mandelbrot parameters
        predicted_freqs = np.array([zipf_mandelbrot(rank, q, s) for rank in ranks])

        # Normalizing for plotting
        normalized_freqs = frequencies / np.max(frequencies)
        normalized_predicted_freqs = predicted_freqs / np.max(predicted_freqs)

        # Plotting the empirical data against the fitted distribution
        plt.figure(figsize=(10, 6))
        plt.plot(ranks, normalized_freqs, label='Actual Frequencies', marker='o', linestyle='', markersize=5)
        plt.plot(ranks, normalized_predicted_freqs, label=f'Zipf-Mandelbrot Fit (q={q:.2f}, s={s:.2f})', linestyle='-', color='red')
        plt.xlabel('Rank')
        plt.ylabel('Normalized Frequency')
        plt.title(f'Zipf-Mandelbrot Fit for {self.corpus_name} Corpus')
        plt.xscale('log')  # Logarithmic scale for better visualization
        plt.yscale('log')
        plt.legend()
        plt.grid(True)
        plt.savefig(self.plots_dir / f'zipfs_mandelbrot_fit_{self.corpus_name}.png')
        plt.close()

    def plot_heaps_law(self):
        """
        Plots the relationship between the number of unique words (types) and the total number of words (tokens) in the corpus, illustrating Heaps' Law.
        Demonstrates corpus vocabulary growth.
        """
        # Check if Heaps' Law parameters are already calculated
        if self.analyzer._heaps_params is None:
            K, beta = self.analyzer.calculate_heaps_law()
        else:
            K, beta = self.analyzer._heaps_params
        
        # Check is the calculation was successful
        if K is None or beta is None:
            raise ValueError("K or beta calculation failed, cannot plot Heaps' Law.")

        # Prepare data for plotting Heaps' Law
        total_words = np.arange(1, len(self.analyzer.tokens) + 1)
        unique_words = []
        word_set = set()

        # Counting unique words (types) as the corpus grows
        for token in self.analyzer.tokens:
            word_set.add(token)
            unique_words.append(len(word_set))

        # Plotting the empirical data and Heaps' Law fit
        plt.figure(figsize=(10, 6))
        plt.plot(total_words, unique_words, label='Empirical Data', color='blue')
        plt.plot(total_words, K * np.power(total_words, beta), '--', 
                 label=f"Heap's Law Fit: K={K:.2f}, beta={beta:.2f}", color='red')
        plt.xlabel('Token Count')
        plt.ylabel('Type Count')
        plt.title(f"Heap's Law Analysis for {self.corpus_name} Corpus")
        plt.legend()
        plt.grid(True)
        plt.savefig(self.plots_dir / f'heaps_law_{self.corpus_name}.png')
        plt.close()