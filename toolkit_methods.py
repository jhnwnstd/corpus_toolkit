import math
import shutil
import string
import subprocess
import sys
import tempfile
import threading
import warnings
from collections import Counter
from pathlib import Path
from typing import List, Set, Union, Tuple, Optional, Dict, Any

import kenlm
import matplotlib.pyplot as plt
import nltk
import numpy as np
import regex as reg
from nltk.corpus import PlaintextCorpusReader, stopwords
from nltk.tokenize import word_tokenize
import scipy.optimize as optimize
from scipy.optimize import curve_fit, differential_evolution

##########################################################################
#                                CorpusLoader
##########################################################################

class CorpusLoader:
    """
    Load a corpus from NLTK or a local file/directory, optimized for performance.
    """

    def __init__(self, corpus_source: str, allow_download: bool = True, custom_download_dir: Optional[str] = None):
        # Source of the corpus (either a local path or an NLTK corpus name)
        self.corpus_source = corpus_source
        # Flag to allow automatic download from NLTK if the corpus isn't available locally
        self.allow_download = allow_download
        # Custom directory for downloading corpora (if needed)
        self.custom_download_dir = custom_download_dir
        # Cache for loaded corpus to avoid reloading
        self.corpus_cache = None

    def _download_corpus(self) -> None:
        """Download the corpus from NLTK."""
        # Append custom download directory to NLTK's data path if provided
        if self.custom_download_dir and self.custom_download_dir not in nltk.data.path:
            nltk.data.path.append(self.custom_download_dir)
        # Download corpus using NLTK's download utility
        nltk.download(self.corpus_source, download_dir=self.custom_download_dir, quiet=True)

    def _load_corpus_from_path(self, path: Path) -> List[str]:
        """Load the corpus from a local file or directory."""
        # Use PlaintextCorpusReader to read tokens from the local path
        corpus_reader = PlaintextCorpusReader(str(path), '.*')
        # Return all tokens from the corpus
        return [token for fileid in corpus_reader.fileids() for token in corpus_reader.words(fileid)]

    def _load_corpus_from_nltk(self) -> List[str]:
        """Load corpus from NLTK with clear error handling."""
        try:
            corpus_reader = getattr(nltk.corpus, self.corpus_source)
        except AttributeError as e:
            raise LookupError(
                f"NLTK corpus '{self.corpus_source}' not found. "
                f"Try allow_download=True or supply a local path."
            ) from e
        return [token for fileid in corpus_reader.fileids() 
                for token in corpus_reader.words(fileid)]

    def _load_corpus(self) -> List[str]:
        """Load the corpus into memory.
        
        For very large corpora, consider using a generator to yield batches:
        (Uncomment the generator-based approach for efficient streaming.)
        
        Example generator-based approach:
        
        # def _load_corpus(self, batch_size=10000):
        #     path = Path(self.corpus_source)
        #     if path.is_file() or path.is_dir():
        #         corpus_reader = PlaintextCorpusReader(str(path), '.*')
        #     else:
        #         corpus_reader = getattr(nltk.corpus, self.corpus_source)
        #     for fileid in corpus_reader.fileids():
        #         tokens_iter = iter(corpus_reader.words(fileid))
        #         while True:
        #             batch = list(itertools.islice(tokens_iter, batch_size))
        #             if not batch:
        #                 break
        #             yield from batch
        """
        # Determine if the source is a local path or an NLTK corpus name
        path = Path(self.corpus_source)
        if path.is_file() or path.is_dir():
            return self._load_corpus_from_path(path)
        else:
            return self._load_corpus_from_nltk()

    def load_corpus(self) -> List[str]:
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

    def is_corpus_available(self) -> bool:
        """Robust availability check for local paths and NLTK corpora."""
        path = Path(self.corpus_source)
        if path.exists():
            return True
        try:
            # Probe actual corpus accessor for more reliable detection
            corpus = getattr(nltk.corpus, self.corpus_source)
            _ = corpus.fileids()  # Test access
            return True
        except (AttributeError, LookupError):
            return False


##########################################################################
#                                Tokenizer
##########################################################################

class Tokenizer:
    """
    High-performance tokenizer with caching and robust defaults.
    """
    
    # Class-level cache for stopwords to avoid repeated downloads
    _STOP_CACHE: Dict[Tuple[str, bool], Set[str]] = {}

    def __init__(
        self,
        remove_stopwords: bool = False,
        remove_punctuation: bool = False,
        use_nltk_tokenizer: bool = True,
        stopwords_language: str = 'english'
    ) -> None:
        self.remove_stopwords = remove_stopwords
        self.remove_punctuation = remove_punctuation
        self.use_nltk_tokenizer = use_nltk_tokenizer
        self.stopwords_language = stopwords_language
        self.custom_regex = None
        self._unwanted_tokens: Set[str] = set()
        self._initialized = False
        self._init_lock = threading.Lock()

    def _initialize(self) -> None:
        """Initialize the tokenizer by ensuring NLTK resources are available and loading unwanted tokens."""
        with self._init_lock:
            if not self._initialized:
                self._ensure_nltk_resources()
                self._load_unwanted_tokens()
                self._initialized = True

    def _ensure_nltk_resources(self) -> None:
        """Ensure that NLTK resources are available."""
        if self.remove_stopwords:
            nltk.download('stopwords', quiet=True)
        if self.use_nltk_tokenizer:
            nltk.download('punkt', quiet=True)

    def _load_unwanted_tokens(self) -> None:
        """Load unwanted tokens with caching for performance."""
        cache_key = (self.stopwords_language, self.remove_punctuation)
        cached = self._STOP_CACHE.get(cache_key)
        
        if cached is not None:
            self._unwanted_tokens = set(cached)
            return

        unwanted = set()
        if self.remove_stopwords:
            unwanted.update(stopwords.words(self.stopwords_language))
        if self.remove_punctuation:
            unwanted.update(string.punctuation)
        
        self._STOP_CACHE[cache_key] = set(unwanted)
        self._unwanted_tokens = unwanted

    def set_custom_regex(self, pattern: str) -> None:
        """Set a custom regex pattern for tokenization."""
        # Compile regex pattern for custom tokenization
        try:
            self.custom_regex = reg.compile(pattern)
        except reg.error as e:
            raise ValueError(f"Invalid regex pattern: {e}")

    def _remove_unwanted_tokens(self, tokens: List[str]) -> List[str]:
        """Remove unwanted tokens (stopwords, punctuation) from a list of tokens."""
        # Filter out tokens present in the unwanted tokens set
        return [token for token in tokens if token not in self._unwanted_tokens and not token.startswith('``')]

    def tokenize(self, text: Union[str, List[str]], lowercase: bool = False) -> List[str]:
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


##########################################################################
#                              CorpusTools
##########################################################################

class CorpusTools:
    """
    Provides basic corpus analysis tools like frequency distribution and querying capabilities.
    """

    def __init__(self, tokens: List[str], shuffle_tokens: bool = False) -> None:
        """
        Initialize CorpusTools with input validation.
        
        :param tokens: List of tokens (words) in the corpus.
        :param shuffle_tokens: Boolean indicating whether to shuffle the tokens.
        """
        if not tokens:
            raise ValueError("Token list cannot be empty.")
        if not all(isinstance(token, str) for token in tokens):
            raise ValueError("All tokens must be strings.")

        # Shuffle the tokens if required
        if shuffle_tokens:
            tokens = tokens.copy()
            np.random.shuffle(tokens)

        # Store tokens as a class attribute for later use
        self.tokens = tokens  # Define self.tokens as an attribute

        # Calculate frequency distribution
        self.frequency = Counter(self.tokens)  # Use self.tokens for consistency
        self._total_token_count = sum(self.frequency.values())

        # Generate token details for querying
        self.token_details: Dict[str, Dict[str, Any]] = {}
        for rank, (token, freq) in enumerate(self.frequency.most_common(), 1):
            self.token_details[token] = {'frequency': freq, 'rank': rank}

    @property
    def total_token_count(self) -> int:
        """
        Return the total number of tokens in the corpus.
        """
        return self._total_token_count

    def find_median_token(self) -> Dict[str, Union[str, int]]:
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
        return {}

    def mean_token_frequency(self) -> float:
        """
        Calculate the mean frequency of tokens in the corpus.

        :return: Float representing the mean token frequency.
        """
        return self.total_token_count / len(self.token_details)

    def query_by_token(self, token: str) -> Dict[str, Union[str, int]]:
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

    def query_by_rank(self, rank: int) -> Dict[str, Union[str, int]]:
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

    def cumulative_frequency_analysis(self, lower_percent=0, upper_percent=100):
        """
        Analyze tokens within a specific cumulative frequency range. 
        Useful for understanding the distribution of common vs. rare tokens.

        :param lower_percent: Lower bound of the cumulative frequency range (in percentage).
        :param upper_percent: Upper bound of the cumulative frequency range (in percentage).
        :return: List of dictionaries with token details in the specified range.
        :raises ValueError: If the provided percentages are out of bounds or lower > upper.
        """
        # Validate percentage inputs
        if not 0 <= lower_percent <= 100 or not 0 <= upper_percent <= 100 or lower_percent > upper_percent:
            raise ValueError("Percentages must be 0–100 and lower ≤ upper.")

        # Calculate the numeric thresholds based on percentages
        lower = self._total_token_count * (lower_percent / 100.0)
        upper = self._total_token_count * (upper_percent / 100.0)

        # Extract tokens within the specified cumulative frequency range
        out, cum = [], 0
        for token, freq in self.frequency.most_common():
            cum += freq
            if lower <= cum <= upper:
                out.append({'token': token, **self.token_details[token]})
            elif cum > upper:
                break
        return out

    def list_tokens_in_rank_range(self, start_rank: int, end_rank: int) -> List[Dict[str, Any]]:
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
        return [
            {'token': token, **details}
            for token, details in self.token_details.items()
            if start_rank <= details['rank'] <= end_rank
        ]

    def x_legomena(self, x: int) -> Set[str]:
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

    def vocabulary(self) -> Set[str]:
        """
        Return the set of distinct tokens in the corpus.
        """
        return set(self.frequency.keys())


##########################################################################
#                             AdvancedTools
##########################################################################

class AdvancedTools(CorpusTools):
    """
    Advanced analysis on a corpus of text, extending CorpusTools.
    Implements advanced linguistic metrics and statistical law calculations.
    """

    def __init__(self, tokens: List[str]) -> None:
        super().__init__(tokens)
        # Caches to store precomputed parameters for efficient reuse
        self._zipf_alpha: Optional[float] = None  # For Zipf's Law
        self._zipf_c: Optional[float] = None      # For Zipf's Law
        self._heaps_params: Optional[Tuple[float, float]] = None  # For Heaps' Law
        self._zipf_mandelbrot_params: Optional[Tuple[float, float]] = None  # For Zipf-Mandelbrot Law
        self.herdans_c_value: Optional[float] = None  # For Herdan's C measure

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
        return float(K)

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
    def heaps_law(N: float, k: float, beta: float) -> float:
        """Heap's Law function."""
        return k * (N ** beta)

    def generate_corpus_sizes(self, corpus_size: int) -> np.ndarray:
        """Generate a range of corpus sizes for sampling."""
        min_size = min(100, max(10, corpus_size // 1000))
        corpus_sizes = np.unique(
            np.logspace(np.log10(min_size), np.log10(corpus_size), num=60, dtype=int)
        )
        return corpus_sizes

    def calculate_vocab_sizes(self, corpus_sizes: np.ndarray) -> List[int]:
        """FIXED: Single pass vocabulary size calculation with checkpoints."""
        checkpoints = set(int(s) for s in corpus_sizes)
        seen: Set[str] = set()
        vocab_sizes = []
        checkpoint_results = {}
        
        for idx, token in enumerate(self.tokens, 1):
            seen.add(token)
            if idx in checkpoints:
                checkpoint_results[idx] = len(seen)
                if len(checkpoint_results) == len(checkpoints):
                    break
        
        # Return results in order of corpus_sizes
        for size in corpus_sizes:
            vocab_sizes.append(checkpoint_results.get(int(size), len(seen)))
        
        return vocab_sizes

    def calculate_heaps_law(self) -> Optional[Tuple[float, float]]:
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
            popt, _ = curve_fit(
                self.heaps_law,
                corpus_sizes,
                vocab_sizes,
                p0=[1, 0.5],
                sigma=1 / weights
            )
            self._heaps_params = tuple(popt)
            return self._heaps_params
        except RuntimeError:
            warnings.warn("Heaps' Law fitting failed")
            return None

    def estimate_vocabulary_size(self, total_tokens: int) -> int:
        """
        Estimate the vocabulary size for a given number of tokens using Heaps' Law.
        Ensures that the output is an integer, as vocabulary size cannot be fractional.
        Use to evaluate Heaps' law fit by comparing with actual vocabulary size.
        """
        if self._heaps_params is None:
            self._heaps_params = self.calculate_heaps_law()  # Calculate parameters if not already cached

        if self._heaps_params is None:
            # If we still don't have parameters, return a default or raise an error
            raise ValueError("Heaps' Law parameters could not be calculated.")

        K, beta = self._heaps_params  # Retrieve parameters
        estimated_vocab_size = K * (total_tokens ** beta)  # Calculate vocabulary size

        # Round the estimated vocabulary size to the nearest integer
        return int(round(estimated_vocab_size))

    def calculate_zipf_alpha(self) -> float:
        """
        Calculate the alpha parameter for Zipf's Law using Maximum Likelihood Estimation with improved numerical stability.
        """
        if self._zipf_alpha is not None:
            return self._zipf_alpha

        freqs = np.array([f for _, f in self.frequency.most_common()], float)
        n = len(freqs)
        probs = freqs / freqs.sum()
        log_ranks = np.log(np.arange(1, n + 1, dtype=float))

        def nll(alpha: float) -> float:
            if alpha <= 0:
                return np.inf
            log_unnorm = -alpha * log_ranks
            m = log_unnorm.max()
            log_Z = m + np.log(np.exp(log_unnorm - m).sum())  # log-sum-exp
            # Expected negative log-prob under empirical probs:
            return float(log_Z + alpha * np.sum(probs * log_ranks))

        res = optimize.minimize_scalar(nll, bounds=(1e-3, 5.0), method='bounded')
        if not res.success:
            raise RuntimeError("Failed to estimate Zipf's alpha parameter")
        self._zipf_alpha = float(res.x)
        return self._zipf_alpha

    def calculate_zipf_mandelbrot(self, verbose: bool = False) -> Tuple[float, float]:
        """
        Fit the Zipf-Mandelbrot distribution to the corpus and find parameters q and s.
        """
        if self._zipf_mandelbrot_params is not None:
            return self._zipf_mandelbrot_params

        freqs = np.array([f for _, f in self.frequency.most_common()], float)
        ranks = np.arange(1, len(freqs) + 1, dtype=float)
        p_emp = freqs / freqs.sum()

        def model(k, q, s):
            w = 1.0 / np.power(k + q, s)
            return w / w.sum()

        def objective(params):
            q, s = params
            if q <= 0 or s <= 0:
                return np.inf
            return float(np.sum((model(ranks, q, s) - p_emp) ** 2))

        bounds = [(0.01, 10.0), (0.5, 2.5)]
        result = differential_evolution(objective, bounds)
        if result.success:
            q, s = result.x
            self._zipf_mandelbrot_params = (q, s)
            if verbose:
                print(f"Fitted parameters: q = {q}, s = {s}")
        else:
            raise RuntimeError("Failed to estimate Zipf-Mandelbrot parameters")

        return self._zipf_mandelbrot_params


##########################################################################
#                          EntropyCalculator
##########################################################################

class EntropyCalculator(CorpusTools):
    def __init__(self, tokens: List[str], q_grams: int = 6) -> None:
        if not (1 <= q_grams <= 12):
            raise ValueError("q_grams must be between 1 and 12")
        # Preprocess tokens similarly to the original method
        cleaned_tokens = [
            ' '.join(reg.sub(r'[^a-zA-Z]', '', token).lower())
            for token in tokens if len(token) >= 2
        ]
        if not cleaned_tokens:
            raise ValueError("No valid tokens after cleaning for entropy calculation")
        super().__init__(cleaned_tokens)
        self.q_grams = q_grams

    def calculate_H0(self) -> float:
        """Calculate the zeroth-order entropy (H0)."""
        # Assuming alphabet consists of only letters and space, case ignored
        alphabet = set(''.join(self.tokens))
        alphabet_size = len(alphabet)
        return math.log2(alphabet_size)

    def calculate_H1(self) -> float:
        """Calculate first-order entropy with better performance."""
        text = ''.join(self.tokens).replace(' ', '')
        if not text:
            return 0.0
            
        letter_freq = Counter(text)
        total_letters = len(text)  # More efficient than sum()
        
        return -sum((freq / total_letters) * math.log2(freq / total_letters) 
                   for freq in letter_freq.values())

    def calculate_H2(self) -> float:
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
        
        return float(H2)

    def _ensure_kenlm_tools(self) -> None:
        """Verify KenLM tools are available."""
        for exe in ("lmplz", "build_binary"):
            if shutil.which(exe) is None:
                raise FileNotFoundError(
                    f"Required KenLM tool '{exe}' not found in PATH. "
                    "Install KenLM and ensure its bin/ directory is on PATH."
                )

    def train_kenlm_model(self, text: str) -> Tuple[Path, Path]:
        """SECURITY FIX: Train KenLM model without shell injection vulnerability."""
        self._ensure_kenlm_tools()
        
        tempdir_path = Path(tempfile.mkdtemp())
        text_file_path = tempdir_path / "corpus.txt"
        arpa_file_path = tempdir_path / "model.arpa"
        model_file_path = tempdir_path / "model.klm"
        
        # Write corpus safely
        text_file_path.write_text('\n'.join(self.tokens), encoding='utf-8')
        
        # Run lmplz safely without shell injection
        with text_file_path.open('rb') as fin, arpa_file_path.open('wb') as fout:
            subprocess.run(
                ["lmplz", "-o", str(self.q_grams), "--discount_fallback"],
                check=True,
                stdin=fin,
                stdout=fout,
                stderr=subprocess.DEVNULL
            )
        
        # Run build_binary safely
        subprocess.run(
            ["build_binary", str(arpa_file_path), str(model_file_path)],
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL
        )
        
        return model_file_path, tempdir_path

    def calculate_H3_kenlm(self) -> float:
        model_path, tempdir_path = self.train_kenlm_model(' '.join(self.tokens))
        model = kenlm.Model(str(model_path))
        
        prepared_text = ' '.join(self.tokens)
        log_prob = model.score(prepared_text, bos=False, eos=False) / math.log(2)
        num_tokens = len(prepared_text.split()) - self.q_grams + 1
        H3 = -log_prob / num_tokens
        
        # Cleanup: remove the temporary directory after the model has been used
        shutil.rmtree(tempdir_path)
        
        return float(H3)

    def calculate_redundancy(self, H3: float, H0: float) -> float:
        """Calculate redundancy based on H3 and H0."""
        return (1 - H3 / H0) * 100

    def cleanup_temp_directory(self) -> None:
        """Remove the temporary directory used for KenLM model training."""
        if hasattr(self, 'tempdir_path') and self.tempdir_path.exists():
            shutil.rmtree(self.tempdir_path)
            print("Temporary directory cleaned up.")


##########################################################################
#                              CorpusPlots
##########################################################################

class CorpusPlots:
    def __init__(self, analyzer: CorpusTools, corpus_name: str, plots_dir: str = 'plots') -> None:
        self.analyzer = analyzer
        self.corpus_name = corpus_name
        self.plots_dir = Path(plots_dir)
        self.plots_dir.mkdir(exist_ok=True)

    def _setup_plot(self, title: str, xlabel: str, ylabel: str, figsize: Tuple[int, int] = (12, 8)) -> None:
        """Set up a new plot with common elements."""
        plt.figure(figsize=figsize)
        plt.title(f"{title} for {self.corpus_name} Corpus")
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.grid(True, which="both", ls="--", alpha=0.5)

    def _save_plot(self, filename: str) -> None:
        """Save the current plot and close the figure."""
        plt.tight_layout()
        plt.savefig(self.plots_dir / f'{filename}_{self.corpus_name}.png', dpi=300)
        plt.close()

    def plot_zipfs_law_fit(self) -> None:
        alpha = self.analyzer.calculate_zipf_alpha()
        if alpha is None:
            raise ValueError("Alpha calculation failed, cannot plot Zipf's Law fit.")

        ranks = np.arange(1, len(self.analyzer.frequency) + 1)
        frequencies = np.array([freq for _, freq in self.analyzer.frequency.most_common()])
        
        self._setup_plot("Zipf's Law Fit", "Rank", "Normalized Frequency")
        
        plt.loglog(ranks, frequencies / frequencies.max(), 'o', label='Actual Frequencies', markersize=3, alpha=0.5)
        plt.loglog(
            ranks,
            (1 / np.power(ranks, alpha)) / (1 / np.power(ranks[0], alpha)),
            label=f"Zipf's Law Fit (α={alpha:.3f})",
            color='red',
            linewidth=2
        )
        
        plt.legend()
        self._save_plot('zipfs_law_fit')

    def plot_zipf_mandelbrot_fit(self) -> None:
        params = self.analyzer.calculate_zipf_mandelbrot()
        if params is None:
            raise ValueError("Zipf-Mandelbrot parameter calculation failed.")
        q, s = params

        ranks = np.array([details['rank'] for details in self.analyzer.token_details.values()])
        frequencies = np.array([details['frequency'] for details in self.analyzer.token_details.values()])

        def zipf_mandelbrot(k: np.ndarray, q: float, s: float) -> np.ndarray:
            return 1 / np.power(k + q, s)

        self._setup_plot("Zipf-Mandelbrot Fit", "Rank", "Normalized Frequency")

        plt.loglog(
            ranks,
            frequencies / frequencies.max(),
            'o',
            label='Actual Frequencies',
            markersize=3,
            alpha=0.5,
            color='navy'  # Changed to a darker blue
        )
        plt.loglog(
            ranks,
            zipf_mandelbrot(ranks, q, s) / zipf_mandelbrot(ranks[0], q, s),
            label=f"Zipf-Mandelbrot Fit (q={q:.3f}, s={s:.3f})",
            color='red',
            linewidth=2
        )

        plt.legend()
        self._save_plot('zipf_mandelbrot_fit')

    def plot_heaps_law(self) -> None:
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
        plt.plot(
            corpus_sizes,
            K * np.power(corpus_sizes, beta),
            '--',
            label=f"Heaps' Law Fit: K={K:.2f}, β={beta:.3f}",
            color='red',
            linewidth=2
        )

        plt.legend()
        self._save_plot('heaps_law')