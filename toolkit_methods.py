from __future__ import annotations

import math
import shutil
import string
import subprocess
import tempfile
import threading
import warnings
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple, Union

import matplotlib.pyplot as plt
import nltk  # type: ignore[import-untyped]
import numpy as np
import regex as reg  # type: ignore[import-untyped]
import scipy.optimize as optimize
from nltk.corpus import PlaintextCorpusReader, stopwords  # type: ignore[import-untyped]
from nltk.tokenize import word_tokenize  # type: ignore[import-untyped]
from scipy.optimize import curve_fit, differential_evolution

try:
    import kenlm  # type: ignore[import-not-found]
except ImportError:  # pragma: no cover
    kenlm = None  # type: ignore[assignment]

##########################################################################
#                                CorpusLoader
##########################################################################


class CorpusLoader:
    """
    Load a corpus from NLTK or a local file/directory, optimized for performance.
    """

    def __init__(
        self,
        corpus_source: str,
        allow_download: bool = True,
        custom_download_dir: Optional[str] = None,
    ):
        # Source of the corpus (either a local path or an NLTK corpus name)
        self.corpus_source = corpus_source
        # Flag to allow automatic download from NLTK if the corpus isn't available locally
        self.allow_download = allow_download
        # Custom directory for downloading corpora (if needed)
        self.custom_download_dir = custom_download_dir
        # Cache for loaded corpus to avoid reloading
        self.corpus_cache: Optional[List[str]] = None

    def _download_corpus(self) -> None:
        """Download the corpus from NLTK."""
        # Append custom download directory to NLTK's data path if provided
        if (
            self.custom_download_dir
            and self.custom_download_dir not in nltk.data.path
        ):
            nltk.data.path.append(self.custom_download_dir)
        # Download corpus using NLTK's download utility
        nltk.download(
            self.corpus_source,
            download_dir=self.custom_download_dir,
            quiet=True,
        )

    def _load_corpus_from_path(self, path: Path) -> List[str]:
        """Load the corpus from a local file or directory."""
        # Use PlaintextCorpusReader to read tokens from the local path
        corpus_reader = PlaintextCorpusReader(str(path), ".*")
        # Return all tokens from the corpus
        return [
            str(token)
            for fileid in corpus_reader.fileids()
            for token in corpus_reader.words(fileid)
        ]

    def _load_corpus_from_nltk(self) -> List[str]:
        """Load corpus from NLTK with clear error handling."""
        try:
            corpus_reader = getattr(nltk.corpus, self.corpus_source)
        except AttributeError as e:
            raise LookupError(
                f"NLTK corpus '{self.corpus_source}' not found. "
                f"Try allow_download=True or supply a local path."
            ) from e
        return [
            token
            for fileid in corpus_reader.fileids()
            for token in corpus_reader.words(fileid)
        ]

    def _load_corpus(self) -> List[str]:
        """Load the corpus into memory."""
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
    _STOP_CACHE: Dict[Tuple[str, bool, bool], Set[str]] = {}

    def __init__(
        self,
        remove_stopwords: bool = False,
        remove_punctuation: bool = False,
        use_nltk_tokenizer: bool = True,
        stopwords_language: str = "english",
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
            nltk.download("stopwords", quiet=True)
        if self.use_nltk_tokenizer:
            nltk.download("punkt", quiet=True)

    def _load_unwanted_tokens(self) -> None:
        """Load unwanted tokens with caching for performance."""
        cache_key = (
            self.stopwords_language,
            self.remove_stopwords,
            self.remove_punctuation,
        )
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
        return [
            token
            for token in tokens
            if token not in self._unwanted_tokens
            and not token.startswith("``")
        ]

    def tokenize(
        self, text: Union[str, List[str]], lowercase: bool = False
    ) -> List[str]:
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
            text = " ".join(text)

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

    def __init__(
        self, tokens: List[str], shuffle_tokens: bool = False
    ) -> None:
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
        self.frequency = Counter(
            self.tokens
        )  # Use self.tokens for consistency
        self._total_token_count = sum(self.frequency.values())

        # Generate token details for querying
        self.token_details: Dict[str, Dict[str, Any]] = {}
        # Rank-indexed list for O(1) rank lookups (index 0 = rank 1)
        ranked = self.frequency.most_common()
        self._rank_to_token: List[Tuple[str, int]] = []
        for rank, (token, freq) in enumerate(ranked, 1):
            self.token_details[token] = {"frequency": freq, "rank": rank}
            self._rank_to_token.append((token, freq))

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
                return {"token": token, "frequency": freq}
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
            return {"token": token, **details}
        else:
            raise ValueError(f"Token '{token}' not found in the corpus.")

    def query_by_rank(self, rank: int) -> Dict[str, Union[str, int]]:
        """
        Retrieve token details for a specific rank in the frequency distribution.

        :param rank: Rank to query.
        :return: Dictionary with token details for the given rank.
        :raises ValueError: If the rank is out of range.
        """
        if rank < 1 or rank > len(self._rank_to_token):
            raise ValueError(
                f"Rank {rank} is out of range. Valid ranks are from 1 to {len(self._rank_to_token)}."
            )
        token, freq = self._rank_to_token[rank - 1]
        return {"token": token, "rank": rank, "frequency": freq}

    def cumulative_frequency_analysis(
        self, lower_percent=0, upper_percent=100
    ):
        """
        Analyze tokens within a specific cumulative frequency range.
        Useful for understanding the distribution of common vs. rare tokens.

        :param lower_percent: Lower bound of the cumulative frequency range (in percentage).
        :param upper_percent: Upper bound of the cumulative frequency range (in percentage).
        :return: List of dictionaries with token details in the specified range.
        :raises ValueError: If the provided percentages are out of bounds or lower > upper.
        """
        # Validate percentage inputs
        if (
            not 0 <= lower_percent <= 100
            or not 0 <= upper_percent <= 100
            or lower_percent > upper_percent
        ):
            raise ValueError("Percentages must be 0–100 and lower ≤ upper.")

        # Calculate the numeric thresholds based on percentages
        lower = self._total_token_count * (lower_percent / 100.0)
        upper = self._total_token_count * (upper_percent / 100.0)

        # Extract tokens within the specified cumulative frequency range
        out, cum = [], 0
        for token, freq in self.frequency.most_common():
            cum += freq
            if lower <= cum <= upper:
                out.append({"token": token, **self.token_details[token]})
            elif cum > upper:
                break
        return out

    def list_tokens_in_rank_range(
        self, start_rank: int, end_rank: int
    ) -> List[Dict[str, Any]]:
        """
        List tokens within a specific rank range.
        Useful for examining the most/least frequent subsets of tokens.

        :param start_rank: Starting rank of the range.
        :param end_rank: Ending rank of the range.
        :return: List of dictionaries with token details within the specified rank range.
        :raises ValueError: If the rank range is out of valid bounds.
        """
        if not (1 <= start_rank <= end_rank <= len(self._rank_to_token)):
            raise ValueError(
                f"Rank range is out of valid bounds. Valid ranks are from 1 to {len(self._rank_to_token)}."
            )
        return [
            {"token": token, "frequency": freq, "rank": rank}
            for rank, (token, freq) in enumerate(
                self._rank_to_token[start_rank - 1 : end_rank], start_rank
            )
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
        return {
            token
            for token, details in self.token_details.items()
            if details["frequency"] == x
        }

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

        # Pre-compute ranked frequency array (used by Zipf, ZM, fit_quality, xmin)
        items = self.frequency.most_common()
        self._ranked_freqs = np.array([f for _, f in items], dtype=float)
        self._ranks = np.arange(1, len(items) + 1, dtype=float)

        # Caches for fitted parameters
        self._zipf_alpha: Optional[float] = None
        self._heaps_params: Optional[Tuple[float, float]] = None
        self._zipf_mandelbrot_params: Optional[Tuple[float, float]] = None
        self.herdans_c_value: Optional[float] = None

    def yules_k(self) -> float:
        """
        Calculate Yule's K measure (10^4 * Simpson's concentration index).
        Higher K means MORE repetition / LESS lexical diversity.
        K = 10^4 * sum(f_i*(f_i-1)) / (N*(N-1))
        """
        freqs = np.fromiter(self.frequency.values(), dtype=float)
        N = freqs.sum()
        if N <= 1:
            return 0.0
        return float(1e4 * np.sum(freqs * (freqs - 1)) / (N * (N - 1)))

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
        V = len(self.frequency)  # Distinct token count (Vocabulary size)
        N = self.total_token_count  # Total token count (Corpus size)

        # Check for edge cases to prevent division by zero or log(1)
        if V <= 1 or N <= 1:
            raise ValueError(
                "Vocabulary size (V) or total token count (N) cannot be one or less than one."
            )

        self.herdans_c_value = float(math.log(V) / math.log(N))

        return self.herdans_c_value

    @staticmethod
    def heaps_law(N: np.ndarray, k: float, beta: float) -> np.ndarray:
        """Heap's Law function."""
        return k * (N**beta)

    def generate_corpus_sizes(self, corpus_size: int) -> np.ndarray:
        """Generate a range of corpus sizes for sampling."""
        min_size = min(100, max(10, corpus_size // 1000))
        # Ensure min_size never exceeds corpus_size
        min_size = min(min_size, corpus_size)
        if min_size == corpus_size:
            return np.array([corpus_size])
        corpus_sizes = np.unique(
            np.logspace(
                np.log10(min_size), np.log10(corpus_size), num=60, dtype=int
            )
        )
        return corpus_sizes

    def calculate_vocab_sizes(self, corpus_sizes: np.ndarray) -> List[int]:
        """Single pass vocabulary size calculation with checkpoints."""
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

    _HEAPS_MIN_POINTS = 3  # Minimum sample points needed for a meaningful fit

    def calculate_heaps_law(self) -> Optional[Tuple[float, float]]:
        """
        Estimate Heaps' Law parameters V(N) = K * N^beta.

        Uses the Zipf-Heaps theoretical relationship (beta ≈ 1/alpha) to
        derive an informed initial guess, then fits in log-space
        (log V = log K + beta * log N) for numerical stability before
        refining in the original space.
        """
        if self._heaps_params is not None:
            return self._heaps_params

        corpus_size = self.total_token_count

        # Generate corpus sizes
        corpus_sizes = self.generate_corpus_sizes(corpus_size)

        if len(corpus_sizes) < self._HEAPS_MIN_POINTS:
            warnings.warn(
                f"Corpus too small ({corpus_size} tokens) for reliable "
                f"Heaps' Law fitting (need >= {self._HEAPS_MIN_POINTS} sample points)"
            )
            return None

        # Calculate vocabulary sizes for each corpus size
        vocab_sizes = self.calculate_vocab_sizes(corpus_sizes)

        corpus_arr = np.asarray(corpus_sizes, dtype=float)
        vocab_arr = np.asarray(vocab_sizes, dtype=float)

        # --- Initial guess from Zipf-Heaps relationship: beta ≈ 1/alpha ---
        alpha = self.calculate_zipf_alpha()
        beta_init = min(1.0 / alpha, 0.99) if alpha > 0 else 0.5
        # Derive K_init from the last data point: K ≈ V_last / N_last^beta
        K_init = float(vocab_arr[-1] / (corpus_arr[-1] ** beta_init))

        # --- Step 1: log-space OLS for a robust starting point ---
        log_N = np.log(corpus_arr)
        log_V = np.log(np.maximum(vocab_arr, 1))
        # log V = log K + beta * log N  →  OLS
        A = np.vstack([log_N, np.ones(len(log_N))]).T
        (beta_ols, logK_ols), _, _, _ = np.linalg.lstsq(A, log_V, rcond=None)
        K_ols = np.exp(logK_ols)

        # Pick the better of the two initial guesses
        def _sse(K: float, beta: float) -> float:
            pred = K * corpus_arr**beta
            return float(np.sum((vocab_arr - pred) ** 2))

        if _sse(K_ols, beta_ols) < _sse(K_init, beta_init):
            p0 = [K_ols, beta_ols]
        else:
            p0 = [K_init, beta_init]

        # --- Step 2: weighted nonlinear refinement in original space ---
        weights = np.sqrt(corpus_arr)

        try:
            popt, _ = curve_fit(
                self.heaps_law,
                corpus_arr,
                vocab_arr,
                p0=p0,
                sigma=1.0 / weights,
            )
            self._heaps_params = (float(popt[0]), float(popt[1]))
            return self._heaps_params
        except RuntimeError:
            # If curve_fit fails, fall back to the best initial guess
            if _sse(*p0) < np.inf:
                self._heaps_params = (float(p0[0]), float(p0[1]))
                warnings.warn(
                    "Heaps' Law curve_fit failed; using log-space estimate"
                )
                return self._heaps_params
            warnings.warn("Heaps' Law fitting failed")
            return None

    def estimate_vocabulary_size(self, total_tokens: int) -> int:
        """
        Estimate the vocabulary size for a given number of tokens using Heaps' Law.
        """
        params = self._heaps_params or self.calculate_heaps_law()
        if params is None:
            raise ValueError("Heaps' Law parameters could not be calculated.")
        K, beta = params
        return int(round(K * total_tokens**beta))

    def calculate_zipf_alpha(self) -> float:
        """
        Calculate the alpha parameter for Zipf's Law using Maximum Likelihood Estimation with improved numerical stability.
        """
        if self._zipf_alpha is not None:
            return self._zipf_alpha

        freqs = self._ranked_freqs
        probs = freqs / freqs.sum()
        log_ranks = np.log(self._ranks)

        def nll(alpha: float) -> float:
            if alpha <= 0:
                return np.inf
            log_unnorm = -alpha * log_ranks
            m = log_unnorm.max()
            log_Z = m + np.log(np.exp(log_unnorm - m).sum())  # log-sum-exp
            # Expected negative log-prob under empirical probs:
            return float(log_Z + alpha * np.sum(probs * log_ranks))

        res = optimize.minimize_scalar(
            nll, bounds=(1e-3, 5.0), method="bounded"
        )
        if not res.success:
            raise RuntimeError("Failed to estimate Zipf's alpha parameter")
        self._zipf_alpha = float(res.x)
        return self._zipf_alpha

    def calculate_zipf_mandelbrot(
        self, verbose: bool = False
    ) -> Tuple[float, float]:
        """
        Fit the Zipf-Mandelbrot distribution P(r) ∝ 1/(r+q)^s via MLE,
        seeded from the already-estimated Zipf alpha.

        Uses negative log-likelihood (equivalent to KL-divergence minimisation)
        instead of L2 on probabilities, which avoids head-dominated bias.
        A fast local optimiser (L-BFGS-B) is tried first from an informed
        initial guess; differential_evolution is used as fallback.
        """
        if self._zipf_mandelbrot_params is not None:
            return self._zipf_mandelbrot_params

        freqs = self._ranked_freqs
        N_total = freqs.sum()
        ranks = self._ranks

        # --- MLE objective: negative log-likelihood ---
        # L(q,s) = -sum_r  n_r * log P(r|q,s)
        #        = s * sum_r n_r * log(r+q) + N * log(Z(q,s))
        # where Z(q,s) = sum_r 1/(r+q)^s

        def nll(params: np.ndarray) -> float:
            q, s = params
            if q < 0 or s <= 0:
                return np.inf
            log_rq = np.log(ranks + q)
            log_w = -s * log_rq
            # log-sum-exp for numerical stability of log(Z)
            m = log_w.max()
            log_Z = m + np.log(np.exp(log_w - m).sum())
            return float(s * np.dot(freqs, log_rq) + N_total * log_Z)

        bounds_local = ((1e-4, 20.0), (0.1, 5.0))

        # --- Informed initial guess from Zipf alpha ---
        alpha = self.calculate_zipf_alpha()
        q0, s0 = 1.0, alpha  # s ≈ alpha, q ≈ 1 (Mandelbrot's typical range)

        # Try fast local optimisation first
        local_result = optimize.minimize(
            nll, x0=[q0, s0], method="L-BFGS-B", bounds=bounds_local
        )

        if local_result.success:
            best_q = float(local_result.x[0])
            best_s = float(local_result.x[1])
            best_nll = float(local_result.fun)
        else:
            best_q, best_s, best_nll = q0, s0, nll(np.array([q0, s0]))

        # Fallback: global search if local fit looks poor
        # (compare against pure Zipf NLL as a sanity check)
        zipf_nll = nll(np.array([0.0, alpha]))
        if best_nll > zipf_nll:
            # Local optimiser did worse than pure Zipf — use global search
            de_result = differential_evolution(
                nll, bounds=bounds_local, seed=42, tol=1e-10
            )
            if de_result.success and de_result.fun < best_nll:
                best_q, best_s = de_result.x

        self._zipf_mandelbrot_params = (float(best_q), float(best_s))
        if verbose:
            print(f"Fitted parameters: q = {best_q:.6f}, s = {best_s:.6f}")

        return self._zipf_mandelbrot_params

    # ----------------------- Goodness-of-fit metrics ---------------------- #

    @staticmethod
    def _r_squared(observed: np.ndarray, predicted: np.ndarray) -> float:
        """Coefficient of determination (R²)."""
        ss_res = np.sum((observed - predicted) ** 2)
        ss_tot = np.sum((observed - observed.mean()) ** 2)
        return float(1 - ss_res / ss_tot) if ss_tot > 0 else 0.0

    @staticmethod
    def _ks_statistic(observed: np.ndarray, predicted: np.ndarray) -> float:
        """Kolmogorov-Smirnov statistic between two distributions (CDF-based)."""
        obs_cdf = np.cumsum(observed) / np.sum(observed)
        pred_cdf = np.cumsum(predicted) / np.sum(predicted)
        return float(np.max(np.abs(obs_cdf - pred_cdf)))

    @staticmethod
    def _rmse(observed: np.ndarray, predicted: np.ndarray) -> float:
        """Root mean squared error."""
        return float(np.sqrt(np.mean((observed - predicted) ** 2)))

    def fit_quality(self) -> Dict[str, Dict[str, float]]:
        """
        Evaluate goodness-of-fit for all fitted models.
        Returns R², KS statistic, RMSE, and the Zipf-Heaps consistency
        ratio (beta * alpha — should be close to 1.0 if both fits agree
        with the theoretical relationship beta ≈ 1/alpha).

        Must be called after the individual fit methods.
        """
        freqs = self._ranked_freqs
        ranks = self._ranks
        results: Dict[str, Dict[str, float]] = {}

        # --- Zipf ---
        if self._zipf_alpha is not None:
            alpha = self._zipf_alpha
            pred = freqs[0] / np.power(ranks, alpha)
            results["zipf"] = {
                "alpha": alpha,
                "R2": self._r_squared(freqs, pred),
                "KS": self._ks_statistic(freqs, pred),
                "RMSE": self._rmse(freqs, pred),
            }

        # --- Zipf-Mandelbrot ---
        if self._zipf_mandelbrot_params is not None:
            q, s = self._zipf_mandelbrot_params
            w = 1.0 / np.power(ranks + q, s)
            pred_zm = w * (freqs.sum() / w.sum())
            results["zipf_mandelbrot"] = {
                "q": q,
                "s": s,
                "R2": self._r_squared(freqs, pred_zm),
                "KS": self._ks_statistic(freqs, pred_zm),
                "RMSE": self._rmse(freqs, pred_zm),
            }

        # --- Heaps ---
        if self._heaps_params is not None:
            K, beta = self._heaps_params
            corpus_sizes = self.generate_corpus_sizes(self.total_token_count)
            vocab_sizes = np.asarray(
                self.calculate_vocab_sizes(corpus_sizes), dtype=float
            )
            pred_heaps = K * np.power(corpus_sizes.astype(float), beta)
            results["heaps"] = {
                "K": K,
                "beta": beta,
                "R2": self._r_squared(vocab_sizes, pred_heaps),
                "KS": self._ks_statistic(vocab_sizes, pred_heaps),
                "RMSE": self._rmse(vocab_sizes, pred_heaps),
            }

        # --- Cross-consistency: beta * alpha ≈ 1.0 ---
        if self._zipf_alpha is not None and self._heaps_params is not None:
            alpha = self._zipf_alpha
            _, beta = self._heaps_params
            ratio = beta * alpha
            results["consistency"] = {
                "beta_times_alpha": ratio,
                "deviation_from_1": abs(ratio - 1.0),
            }

        return results

    # -------------------- Zipf x_min estimation (CSN) --------------------- #

    def estimate_zipf_xmin(self) -> Dict[str, Any]:
        """
        Estimate the lower bound (x_min) of the power-law regime using the
        Clauset-Shalizi-Newman KS-minimisation method on frequencies.

        Returns a dict with x_min, alpha (re-estimated above x_min),
        KS statistic, and the number of words in the power-law tail.
        """
        # _ranked_freqs is already descending; we need it for tail slicing
        freqs_desc = self._ranked_freqs
        unique_freqs = np.unique(freqs_desc)

        best_ks = np.inf
        best_xmin = int(unique_freqs[0])
        best_alpha = 1.0

        for xmin_candidate in unique_freqs:
            tail = freqs_desc[freqs_desc >= xmin_candidate]
            if len(tail) < 10:
                continue

            # Continuous MLE approximation for discrete power law (Hill estimator)
            # alpha_hat = 1 + n / sum(ln(x_i / (x_min - 0.5)))
            denom = np.sum(np.log(tail / (xmin_candidate - 0.5)))
            if denom <= 0:
                continue
            n = len(tail)
            alpha_hat = 1.0 + n / denom

            # Theoretical CDF: P(X >= x) = (x / x_min)^(-alpha+1)
            # Empirical CDF from the tail
            tail_sorted = np.sort(tail)
            ecdf = np.arange(1, n + 1) / n
            tcdf = 1.0 - np.power(tail_sorted / xmin_candidate, -alpha_hat + 1)
            tcdf = np.clip(tcdf, 0, 1)
            ks = float(np.max(np.abs(ecdf - tcdf)))

            if ks < best_ks:
                best_ks = ks
                best_xmin = int(xmin_candidate)
                best_alpha = float(alpha_hat)

        return {
            "x_min": best_xmin,
            "alpha": best_alpha,
            "KS": best_ks,
            "n_tail": int(np.sum(freqs_desc >= best_xmin)),
            "n_total": len(freqs_desc),
            "tail_fraction": float(
                np.sum(freqs_desc >= best_xmin) / len(freqs_desc)
            ),
        }


##########################################################################
#                          EntropyCalculator
##########################################################################


class EntropyCalculator(CorpusTools):
    """
    Entropy metrics (H0–H3) with KenLM-backed H3 at character level.
    - Cleaning: letters only, lowercased; each original token becomes a sequence of
      space-separated characters (e.g., 'the' -> 't h e').
    - H0 excludes spaces from the alphabet.
    - H1 includes Miller-Madow bias correction: H + (m-1)/(2*N*ln2).
    - H2 uses unbiased collision probability: sum(n_i*(n_i-1))/(N*(N-1)).
    - H3 uses KenLM n-gram model; score is log10, converted to bits via /log10(2).
    """

    # ------------------------------- Init -------------------------------- #
    def __init__(self, tokens: List[str], q_grams: int = 6) -> None:
        if not (1 <= q_grams <= 12):
            raise ValueError("q_grams must be between 1 and 12")

        # Keep original cleaning: letters-only, lowercased, characters space-separated
        # Filter on the cleaned result (not the original) to avoid empty strings
        cleaned_tokens = []
        for token in tokens:
            alpha_only = reg.sub(r"[^a-zA-Z]", "", token).lower()
            if len(alpha_only) >= 2:
                cleaned_tokens.append(" ".join(alpha_only))
        if not cleaned_tokens:
            raise ValueError(
                "No valid tokens after cleaning for entropy calculation"
            )

        super().__init__(cleaned_tokens)
        self.q_grams = q_grams

        # Pre-compute character data once (used by H0, H1, H2)
        self._chars_no_space: str = "".join(self.tokens).replace(" ", "")
        self._char_counts: Counter = Counter(self._chars_no_space)
        self._char_total: int = len(self._chars_no_space)

        # Character stream for KenLM evaluation: 't h e c a t ...'
        self._char_stream: str = " ".join(self.tokens)

        # Training text: group tokens into ~1000-char chunks so KenLM
        # sees cross-word context (not one-word-per-line which wastes
        # n-gram capacity on sentence boundaries).
        lines: List[str] = []
        current: List[str] = []
        length = 0
        for tok in self.tokens:
            current.append(tok)
            length += len(tok.split()) + 1
            if length >= 1000:
                lines.append(" ".join(current))
                current = []
                length = 0
        if current:
            lines.append(" ".join(current))
        self._train_text: str = "\n".join(lines)

        self._h3_cache: Optional[float] = None

    # --------------------------- H0 / H1 / H2 ---------------------------- #
    def calculate_H0(self) -> float:
        """Zeroth-order entropy over the observed alphabet (spaces excluded)."""
        if not self._char_counts:
            return 0.0
        return math.log2(len(self._char_counts))

    def calculate_H1(self) -> float:
        """
        Shannon entropy (bits/char) with Miller-Madow bias correction.
        H_MM = H_plugin + (m - 1) / (2 * N * ln(2))
        """
        N = self._char_total
        if N == 0:
            return 0.0
        m = len(self._char_counts)
        H_plugin = -sum(
            (c / N) * math.log2(c / N) for c in self._char_counts.values()
        )
        correction = (m - 1) / (2 * N * math.log(2))
        return H_plugin + correction

    def calculate_H2(self) -> float:
        """
        Rényi entropy of order 2 using unbiased collision probability
        estimator: sum(n_i*(n_i-1)) / (N*(N-1)).
        """
        N = self._char_total
        if N <= 1:
            return 0.0
        collision_prob = sum(
            c * (c - 1) for c in self._char_counts.values()
        ) / (N * (N - 1))
        if collision_prob <= 0:
            return 0.0
        return float(-math.log2(collision_prob))

    # ------------------------------ KenLM -------------------------------- #
    @staticmethod
    def _require(exe: str) -> None:
        if shutil.which(exe) is None:
            raise FileNotFoundError(
                f"Required KenLM tool '{exe}' not found in PATH. "
                "Install KenLM and ensure its bin/ directory is on PATH."
            )

    def _ensure_kenlm_tools(self) -> None:
        self._require("lmplz")
        self._require("build_binary")

    def train_kenlm_model(self, text: str) -> Tuple[Path, Path]:
        """
        Train KenLM from the *provided* text.
        Expectation (as used here):
          - `text` is already space-tokenized at the character level.
          - one sentence per line (so </s> exists for build_binary).
        """
        self._ensure_kenlm_tools()

        tmp = Path(tempfile.mkdtemp())
        txt = tmp / "corpus.txt"
        arpa = tmp / "model.arpa"
        klm = tmp / "model.klm"

        txt.write_text(text, encoding="utf-8")

        # Build ARPA (stdin -> stdout) with discount_fallback, same as before
        with txt.open("rb") as fin, arpa.open("wb") as fout:
            subprocess.run(
                ["lmplz", "-o", str(self.q_grams), "--discount_fallback"],
                check=True,
                stdin=fin,
                stdout=fout,
                stderr=subprocess.DEVNULL,
            )

        # Build binary (strict; requires </s> present)
        subprocess.run(
            ["build_binary", str(arpa), str(klm)],
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )

        return klm, tmp

    def calculate_H3_kenlm(self) -> float:
        """
        Character-level cross-entropy (bits/char) using a KenLM n-gram model.

        Trained on continuous text chunks so the model learns cross-word
        character dependencies. KenLM .score() returns log10(P); converted
        to bits via / log10(2). Normalized by total character tokens T.
        """
        if self._h3_cache is not None:
            return self._h3_cache

        if kenlm is None:
            raise ImportError(
                "KenLM is required for H3 calculation. "
                "Install it with: pip install kenlm"
            )
        model_path, tmp = self.train_kenlm_model(self._train_text)
        try:
            model = kenlm.Model(str(model_path))

            # KenLM .score() returns total log10(P)
            total_log10 = model.score(self._char_stream, bos=False, eos=False)
            # Convert log10 → log2 (bits)
            total_log2 = total_log10 / math.log10(2)

            T = len(self._char_stream.split())  # number of character tokens
            self._h3_cache = float(-total_log2 / max(1, T))
            return self._h3_cache
        finally:
            shutil.rmtree(tmp, ignore_errors=True)

    # ---------------------------- Redundancy ----------------------------- #
    @staticmethod
    def calculate_redundancy(H3: float, H0: float) -> float:
        """Redundancy (%) relative to alphabet capacity: R = (1 - H3/H0) * 100."""
        return (1 - H3 / H0) * 100 if H0 > 0 else 0.0


##########################################################################
#                              CorpusPlots
##########################################################################


class CorpusPlots:
    """
    Minimal, robust plotting for Zipf, Zipf–Mandelbrot, and Heaps.

    Changes requested:
      - Heaps' Law back to original *linear* scaling (no log axes by default).
      - Distinct color scheme per plot type:
          * Zipf:           points=tab:blue, fit=tab:red
          * Zipf–Mandelbrot points=navy,     fit=tab:red
          * Heaps:          empirical=tab:blue, fit=tab:red (dashed)
    """

    def __init__(
        self,
        analyzer,
        corpus_name: str,
        plots_dir: str | Path = "plots",
        *,
        dpi: int = 300,
        file_format: str = "png",
        transparent: bool = False,
    ) -> None:
        self.analyzer = analyzer
        self.corpus_name = str(corpus_name)
        self.plots_dir = Path(plots_dir)
        self.plots_dir.mkdir(parents=True, exist_ok=True)
        self.dpi = int(dpi)
        self.file_format = str(file_format)
        self.transparent = bool(transparent)

        # Consistent rank/frequency arrays from most_common()
        items = self.analyzer.frequency.most_common()
        self._ranks = np.arange(1, len(items) + 1, dtype=float)
        self._freqs = np.array([f for _, f in items], dtype=float)
        self._freqs_norm_max = (
            self._freqs / self._freqs.max()
            if self._freqs.size
            else self._freqs
        )

    # ----------------------------- helpers ------------------------------ #
    def _new_axes(
        self,
        title: str,
        xlabel: str,
        ylabel: str,
        figsize: Tuple[int, int] = (12, 8),
    ):
        fig, ax = plt.subplots(figsize=figsize, dpi=self.dpi)
        ax.set_title(f"{title} — {self.corpus_name}")
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.grid(True, which="both", linestyle="--", alpha=0.5)
        return fig, ax

    def _save(self, fig, stem: str) -> Path:
        out = self.plots_dir / f"{stem}_{self.corpus_name}.{self.file_format}"
        fig.tight_layout()
        fig.savefig(out, dpi=self.dpi, transparent=self.transparent)
        plt.close(fig)
        return out

    @staticmethod
    def _downsample(
        x: np.ndarray, y: np.ndarray, *, top_n: Optional[int], stride: int
    ):
        """Optional point thinning for faster scatter; math unchanged."""
        if top_n is not None and top_n > 0:
            x, y = x[:top_n], y[:top_n]
        if stride > 1:
            x, y = x[::stride], y[::stride]
        return x, y

    # ---------------------------- Zipf plot ----------------------------- #
    def plot_zipfs_law_fit(
        self,
        *,
        top_n: Optional[int] = None,
        stride: int = 1,
        figsize: Tuple[int, int] = (12, 8),
    ) -> Path:
        alpha = self.analyzer.calculate_zipf_alpha()
        if alpha is None:
            raise ValueError(
                "Alpha calculation failed, cannot plot Zipf's Law fit."
            )

        ranks = self._ranks
        y_emp = self._freqs_norm_max
        y_fit = (1.0 / np.power(ranks, alpha)) / (
            1.0 / np.power(ranks[0], alpha)
        )

        x_sc, y_sc = self._downsample(ranks, y_emp, top_n=top_n, stride=stride)

        fig, ax = self._new_axes(
            "Zipf's Law Fit", "Rank", "Normalized Frequency", figsize
        )
        ax.set_xscale("log")
        ax.set_yscale("log")

        # Distinct colors: blue points, red fit
        ax.plot(
            x_sc,
            y_sc,
            linestyle="none",
            marker="o",
            color="tab:blue",
            markersize=3,
            alpha=0.5,
            label="Empirical",
        )
        ax.plot(
            ranks,
            y_fit,
            color="tab:red",
            linewidth=2,
            label=f"Zipf Fit (α={alpha:.3f})",
        )

        ax.legend()
        return self._save(fig, "zipfs_law_fit")

    # ---------------------- Zipf–Mandelbrot plot ------------------------ #
    def plot_zipf_mandelbrot_fit(
        self,
        *,
        top_n: Optional[int] = None,
        stride: int = 1,
        figsize: Tuple[int, int] = (12, 8),
    ) -> Path:
        params = self.analyzer.calculate_zipf_mandelbrot()
        if params is None:
            raise ValueError("Zipf–Mandelbrot parameter calculation failed.")
        q, s = params

        ranks = self._ranks
        y_emp = self._freqs_norm_max
        w = 1.0 / np.power(ranks + q, s)
        y_fit = w / w[0]

        x_sc, y_sc = self._downsample(ranks, y_emp, top_n=top_n, stride=stride)

        fig, ax = self._new_axes(
            "Zipf–Mandelbrot Fit", "Rank", "Normalized Frequency", figsize
        )
        ax.set_xscale("log")
        ax.set_yscale("log")

        # Distinct colors: navy points, red fit
        ax.plot(
            x_sc,
            y_sc,
            linestyle="none",
            marker="o",
            color="navy",
            markersize=3,
            alpha=0.5,
            label="Empirical",
        )
        ax.plot(
            ranks,
            y_fit,
            color="tab:red",
            linewidth=2,
            label=f"ZM Fit (q={q:.3f}, s={s:.3f})",
        )

        ax.legend()
        return self._save(fig, "zipf_mandelbrot_fit")

    # ---------------------------- Heaps plot ---------------------------- #
    def plot_heaps_law(
        self,
        *,
        figsize: Tuple[int, int] = (12, 8),
    ) -> Path:
        """
        Heaps' Law with original *linear* scaling (no log axes).
        """
        params = self.analyzer.calculate_heaps_law()
        if params is None:
            raise ValueError("Heaps' Law parameters calculation failed.")
        K, beta = params

        corpus_size = len(self.analyzer.tokens)
        corpus_sizes = self.analyzer.generate_corpus_sizes(corpus_size)
        vocab_sizes = self.analyzer.calculate_vocab_sizes(corpus_sizes)

        fig, ax = self._new_axes(
            "Heaps' Law Analysis", "Token Count", "Type Count", figsize
        )

        # Distinct colors: blue empirical, red dashed fit; linear axes
        ax.plot(
            corpus_sizes,
            vocab_sizes,
            color="tab:blue",
            marker="o",
            markersize=3,
            linewidth=1.5,
            alpha=0.7,
            label="Empirical",
        )
        ax.plot(
            corpus_sizes,
            K * np.power(corpus_sizes, beta),
            linestyle="--",
            color="tab:red",
            linewidth=2,
            label=f"Heaps Fit: K={K:.2f}, β={beta:.3f}",
        )

        ax.legend()
        return self._save(fig, "heaps_law")
