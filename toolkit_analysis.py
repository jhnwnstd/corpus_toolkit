from __future__ import annotations

import logging
from typing import Iterable, List, Sequence

import nltk
from toolkit_methods import (  # type: ignore
    CorpusLoader,
    CorpusPlots,
    EntropyCalculator,
    Tokenizer,
    CorpusTools,
    AdvancedTools,
)

# ------------------------------- Logging -------------------------------- #

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger("CorpusAnalysis")

SEP = "=" * 50
SUB = "-" * 28
SUB_SHORT = "-" * 22
SUB_TINY = "-" * 20


def _section(title: str) -> None:
    logger.info("\n" + SEP)
    logger.info(title.center(50))
    logger.info(SEP)


# ------------------------------- Config --------------------------------- #

# Change this to ONE corpus (str) or MANY (list/tuple of str):
# Examples: 'brown'  OR  ['brown', 'gutenberg', 'reuters']
corpus_name: str | list[str] | tuple[str, ...] = 'brown', 'gutenberg', 'reuters', 'inaugural', 'webtext', 'nps_chat'

# Download required NLTK data (quiet, unconditional as in original)
if isinstance(corpus_name, str):
    nltk.download(corpus_name, quiet=True)
else:
    for name in corpus_name:
        nltk.download(name, quiet=True)

nltk.download("stopwords", quiet=True)
nltk.download("punkt", quiet=True)


# ------------------------------ Pipeline -------------------------------- #

def load_and_tokenize_corpus(name: str) -> List[str]:
    """
    Load and tokenize the specified NLTK corpus.
    Preserves original behavior: remove_punctuation=True, lowercase=True.
    """
    logger.info(f"\nLoading and tokenizing {name}")
    corpus_tokens = CorpusLoader(name).load_corpus()

    tokenizer = Tokenizer(remove_punctuation=True)  # keep default behavior
    tokenized = tokenizer.tokenize(corpus_tokens, lowercase=True)
    logger.info(f"Completed tokenizing {name}\n")
    return tokenized


def analyze_corpus(tokenized: Sequence[str], shuffle: bool = True) -> AdvancedTools:
    """
    Run the complete analysis suite on the tokenized corpus.
    When shuffle=True, the flag is passed through so CorpusTools handles the shuffling,
    and the same order is used consistently in AdvancedTools and EntropyCalculator.
    """
    # BASIC: CorpusTools handles optional shuffling internally
    corpus_tools = CorpusTools(list(tokenized), shuffle_tokens=shuffle)
    basic_analysis(corpus_tools)

    # ADVANCED: reuse the tokens in the order CorpusTools produced
    advanced_tools = AdvancedTools(corpus_tools.tokens)
    advanced_analysis(advanced_tools)

    # ENTROPY: also reuse the same ordering
    entropy_calculator = EntropyCalculator(corpus_tools.tokens)
    entropy_metrics(entropy_calculator)

    return advanced_tools


def basic_analysis(corpus_tools: CorpusTools) -> None:
    """Basic corpus statistics (unchanged outputs; clearer structure)."""
    _section("BASIC CORPUS ANALYSIS")

    logger.info("\nGeneral Statistics:")
    logger.info(SUB_TINY)
    median = corpus_tools.find_median_token()
    logger.info(f"• Median Token:         '{median['token']}' (Frequency: {median['frequency']})")
    logger.info(f"• Mean Token Frequency: {corpus_tools.mean_token_frequency():.2f}")
    logger.info(f"• Vocabulary Size:      {len(corpus_tools.vocabulary())}")
    logger.info(f"• Hapax Legomena Size:  {len(corpus_tools.x_legomena(1))}")
    logger.info(f"• Dis Legomena Size:    {len(corpus_tools.x_legomena(2))}")

    logger.info("\nFrequency Analysis:")
    logger.info(SUB_TINY)
    top1 = corpus_tools.query_by_rank(1)
    logger.info(f"• Most Frequent Token:  '{top1['token']}' (Frequency: {top1['frequency']})")
    top_10pct = corpus_tools.cumulative_frequency_analysis(0, 10)
    logger.info(f"• Tokens in top 10%:    {len(top_10pct)}")

    logger.info("\nTop 5 tokens by rank:")
    logger.info(SUB_TINY)
    for tk in corpus_tools.list_tokens_in_rank_range(1, 5):
        logger.info(f"• Rank {tk['rank']:<2}: '{tk['token']}' (Frequency: {tk['frequency']})")

    logger.info("\nExample Token Analysis:")
    logger.info(SUB_TINY)
    example_token = "example"
    try:
        info = corpus_tools.query_by_token(example_token)
        logger.info(f"• Info for '{example_token}': Frequency: {info['frequency']}, Rank: {info['rank']}")
    except ValueError as e:
        logger.warning(str(e))


def advanced_analysis(advanced_tools: AdvancedTools) -> None:
    """Advanced lexical measures + Heaps, Zipf, Zipf–Mandelbrot (unchanged results)."""
    _section("ADVANCED CORPUS ANALYSIS")

    logger.info("\nLexical Diversity Measures:")
    logger.info(SUB)
    logger.info(f"• Yule's K:   {advanced_tools.yules_k():.2f}")
    logger.info(f"• Herdan's C: {advanced_tools.herdans_c():.2f}")

    logger.info("\nHeaps' Law Analysis:")
    logger.info(SUB_SHORT)
    K, beta = advanced_tools.calculate_heaps_law()
    logger.info(f"• K:      {K:.4f}")
    logger.info(f"• Beta:   {beta:.4f}")

    total_tokens = advanced_tools.total_token_count
    est_vocab = advanced_tools.estimate_vocabulary_size(total_tokens)
    actual_vocab = len(advanced_tools.vocabulary())
    diff_pct = abs((actual_vocab - est_vocab) / actual_vocab) * 100 if actual_vocab else 0.0

    logger.info(f"• Estimated Vocabulary: {est_vocab}")
    logger.info(f"• Actual Vocabulary:    {actual_vocab}")
    logger.info(f"• Difference:           {diff_pct:.2f}%")

    logger.info("\nZipf's Law Analysis:")
    logger.info(SUB_SHORT)
    logger.info(f"• Zipf's Law Alpha:     {advanced_tools.calculate_zipf_alpha():.2f}")
    q, s = advanced_tools.calculate_zipf_mandelbrot()
    logger.info(f"• Zipf-Mandelbrot q:    {q:.2f}")
    logger.info(f"• Zipf-Mandelbrot s:    {s:.2f}")


def entropy_metrics(entropy_calculator: EntropyCalculator) -> None:
    """Entropy metrics H0–H3 and redundancy (unchanged results)."""
    _section("ENTROPY METRICS ANALYSIS")

    H0 = entropy_calculator.calculate_H0()
    H1 = entropy_calculator.calculate_H1()
    H2 = entropy_calculator.calculate_H2()
    H3 = entropy_calculator.calculate_H3_kenlm()
    redundancy = entropy_calculator.calculate_redundancy(H3, H0)

    logger.info(f"\n• H0: {H0:.2f} bits")
    logger.info("   - Maximum entropy, uniform probability distribution")
    logger.info(f"\n• H1: {H1:.2f} bits")
    logger.info("   - Based on unigram character frequencies")
    logger.info(f"\n• H2: {H2:.2f} bits")
    logger.info("   - Collision entropy or character pair probabilities")
    logger.info(f"\n• H3: {H3:.2f} bits")
    logger.info(f"   - {entropy_calculator.q_grams}-gram model, captures sub-linguistic patterns")
    logger.info(f"\n• Redundancy: {redundancy:.2f}%")
    logger.info("   - Predictability based on linguistic structure")


def generate_plots(advanced_tools: AdvancedTools, name: str, plots_to_generate: Iterable[str]) -> None:
    """Generate requested plots; preserves mapping and messages."""
    _section("PLOT GENERATION")

    cp = CorpusPlots(advanced_tools, name)
    plot_map = {
        "heaps": cp.plot_heaps_law,
        "zipf": cp.plot_zipfs_law_fit,
        "zipf_mandelbrot": cp.plot_zipf_mandelbrot_fit,
    }

    for plot_id in plots_to_generate:
        func = plot_map.get(plot_id)
        if func:
            func()
            logger.info(f"• Generated {plot_id.replace('_', ' ').title()} plot.")
        else:
            logger.warning(f"Plot type '{plot_id}' not recognized.")


# ------------------------------- Main ----------------------------------- #

def _run_once(name: str) -> None:
    tokens = load_and_tokenize_corpus(name)
    adv = analyze_corpus(tokens)  # shuffle=True (default)
    generate_plots(adv, name, ["heaps", "zipf", "zipf_mandelbrot"])

if __name__ == "__main__":
    if isinstance(corpus_name, str):
        _run_once(corpus_name)
    else:
        for name in corpus_name:
            _run_once(name)