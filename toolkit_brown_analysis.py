import logging
from functools import lru_cache
import nltk
from toolkit_methods import (
    CorpusLoader, CorpusPlots, EntropyCalculator, Tokenizer,
    CorpusTools, AdvancedTools
)

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger('CorpusAnalysis')

# Define corpus name
corpus_name = 'brown' # Change to NLTK corpus name of interest

# list of corpora available in NLTK
# 'brown', 'gutenberg', 'reuters', 'inaugural', 'webtext', 'nps_chat'

# Download required NLTK data
nltk.download(corpus_name, quiet=True)
nltk.download('stopwords', quiet=True)
nltk.download('punkt', quiet=True)

def load_and_tokenize_corpus(corpus_name):
    logger.info(f"Loading and tokenizing {corpus_name}")
    corpus_loader = CorpusLoader(corpus_name)
    corpus_tokens = corpus_loader.load_corpus()

    tokenizer = Tokenizer(remove_punctuation=True) # Change to False to keep punctuation
    tokenized_corpus = tokenizer.tokenize(corpus_tokens, lowercase=True) # Change to False to keep case sensitivity
    logger.info(f"Completed tokenizing {corpus_name}")
    return tokenized_corpus

def analyze_corpus(tokenized_corpus, shuffle=False): # Set shuffle to True for better Heaps' parameter estimation
    corpus_tools = CorpusTools(tokenized_corpus, shuffle_tokens=shuffle)
    basic_analysis(corpus_tools)

    advanced_tools = AdvancedTools(tokenized_corpus)
    advanced_analysis(advanced_tools)

    entropy_calculator = EntropyCalculator(tokenized_corpus)
    entropy_metrics(entropy_calculator)

    return advanced_tools

@lru_cache(maxsize=128)
def basic_analysis(corpus_tools):
    logger.info("Basic Corpus Analysis:")
    
    # Median Token Analysis
    median_token_info = corpus_tools.find_median_token()
    logger.info(f"  Median Token: '{median_token_info['token']}' appears {median_token_info['frequency']} times")
    
    # Mean Token Frequency
    mean_frequency = corpus_tools.mean_token_frequency()
    logger.info(f"  Mean Token Frequency: {mean_frequency:.2f}")
    
    # Vocabulary Size
    vocabulary_size = len(corpus_tools.vocabulary())
    logger.info(f"  Vocabulary Size: {vocabulary_size}")
    
    # Hapax Legomena
    hapax_legomena_count = len(corpus_tools.x_legomena(1))
    logger.info(f"  Hapax Legomena (tokens appearing exactly once): {hapax_legomena_count}")
    
    # Query Specific Token
    example_token = 'example' # Change to a token of interest
    try:
        example_token_info = corpus_tools.query_by_token(example_token)
        logger.info(f"  Info for token '{example_token}': {example_token_info}")
    except ValueError as e:
        logger.warning(f"  {str(e)}")
    
    # Most Frequent Token
    rank_1_token_info = corpus_tools.query_by_rank(1)
    logger.info(f"  Most Frequent Token: '{rank_1_token_info['token']}' with {rank_1_token_info['frequency']} occurrences")
    
    # Cumulative Frequency Analysis for top 10% most frequent tokens
    top_10_percent_tokens = corpus_tools.cumulative_frequency_analysis(0, 10)
    logger.info(f"  Number of tokens in the top 10% of cumulative frequency: {len(top_10_percent_tokens)}")
    
    # Tokens within a specific rank range
    start_rank, end_rank = 1, 5
    tokens_in_rank_range = corpus_tools.list_tokens_in_rank_range(start_rank, end_rank)
    tokens_info_str = "\n".join([
        f"  Rank {token_info['rank']}: '{token_info['token']}' (Frequency: {token_info['frequency']})"
        for token_info in tokens_in_rank_range
    ])
    logger.info(f"Tokens in rank range {start_rank} to {end_rank}:\n{tokens_info_str}")

    # Bi-legomena
    bi_legomena_count = len(corpus_tools.x_legomena(2))
    logger.info(f"  Bi-legomena (types appearing exactly twice): {bi_legomena_count}")

@lru_cache(maxsize=128)
def advanced_analysis(advanced_tools):
    logger.info("Advanced Corpus Analysis:")
    
    # Yule's K Measure for Lexical Diversity
    yules_k_value = advanced_tools.yules_k()
    logger.info(f"  Yule's K (Lexical Diversity): {yules_k_value:.2f}")

    # Herdan's C for Vocabulary Richness
    herdans_c_value = advanced_tools.herdans_c()
    logger.info(f"  Herdan's C (Vocabulary Richness): {herdans_c_value:.2f}")

    # Heaps' K and Beta Parameters
    K, beta = advanced_tools.calculate_heaps_law()
    logger.info(f"  Heaps' Law: K = {K:.4f}, beta = {beta:.4f}")

    # Estimating Vocabulary Size with Heaps' Law
    total_tokens = advanced_tools.total_token_count
    estimated_vocab_size = advanced_tools.estimate_vocabulary_size(total_tokens)
    actual_vocab_size = len(advanced_tools.vocabulary())
    percent_difference = abs((actual_vocab_size - estimated_vocab_size) / actual_vocab_size) * 100
    logger.info(f"  Heaps' Law: Estimated Vocabulary Size: {estimated_vocab_size} (Actual: {actual_vocab_size}, Diff: {percent_difference:.2f}%)")

    # Zipf's Law Alpha Calculation
    zipfs_alpha = advanced_tools.calculate_zipf_alpha()
    logger.info(f"  Zipf's Law Alpha: {zipfs_alpha:.2f}")

    # Zipf-Mandelbrot Law Parameters
    q, s = advanced_tools.calculate_zipf_mandelbrot()
    logger.info(f"  Zipf-Mandelbrot Parameters: q = {q:.2f}, s = {s:.2f}")

@lru_cache(maxsize=128)
def entropy_metrics(entropy_calculator):
    logger.info("Entropy Metrics Analysis:")

    # Zeroth-Order Entropy
    H0 = entropy_calculator.calculate_H0()
    logger.info(f"  H0: {H0:.2f} bits - Maximum entropy, assuming uniform distribution.")

    # First-Order Entropy
    H1 = entropy_calculator.calculate_H1()
    logger.info(f"  H1: {H1:.2f} bits - Based on individual character frequencies.")

    # Second-Order Rényi Entropy
    H2 = entropy_calculator.calculate_H2()
    logger.info(f"  H2: {H2:.2f} bits - Collision entropy, considers character pair probabilities.")

    # Higher-Order Entropy (KenLM)
    H3_kenlm = entropy_calculator.calculate_H3_kenlm()
    logger.info(f"  H3: {H3_kenlm:.2f} bits - {entropy_calculator.q_grams}-gram model, captures linguistic patterns.")

    # Redundancy
    redundancy = entropy_calculator.calculate_redundancy(H3_kenlm, H0)
    logger.info(f"  Redundancy: {redundancy:.2f}% - Predictability based on linguistic structure.")

def generate_plots(advanced_tools, corpus_name, plots_to_generate):
    logger.info("Generating plots...")
    corpus_plots = CorpusPlots(advanced_tools, corpus_name)

    plot_functions = {
        'heaps': corpus_plots.plot_heaps_law,
        'zipf': corpus_plots.plot_zipfs_law_fit,
        'zipf_mandelbrot': corpus_plots.plot_zipf_mandelbrot_fit,
    }

    for plot_type in plots_to_generate:
        plot_function = plot_functions.get(plot_type)
        if plot_function:
            plot_function()
            logger.info(f"Generated {plot_type.replace('_', ' ').title()} plot.")
        else:
            logger.warning(f"Plot type '{plot_type}' not recognized.")

if __name__ == "__main__":
    tokenized_corpus = load_and_tokenize_corpus(corpus_name)
    advanced_tools = analyze_corpus(tokenized_corpus)
    generate_plots(advanced_tools, corpus_name, ["heaps", "zipf", "zipf_mandelbrot"])