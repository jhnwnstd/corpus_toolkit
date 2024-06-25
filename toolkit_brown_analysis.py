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
    logger.info("\n" + "=" * 50)
    logger.info("BASIC CORPUS ANALYSIS".center(50))
    logger.info("=" * 50)
    
    logger.info("\nGeneral Statistics:")
    logger.info("-" * 20)
    median_token_info = corpus_tools.find_median_token()
    logger.info(f"Median Token:        '{median_token_info['token']}' (Frequency: {median_token_info['frequency']})")
    logger.info(f"Mean Token Frequency: {corpus_tools.mean_token_frequency():.2f}")
    logger.info(f"Vocabulary Size:      {len(corpus_tools.vocabulary())}")
    logger.info(f"Hapax Legomena Size:  {len(corpus_tools.x_legomena(1))}")
    logger.info(f"Dis Legomena Size:    {len(corpus_tools.x_legomena(2))}") # Change to any value to get which words occur that many times
    
    logger.info("\nFrequency Analysis:")
    logger.info("-" * 20)
    rank_1_token_info = corpus_tools.query_by_rank(1)
    logger.info(f"Most Frequent Token:  '{rank_1_token_info['token']}' (Frequency: {rank_1_token_info['frequency']})")
    top_10_percent_tokens = corpus_tools.cumulative_frequency_analysis(0, 10)
    logger.info(f"Tokens in top 10%:     {len(top_10_percent_tokens)}")
    
    logger.info("\nTop 5 tokens by rank:")
    logger.info("-" * 20)
    for token_info in corpus_tools.list_tokens_in_rank_range(1, 5):
        logger.info(f"Rank {token_info['rank']:<2}: '{token_info['token']}' (Frequency: {token_info['frequency']})")

    logger.info("\nExample Token Analysis:")
    logger.info("-" * 20)
    example_token = 'example'
    try:
        example_token_info = corpus_tools.query_by_token(example_token)
        logger.info(f"Info for '{example_token}': Frequency: {example_token_info['frequency']}, Rank: {example_token_info['rank']}")
    except ValueError as e:
        logger.warning(f"{str(e)}")

@lru_cache(maxsize=128)
def advanced_analysis(advanced_tools):
    logger.info("\n" + "=" * 50)
    logger.info("ADVANCED CORPUS ANALYSIS".center(50))
    logger.info("=" * 50)
    
    logger.info("\nLexical Diversity Measures:")
    logger.info("-" * 28)
    logger.info(f"Yule's K:  {advanced_tools.yules_k():.2f}")
    logger.info(f"Herdan's C: {advanced_tools.herdans_c():.2f}")

    logger.info("\nHeaps' Law Analysis:")
    logger.info("-" * 22)
    K, beta = advanced_tools.calculate_heaps_law()
    logger.info(f"K:      {K:.4f}")
    logger.info(f"Beta:   {beta:.4f}")
    total_tokens = advanced_tools.total_token_count
    estimated_vocab_size = advanced_tools.estimate_vocabulary_size(total_tokens)
    actual_vocab_size = len(advanced_tools.vocabulary())
    percent_difference = abs((actual_vocab_size - estimated_vocab_size) / actual_vocab_size) * 100
    logger.info(f"Estimated Vocabulary: {estimated_vocab_size}")
    logger.info(f"Actual Vocabulary:    {actual_vocab_size}")
    logger.info(f"Difference:           {percent_difference:.2f}%")

    logger.info("\nZipf's Law Analysis:")
    logger.info("-" * 22)
    logger.info(f"Zipf's Law Alpha:     {advanced_tools.calculate_zipf_alpha():.2f}")
    q, s = advanced_tools.calculate_zipf_mandelbrot()
    logger.info(f"Zipf-Mandelbrot q:    {q:.2f}")
    logger.info(f"Zipf-Mandelbrot s:    {s:.2f}")

@lru_cache(maxsize=128)
def entropy_metrics(entropy_calculator):
    logger.info("\n" + "=" * 50)
    logger.info("ENTROPY METRICS ANALYSIS".center(50))
    logger.info("=" * 50)

    H0 = entropy_calculator.calculate_H0()
    H1 = entropy_calculator.calculate_H1()
    H2 = entropy_calculator.calculate_H2()
    H3_kenlm = entropy_calculator.calculate_H3_kenlm()
    redundancy = entropy_calculator.calculate_redundancy(H3_kenlm, H0)

    logger.info(f"\nH0: {H0:.2f} bits")
    logger.info("   Maximum entropy, uniform probablity distribution")
    logger.info(f"\nH1: {H1:.2f} bits")
    logger.info("   Based on unigram character frequencies")
    logger.info(f"\nH2: {H2:.2f} bits")
    logger.info("   Collision entropy or character pair probabilities")
    logger.info(f"\nH3: {H3_kenlm:.2f} bits")
    logger.info(f"   {entropy_calculator.q_grams}-gram model, captures sub-linguistic patterns")
    logger.info(f"\nRedundancy: {redundancy:.2f}%")
    logger.info("   Predictability based on linguistic structure")

def generate_plots(advanced_tools, corpus_name, plots_to_generate):
    logger.info("\n" + "=" * 50)
    logger.info("PLOT GENERATION".center(50))
    logger.info("=" * 50)

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