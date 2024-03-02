import logging
from pathlib import Path
from toolkit_methods import AdvancedTools, CorpusLoader, CorpusPlots, EntropyCalculator, Tokenizer, CorpusTools

# Configure logging with a concise format.
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger('CorpusAnalysis')

def load_and_tokenize_corpus(corpus_name):
    logger.info(f"Loading and tokenizing {corpus_name}")
    corpus_loader = CorpusLoader(corpus_name)
    corpus_tokens = corpus_loader.load_corpus()

    tokenizer = Tokenizer(remove_punctuation=True, remove_stopwords=False)
    return tokenizer.tokenize(' '.join(corpus_tokens), lowercase=True)

def analyze_corpus(corpus_name, tokenized_corpus, shuffle=False): # Shuffle tokens for best heap's fit
    advanced_tools = AdvancedTools(tokenized_corpus, shuffle_tokens=shuffle)
    entropy_calculator = EntropyCalculator(tokenized_corpus)

    # Analysis logging
    logger.info("CorpusTools Analysis:")
    logger.info(f"  Median Token: {advanced_tools.find_median_token()['token']}")
    logger.info(f"  Mean Token Frequency: {advanced_tools.mean_token_frequency():.2f}")
    logger.info(f"  Token 'example' Info: {advanced_tools.query_by_token('example')}")
    logger.info(f"  Rank 1 Token: {advanced_tools.query_by_rank(1)}")
    logger.info(f"  Vocabulary Size: {len(advanced_tools.vocabulary())}")

    logger.info("Advanced Analysis:")
    logger.info(f"  Yule's K: {advanced_tools.yules_k():.2f}")
    logger.info(f"  Herdan's C: {advanced_tools.herdans_c():.2f}")

    logger.info("Entropy Metrics:")
    logger.info(f"  H0: {entropy_calculator.calculate_H0():.2f}")
    logger.info(f"  H1: {entropy_calculator.calculate_H1():.2f}")
    logger.info(f"  H3 (KenLM): {entropy_calculator.calculate_H3_kenlm():.2f}")
    logger.info(f"  Redundancy: {entropy_calculator.calculate_redundancy(entropy_calculator.calculate_H3_kenlm(), entropy_calculator.calculate_H0()):.2f}%")

    actual_vocab_size = len(advanced_tools.vocabulary())
    estimated_vocab_size = advanced_tools.estimate_vocabulary_size(len(tokenized_corpus))
    percent_difference = ((actual_vocab_size - estimated_vocab_size) / actual_vocab_size) * 100
    logger.info("Heaps' Law Analysis:")
    logger.info(f"  Actual vs. Estimated Vocabulary Size: {actual_vocab_size} vs. {estimated_vocab_size} ({percent_difference:.2f}%)")

    return advanced_tools

def generate_plots(advanced_tools, corpus_name, plots_to_generate):
    logger.info("Generating plots...")
    plots_dir = Path("plots")
    plots_dir.mkdir(exist_ok=True)
    corpus_plots = CorpusPlots(advanced_tools, corpus_name, plots_dir)

    if 'heaps' in plots_to_generate:
        corpus_plots.plot_heaps_law()
        logger.info("Generated Heaps' Law plot.")
    if 'zipf' in plots_to_generate:
        corpus_plots.plot_zipfs_law_fit()
        logger.info("Generated Zipf's Law plot.")
    if 'zipf_mandelbrot' in plots_to_generate:
        corpus_plots.plot_zipf_mandelbrot_fit()
        logger.info("Generated Zipf-Mandelbrot plot.")

if __name__ == "__main__":
    corpus_name = "brown"
    tokenized_corpus = load_and_tokenize_corpus(corpus_name)
    advanced_tools = analyze_corpus(corpus_name, tokenized_corpus, shuffle=True)  # Shuffle tokens if desired
    
    # Define which plots to generate - can be "heaps", "zipf", "zipf_mandelbrot"
    plots_to_generate = ["heaps", "zipf", "zipf_mandelbrot"]  # Example: generate all plots
    generate_plots(advanced_tools, corpus_name, plots_to_generate)
