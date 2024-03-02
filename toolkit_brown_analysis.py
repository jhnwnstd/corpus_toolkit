import logging
from pathlib import Path
from toolkit_methods import CorpusLoader, CorpusPlots, EntropyCalculator, Tokenizer, CorpusTools, AdvancedTools

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(message)s')
logger = logging.getLogger('CorpusAnalysis')

def load_and_tokenize_corpus(corpus_name):
    logger.info(f"Loading and tokenizing {corpus_name}")
    corpus_loader = CorpusLoader(corpus_name)
    corpus_tokens = corpus_loader.load_corpus()

    tokenizer = Tokenizer(remove_punctuation=True, remove_stopwords=False)
    tokenized_corpus = tokenizer.tokenize(' '.join(corpus_tokens), lowercase=True)
    logger.info(f"Completed tokenizing {corpus_name}")
    return tokenized_corpus

def analyze_corpus(corpus_name, tokenized_corpus, shuffle=False):
    # First, utilize CorpusTools for basic corpus analysis
    corpus_tools = CorpusTools(tokenized_corpus, shuffle_tokens=shuffle)
    basic_analysis(corpus_tools)

    # Then, utilize AdvancedTools for advanced corpus analysis
    advanced_tools = AdvancedTools(tokenized_corpus)
    advanced_analysis(advanced_tools)

    # Entropy metrics analysis
    entropy_calculator = EntropyCalculator(tokenized_corpus)
    entropy_metrics(entropy_calculator)

    return advanced_tools

def basic_analysis(corpus_tools):
    logger.info("Basic Corpus Analysis:")
    median_token = corpus_tools.find_median_token()
    logger.info(f"  Median Token: {median_token['token']} with frequency {median_token['frequency']}")
    logger.info(f"  Mean Token Frequency: {corpus_tools.mean_token_frequency():.2f}")
    logger.info(f"  Vocabulary Size: {len(corpus_tools.vocabulary())}")

def advanced_analysis(advanced_tools):
    logger.info("Advanced Analysis:")
    logger.info(f"  Yule's K: {advanced_tools.yules_k():.2f}")
    logger.info(f"  Herdan's C: {advanced_tools.herdans_c():.2f}")
    actual_vocab_size = len(advanced_tools.vocabulary())
    estimated_vocab_size = advanced_tools.estimate_vocabulary_size(len(tokenized_corpus))
    percent_difference = ((actual_vocab_size - estimated_vocab_size) / actual_vocab_size) * 100
    logger.info(f"  Heaps' Law Analysis: Actual vs. Estimated Vocabulary Size: {actual_vocab_size} vs. {estimated_vocab_size} ({percent_difference:.2f}%)")

def entropy_metrics(entropy_calculator):
    logger.info("Entropy Metrics:")
    logger.info(f"  H0: {entropy_calculator.calculate_H0():.2f}")
    logger.info(f"  H1: {entropy_calculator.calculate_H1():.2f}")
    H3_kenlm = entropy_calculator.calculate_H3_kenlm()
    logger.info(f"  H3 (KenLM): {H3_kenlm:.2f}")
    redundancy = entropy_calculator.calculate_redundancy(H3_kenlm, entropy_calculator.calculate_H0())
    logger.info(f"  Redundancy: {redundancy:.2f}%")

def generate_plots(advanced_tools, corpus_name, plots_to_generate):
    logger.info("Generating plots...")
    plots_dir = Path("plots")
    plots_dir.mkdir(exist_ok=True)
    corpus_plots = CorpusPlots(advanced_tools, corpus_name, plots_dir)

    for plot_type in plots_to_generate:
        if plot_type == 'heaps':
            corpus_plots.plot_heaps_law()
            logger.info("Generated Heaps' Law plot.")
        elif plot_type == 'zipf':
            corpus_plots.plot_zipfs_law_fit()
            logger.info("Generated Zipf's Law plot.")
        elif plot_type == 'zipf_mandelbrot':
            corpus_plots.plot_zipf_mandelbrot_fit()
            logger.info("Generated Zipf-Mandelbrot plot.")

if __name__ == "__main__":
    corpus_name = "brown"
    tokenized_corpus = load_and_tokenize_corpus(corpus_name)
    advanced_tools = analyze_corpus(corpus_name, tokenized_corpus, shuffle=True)
    plots_to_generate = ["heaps", "zipf", "zipf_mandelbrot"]
    generate_plots(advanced_tools, corpus_name, plots_to_generate)
