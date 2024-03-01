import logging
from pathlib import Path
from toolkit_methods import AdvancedTools, CorpusLoader, CorpusPlots, EntropyCalculator, Tokenizer, CorpusTools

# Setup logging with detailed format including timestamp and logging level.
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(message)s')
logger = logging.getLogger(__name__)

# Function to load and tokenize a corpus specified by its name.
def load_and_tokenize_corpus(corpus_name):
    # Start of the loading process
    logger.info(f"Loading {corpus_name}")
    # Initializing the corpus loader with the given corpus name
    corpus_loader = CorpusLoader(corpus_name)
    # Loading the corpus using the corpus loader
    corpus_tokens = corpus_loader.load_corpus()
    
    # Start of the tokenizing process
    logger.info(f"Tokenizing {corpus_name}")
    # Initializing the tokenizer with configurations to remove punctuation and not to remove stopwords
    tokenizer = Tokenizer(remove_punctuation=True, remove_stopwords=False)
    # Tokenizing the loaded corpus
    tokenized_corpus = tokenizer.tokenize(' '.join(corpus_tokens), lowercase=True)
    
    # Completion of the tokenizing process
    logger.info(f"Completed tokenizing {corpus_name}")
    return tokenized_corpus

# Main function to perform analysis on a given corpus including plots generation.
def perform_analysis(corpus_name, plots_to_generate):
    # Loading and tokenizing the corpus
    tokenized_corpus = load_and_tokenize_corpus(corpus_name)
    # Demonstrating the use of CorpusTools on the tokenized corpus
    corpus_tools(tokenized_corpus)
    
    # Initializing the advanced analyzer with the tokenized corpus
    advanced_analyzer = AdvancedTools(tokenized_corpus)
    # Initializing the entropy calculator with the tokenized corpus
    entropy_calculator = EntropyCalculator(tokenized_corpus)
    
    # Preparing the directory to save plots
    plots_dir = Path("plots")     # Technically redundant since this maintained by the toolkit_methods.py
    plots_dir.mkdir(exist_ok=True)
    # Initializing the plotter with the advanced analyzer and specified corpus name
    plotter = CorpusPlots(advanced_analyzer, corpus_name, plots_dir=plots_dir)
    
    # Analyzing the corpus to generate results based on specified plots to generate
    results = analyze_corpus(advanced_analyzer, entropy_calculator, plotter, plots_to_generate, tokenized_corpus)
    
    # Logging the results of the analysis
    log_results(corpus_name, results)

# Analyze the corpus and generate specified plots and calculations.
def analyze_corpus(advanced_analyzer, entropy_calculator, plotter, plots_to_generate, tokenized_corpus):
    results = []
    # Calculates the alpha parameter and generate the plot.
    if "zipf" in plots_to_generate:
        alpha = advanced_analyzer.calculate_zipf_alpha()
        plotter.plot_zipfs_law_fit()
        results.append(f"Zipf's Law Alpha: {alpha:.3f}")
    # Calculates Heaps' Law parameters, estimate vocabulary size, and generate the plot.
    if "heaps" in plots_to_generate:
        k, beta = advanced_analyzer.calculate_heaps_law()
        plotter.plot_heaps_law()
        estimated_vocab_size = advanced_analyzer.estimate_vocabulary_size(len(tokenized_corpus))
        actual_vocab_size = len(advanced_analyzer.vocabulary())
        difference = abs(estimated_vocab_size - actual_vocab_size)
        percentage_error = (difference / actual_vocab_size) * 100
        results.append(f"Heaps' Law Parameters: K={k:.2f}, Beta={beta:.3f}")
        results.append(f"Estimated Vocabulary Size: {estimated_vocab_size}, Actual Vocabulary Size: {actual_vocab_size}")
        results.append(f"Vocabulary Size Difference: {difference} ({percentage_error:.2f}%)")
    # Calculates Zipf-Mandelbrot parameters and generate the plot.
    if "zipf_mandelbrot" in plots_to_generate:
        q, s = advanced_analyzer.calculate_zipf_mandelbrot()
        plotter.plot_zipf_mandelbrot_fit()
        results.append(f"Zipf-Mandelbrot Parameters: q={q:.2f}, s={s:.3f}")
    # Calculates entropy measures and text redundancy.
    if "entropy" in plots_to_generate:
        H0 = entropy_calculator.calculate_H0()
        H1 = entropy_calculator.calculate_H1()
        H3 = entropy_calculator.calculate_H3_kenlm()
        redundancy = entropy_calculator.calculate_redundancy(H3, H0)
        results.extend([
            f"Zero-order Entropy (H0): {H0:.3f}",
            f"First-order Entropy (H1): {H1:.3f}",
            f"Third-order Entropy (H3) using KenLM: {H3:.3f}",
            f"Text Redundancy: {redundancy:.2f}%"
        ])
    return results

# Log the summary of the analysis.
def log_results(corpus_name, results):
    logger.info(f"Analysis Summary for '{corpus_name}':")
    for result in results:
        logger.info(result)

# Function to demonstrate the use of CorpusTools.
def corpus_tools(tokenized_corpus):
    logger.info("CorpusTools Analysis:")
    corpus_analyzer = CorpusTools(tokenized_corpus, shuffle_tokens=True) # Shuffle tokens for smoother Heaps' Law plot fit.
    
    median_token_info = corpus_analyzer.find_median_token()
    logger.info(f"  Median Token: {median_token_info['token']} (Frequency: {median_token_info['frequency']})")
    
    mean_frequency = corpus_analyzer.mean_token_frequency()
    logger.info(f"  Mean Token Frequency: {mean_frequency:.2f}")
    
    sample_token = "example"  # Assumes 'example' is present in the corpus.
    token_query = corpus_analyzer.query_by_token(sample_token)
    logger.info(f"  Token '{sample_token}' Info: Frequency: {token_query['frequency']}, Rank: {token_query['rank']}")
    
    sample_rank = 1  # Query by the most frequent token.
    rank_query = corpus_analyzer.query_by_rank(sample_rank)
    logger.info(f"  Rank {sample_rank} Token: '{rank_query['token']}' (Frequency: {rank_query['frequency']})")
    
    vocab_size = len(corpus_analyzer.vocabulary())
    logger.info(f"  Vocabulary Size: {vocab_size}")

# Main execution to perform analysis on specified corpora.
corpora = ['brown']
plots_to_generate = ["zipf", "heaps", "zipf_mandelbrot", "entropy"]

for corpus in corpora:
    perform_analysis(corpus, plots_to_generate)
