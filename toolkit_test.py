import logging
from pathlib import Path

from corpus_toolkit import AdvancedTools, CorpusLoader, CorpusPlots, Tokenizer

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)

def load_and_tokenize_corpus(corpus_name):
    logger.info(f"\nLoading and tokenizing {corpus_name}...")
    corpus_loader = CorpusLoader(corpus_name)
    corpus_tokens = corpus_loader.load_corpus()
    tokenizer = Tokenizer(remove_punctuation=True, remove_stopwords=False)
    tokenized_corpus = tokenizer.tokenize(' '.join(corpus_tokens), lowercase=True)
    return tokenized_corpus

def perform_advanced_analysis(corpus_name, plots_to_generate):
    tokenized_corpus = load_and_tokenize_corpus(corpus_name)
    advanced_analyzer = AdvancedTools(tokenized_corpus)
    
    plots_dir = Path('plots')
    plots_dir.mkdir(exist_ok=True)
    
    plotter = CorpusPlots(advanced_analyzer, corpus_name, plots_dir=plots_dir)

    results = []
    for plot_type in plots_to_generate:
        if plot_type == "zipf":
            alpha = advanced_analyzer.calculate_zipf_alpha()
            plotter.plot_zipfs_law_fit()
            results.append(f"Zipf's Law Alpha for {corpus_name}: {alpha:.3f}")
        
        if plot_type == "heaps":
            k, beta = advanced_analyzer.calculate_heaps_law()
            plotter.plot_heaps_law()
            estimated_vocab_size = advanced_analyzer.estimate_vocabulary_size(advanced_analyzer.total_token_count)
            actual_vocab_size = len(advanced_analyzer.vocabulary())
            results.append(f"Heaps' Law Parameters: K={k:.2f}, Beta={beta:.3f}")
            results.append(f"Estimated Vocabulary Size: {estimated_vocab_size}")
            results.append(f"Actual Vocabulary Size: {actual_vocab_size}")
            difference = abs(estimated_vocab_size - actual_vocab_size)
            percentage_error = (difference / actual_vocab_size) * 100
            results.append(f"Difference: {difference} ({percentage_error:.2f}%)")
        
        if plot_type == "zipf_mandelbrot":
            q, s = advanced_analyzer.calculate_zipf_mandelbrot()
            plotter.plot_zipf_mandelbrot_fit()
            results.append(f"Zipf-Mandelbrot Parameters for {corpus_name}: q={q:.2f}, s={s:.3f}")
            
    log_results(corpus_name, results)

def log_results(corpus_name, results):
    logger.info(f"\nAnalysis Summary for '{corpus_name}':")
    logger.info(f"{'=' * 50}")
    for result in results:
        logger.info(result)
    logger.info(f"{'=' * 50}\n")

# Example usage
corpora = ['brown', 'gutenberg', 'inaugural', 'reuters', 'webtext']
plots_to_generate = ["zipf", "heaps", "zipf_mandelbrot"]

for corpus in corpora:
    perform_advanced_analysis(corpus, plots_to_generate)
