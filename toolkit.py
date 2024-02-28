import logging
from pathlib import Path
from toolkit_methods import AdvancedTools, CorpusLoader, CorpusPlots, EntropyCalculator, Tokenizer, CorpusTools

# Setup enhanced logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(message)s')
logger = logging.getLogger(__name__)

def load_and_tokenize_corpus(corpus_name):
    logger.info(f"Loading {corpus_name}")
    corpus_loader = CorpusLoader(corpus_name)
    corpus_tokens = corpus_loader.load_corpus()
    
    logger.info(f"Tokenizing {corpus_name}")
    tokenizer = Tokenizer(remove_punctuation=True, remove_stopwords=False)
    tokenized_corpus = tokenizer.tokenize(' '.join(corpus_tokens), lowercase=True)
    
    logger.info(f"Completed tokenizing {corpus_name}")
    return tokenized_corpus

def corpus_tools(tokenized_corpus):
    logger.info("CorpusTools Analysis:")
    corpus_analyzer = CorpusTools(tokenized_corpus, shuffle_tokens=True)
    
    median_token_info = corpus_analyzer.find_median_token()
    logger.info(f"  Median Token: {median_token_info['token']} (Frequency: {median_token_info['frequency']})")
    
    mean_frequency = corpus_analyzer.mean_token_frequency()
    logger.info(f"  Mean Token Frequency: {mean_frequency:.2f}")
    
    sample_token = "example"  # Assuming 'example' is in the corpus
    token_query = corpus_analyzer.query_by_token(sample_token)
    logger.info(f"  Token '{sample_token}' Info: Frequency: {token_query['frequency']}, Rank: {token_query['rank']}")
    
    sample_rank = 1  # Most frequent token
    rank_query = corpus_analyzer.query_by_rank(sample_rank)
    logger.info(f"  Rank {sample_rank} Token: '{rank_query['token']}' (Frequency: {rank_query['frequency']})")
    
    vocab_size = len(corpus_analyzer.vocabulary())
    logger.info(f"  Vocabulary Size: {vocab_size}")

def perform_analysis(corpus_name, plots_to_generate):
    tokenized_corpus = load_and_tokenize_corpus(corpus_name)
    corpus_tools(tokenized_corpus)
    
    advanced_analyzer = AdvancedTools(tokenized_corpus)
    entropy_calculator = EntropyCalculator(tokenized_corpus)
    
    plotter = CorpusPlots(advanced_analyzer, corpus_name, plots_dir=Path("plots"))
    
    results = analyze_corpus(advanced_analyzer, entropy_calculator, plotter, plots_to_generate)
    log_results(corpus_name, results)

def analyze_corpus(advanced_analyzer, entropy_calculator, plotter, plots_to_generate):
    results = []
    if "zipf" in plots_to_generate:
        alpha = advanced_analyzer.calculate_zipf_alpha()
        plotter.plot_zipfs_law_fit()
        results.append(f"Zipf's Law Alpha: {alpha:.3f}")
    if "heaps" in plots_to_generate:
        k, beta = advanced_analyzer.calculate_heaps_law()
        plotter.plot_heaps_law()
        results.append(f"Heaps' Law Parameters: K={k:.2f}, Beta={beta:.3f}")
    if "zipf_mandelbrot" in plots_to_generate:
        q, s = advanced_analyzer.calculate_zipf_mandelbrot()
        plotter.plot_zipf_mandelbrot_fit()
        results.append(f"Zipf-Mandelbrot Parameters: q={q:.2f}, s={s:.3f}")
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

def log_results(corpus_name, results):
    logger.info(f"Analysis Summary for '{corpus_name}':")
    for result in results:
        logger.info(result)

# Example usage
corpora = ['brown']
plots_to_generate = ["zipf", "heaps", "zipf_mandelbrot", "entropy"]

for corpus in corpora:
    perform_analysis(corpus, plots_to_generate)