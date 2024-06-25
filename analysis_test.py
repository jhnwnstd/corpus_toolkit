from toolkit_methods import (
    CorpusLoader, CorpusPlots, EntropyCalculator, Tokenizer,
    CorpusTools, AdvancedTools
)

# Define corpus name
corpus_name = 'brown' # Change to NLTK corpus name of interest
corpus_loader = CorpusLoader(corpus_name)
corpus_tokens = corpus_loader.load_corpus()
# print the first 10 tokens
print(corpus_tokens[:10])

# Tokenize the corpus
tokenizer = Tokenizer(remove_punctuation=True) # Change to False to keep punctuation
tokenized_corpus = tokenizer.tokenize(corpus_tokens, lowercase=True) # Change to False to keep case sensitivity

# Analyze the corpus
advanced_tools = AdvancedTools(tokenized_corpus)
K, beta = advanced_tools.calculate_heaps_law()
print(f"Heaps' Law parameters for {corpus_name} corpus: K = {K:.4f}, beta = {beta:.4f}")

# Estimating Vocabulary Size with Heaps' Law
total_tokens = advanced_tools.total_token_count
estimated_vocab_size = advanced_tools.estimate_vocabulary_size(total_tokens)
actual_vocab_size = len(advanced_tools.vocabulary())
percent_difference = abs((actual_vocab_size - estimated_vocab_size) / actual_vocab_size) * 100
print(f"Estimated Vocabulary Size: {estimated_vocab_size}")
print(f"Actual Vocabulary Size: {actual_vocab_size}")
print(f"Percent Difference: {percent_difference:.2f}%")