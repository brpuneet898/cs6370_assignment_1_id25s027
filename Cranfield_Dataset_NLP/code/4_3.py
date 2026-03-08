import os
import json
from collections import Counter
from sentenceSegmentation import SentenceSegmentation
from tokenization import Tokenization
from stopwordRemoval import StopwordRemoval
import nltk
from nltk.corpus import stopwords

nltk.download('stopwords')

## this file is createad for question 4 part 3. It identifies stopwords based on the corpus and compares them with 
## NLTK's stopword list.
def main():
    ## Set up paths
    base_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(base_dir)

    dataset_path = os.path.join(project_root, "cranfield_dataset", "cran_docs.json")
    output_dir = os.path.join(project_root, "output_theory", "4_3")

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    with open(dataset_path, "r") as f:
        docs_json = json.load(f)
    ## Extract document bodies
    docs = [doc["body"] for doc in docs_json]

    segmenter = SentenceSegmentation()
    tokenizer = Tokenization()

    segmented_docs = []
    tokenized_docs = []

    for doc in docs:
        sentences = segmenter.punkt(doc)
        segmented_docs.append(sentences)

        tokens = tokenizer.pennTreeBank(sentences)
        tokenized_docs.append(tokens)
    ## Calculate document frequency for each word
    doc_frequency = Counter()
    total_docs = len(tokenized_docs)

    for doc in tokenized_docs:

        unique_words = set()

        for sentence in doc:
            for word in sentence:
                unique_words.add(word.lower())

        for word in unique_words:
            doc_frequency[word] += 1

    threshold = 0.7 * total_docs

    corpus_stopwords = set()

    for word, freq in doc_frequency.items():
        if freq >= threshold:
            corpus_stopwords.add(word)

    nltk_stopwords = set(stopwords.words("english"))

    overlap = corpus_stopwords.intersection(nltk_stopwords)

    only_corpus = corpus_stopwords - nltk_stopwords
    only_nltk = nltk_stopwords - corpus_stopwords

    with open(os.path.join(output_dir, "corpus_stopwords.txt"), "w") as f:
        for w in sorted(corpus_stopwords):
            f.write(w + "\n")

    with open(os.path.join(output_dir, "nltk_stopwords.txt"), "w") as f:
        for w in sorted(nltk_stopwords):
            f.write(w + "\n")

    with open(os.path.join(output_dir, "overlap_stopwords.txt"), "w") as f:
        for w in sorted(overlap):
            f.write(w + "\n")

    print("\nSTOPWORD COMPARISON RESULTS")
    print("="*50)

    print("\n1. Number of stopwords identified")
    print("Corpus-derived stopwords:", len(corpus_stopwords))
    print("NLTK stopwords:", len(nltk_stopwords))

    print("\n2. Overlap between the two lists")
    print("Common stopwords:", len(overlap))

    print("\n3. Words appearing only in corpus-derived list")
    print(list(sorted(only_corpus))[:20])

    print("\nWords appearing only in NLTK list")
    print(list(sorted(only_nltk))[:20])

    print("\nFull lists saved in:", output_dir)


if __name__ == "__main__":
    main()