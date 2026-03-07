import os
import json
from collections import Counter
import nltk
from nltk.corpus import wordnet
from nltk.stem import WordNetLemmatizer
from sentenceSegmentation import SentenceSegmentation
from tokenization import Tokenization
from inflectionReduction import InflectionReduction

def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)

def safe_nltk_downloads():
    resources = [
        "punkt",
        "wordnet",
        "omw-1.4",
        "averaged_perceptron_tagger",
        "averaged_perceptron_tagger_eng",
    ]

    for resource in resources:
        try:
            nltk.download(resource, quiet=True)
        except:
            pass


def get_wordnet_pos(treebank_tag):
    if treebank_tag.startswith("J"):
        return wordnet.ADJ
    if treebank_tag.startswith("V"):
        return wordnet.VERB
    if treebank_tag.startswith("N"):
        return wordnet.NOUN
    if treebank_tag.startswith("R"):
        return wordnet.ADV
    return wordnet.NOUN


def flatten_tokenized_docs(tokenized_docs):
    flat_tokens = []
    for doc in tokenized_docs:
        for sent in doc:
            for token in sent:
                flat_tokens.append(token)
    return flat_tokens


def lowercase_alpha_tokens(tokens):
    cleaned = []
    for token in tokens:
        if token.isalpha():
            cleaned.append(token.lower())
    return cleaned


def pos_aware_lemmatize_docs(tokenized_docs):
    lemmatizer = WordNetLemmatizer()
    lemmatized_docs = []

    for doc in tokenized_docs:
        lemmatized_doc = []
        for sent in doc:
            tagged_sent = nltk.pos_tag(sent)
            lemmatized_sent = []
            for word, tag in tagged_sent:
                wn_pos = get_wordnet_pos(tag)
                lemma = lemmatizer.lemmatize(word, pos=wn_pos)
                lemmatized_sent.append(lemma)
            lemmatized_doc.append(lemmatized_sent)
        lemmatized_docs.append(lemmatized_doc)

    return lemmatized_docs


def compute_vocab(tokens):
    return set(tokens)


def token_frequency(tokens):
    return Counter(tokens)


def is_good_example_token(token):
    return token.isalpha() and len(token) >= 4


def find_overstemming_examples(original_tokens, stemmed_tokens, lemmatized_tokens, max_examples=12):
    examples = []
    seen = set()

    for orig, stem, lemma in zip(original_tokens, stemmed_tokens, lemmatized_tokens):
        o = orig.lower()
        s = stem.lower()
        l = lemma.lower()

        if not is_good_example_token(o):
            continue
        if o == s:
            continue
        if s == l:
            continue

        stem_valid = len(wordnet.synsets(s)) > 0
        lemma_valid = len(wordnet.synsets(l)) > 0

        if (not stem_valid) and lemma_valid:
            key = (o, s, l)
            if key not in seen:
                seen.add(key)
                examples.append({
                    "original": o,
                    "stemmed": s,
                    "lemmatized": l
                })

        if len(examples) >= max_examples:
            break

    return examples


def find_semantic_preservation_examples(original_tokens, stemmed_tokens, lemmatized_tokens, max_examples=12):
    examples = []
    seen = set()

    for orig, stem, lemma in zip(original_tokens, stemmed_tokens, lemmatized_tokens):
        o = orig.lower()
        s = stem.lower()
        l = lemma.lower()

        if not is_good_example_token(o):
            continue
        if o == l:
            if o == s:
                continue

        stem_valid = len(wordnet.synsets(s)) > 0
        lemma_valid = len(wordnet.synsets(l)) > 0

        if lemma_valid and ((not stem_valid) or (len(s) < len(l))):
            key = (o, s, l)
            if key not in seen:
                seen.add(key)
                examples.append({
                    "original": o,
                    "stemmed": s,
                    "lemmatized": l
                })

        if len(examples) >= max_examples:
            break

    return examples


def main():
    safe_nltk_downloads()

    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(current_dir)

    dataset_path = os.path.join(project_root, "cranfield_dataset", "cran_docs.json")
    output_dir = os.path.join(project_root, "output_theory", "3_2")

    ensure_dir(output_dir)

    with open(dataset_path, "r", encoding="utf-8") as f:
        docs_json = json.load(f)

    doc_ids = []
    docs = []
    for item in docs_json:
        doc_ids.append(item.get("id"))
        docs.append(item.get("body", ""))

    sentence_segmenter = SentenceSegmentation()
    segmented_docs = []
    for doc in docs:
        segmented_docs.append(sentence_segmenter.punkt(doc))

    tokenizer = Tokenization()
    tokenized_docs = []
    for sent_list in segmented_docs:
        tokenized_docs.append(tokenizer.pennTreeBank(sent_list))

    inflection_reducer = InflectionReduction()
    stemmed_docs = []
    for doc in tokenized_docs:
        stemmed_docs.append(inflection_reducer.porterStemmer(doc))

    lemmatized_docs = pos_aware_lemmatize_docs(tokenized_docs)

    tokenized_output = []
    stemmed_output = []
    lemmatized_output = []

    for i in range(len(doc_ids)):
        tokenized_output.append({
            "doc_id": doc_ids[i],
            "tokenized_sentences": tokenized_docs[i]
        })
        stemmed_output.append({
            "doc_id": doc_ids[i],
            "stemmed_sentences": stemmed_docs[i]
        })
        lemmatized_output.append({
            "doc_id": doc_ids[i],
            "lemmatized_sentences": lemmatized_docs[i]
        })

    with open(os.path.join(output_dir, "tokenized_docs_ptb.json"), "w", encoding="utf-8") as f:
        json.dump(tokenized_output, f, indent=2, ensure_ascii=False)

    with open(os.path.join(output_dir, "stemmed_docs_porter.json"), "w", encoding="utf-8") as f:
        json.dump(stemmed_output, f, indent=2, ensure_ascii=False)

    with open(os.path.join(output_dir, "lemmatized_docs_wordnet.json"), "w", encoding="utf-8") as f:
        json.dump(lemmatized_output, f, indent=2, ensure_ascii=False)

    original_tokens = flatten_tokenized_docs(tokenized_docs)
    stemmed_tokens = flatten_tokenized_docs(stemmed_docs)
    lemmatized_tokens = flatten_tokenized_docs(lemmatized_docs)

    original_vocab_tokens = lowercase_alpha_tokens(original_tokens)
    stemmed_vocab_tokens = lowercase_alpha_tokens(stemmed_tokens)
    lemmatized_vocab_tokens = lowercase_alpha_tokens(lemmatized_tokens)

    original_vocab = compute_vocab(original_vocab_tokens)
    stemmed_vocab = compute_vocab(stemmed_vocab_tokens)
    lemmatized_vocab = compute_vocab(lemmatized_vocab_tokens)

    original_freq = token_frequency(original_vocab_tokens)
    stemmed_freq = token_frequency(stemmed_vocab_tokens)
    lemmatized_freq = token_frequency(lemmatized_vocab_tokens)

    overstemming_examples = find_overstemming_examples(
        original_tokens, stemmed_tokens, lemmatized_tokens, max_examples=10
    )

    semantic_examples = find_semantic_preservation_examples(
        original_tokens, stemmed_tokens, lemmatized_tokens, max_examples=10
    )

    summary = {
        "number_of_documents": len(docs),
        "pipeline_used": {
            "sentence_segmentation": "NLTK Punkt",
            "word_tokenization": "NLTK Penn Treebank",
            "stemming": "Porter Stemmer",
            "lemmatization": "WordNet Lemmatizer (POS-aware)"
        },
        "token_statistics": {
            "total_original_tokens": len(original_tokens),
            "total_stemmed_tokens": len(stemmed_tokens),
            "total_lemmatized_tokens": len(lemmatized_tokens)
        },
        "vocabulary_statistics": {
            "original_vocabulary_size": len(original_vocab),
            "stemmed_vocabulary_size": len(stemmed_vocab),
            "lemmatized_vocabulary_size": len(lemmatized_vocab),
            "reduction_after_stemming": len(original_vocab) - len(stemmed_vocab),
            "reduction_after_lemmatization": len(original_vocab) - len(lemmatized_vocab)
        },
        "over_stemming_examples": overstemming_examples,
        "lemmatization_preserves_meaning_examples": semantic_examples
    }

    with open(os.path.join(output_dir, "analysis_summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    with open(os.path.join(output_dir, "original_vocabulary.txt"), "w", encoding="utf-8") as f:
        for word in sorted(original_vocab):
            f.write(word + "\n")

    with open(os.path.join(output_dir, "stemmed_vocabulary.txt"), "w", encoding="utf-8") as f:
        for word in sorted(stemmed_vocab):
            f.write(word + "\n")

    with open(os.path.join(output_dir, "lemmatized_vocabulary.txt"), "w", encoding="utf-8") as f:
        for word in sorted(lemmatized_vocab):
            f.write(word + "\n")

    report_lines = []
    report_lines.append("PART 3 - QUESTION 2")
    report_lines.append("=" * 70)
    report_lines.append("")
    report_lines.append("Task performed:")
    report_lines.append("Using the word-tokenized Cranfield dataset, Porter stemming and WordNet lemmatization were performed.")
    report_lines.append("Sentence segmentation was done using NLTK Punkt and word tokenization was done using the NLTK Penn Treebank tokenizer.")
    report_lines.append("")
    report_lines.append("Vocabulary size comparison:")
    report_lines.append(f"Original vocabulary size      : {len(original_vocab)}")
    report_lines.append(f"Stemmed vocabulary size       : {len(stemmed_vocab)}")
    report_lines.append(f"Lemmatized vocabulary size    : {len(lemmatized_vocab)}")
    report_lines.append(f"Reduction after stemming      : {len(original_vocab) - len(stemmed_vocab)}")
    report_lines.append(f"Reduction after lemmatization : {len(original_vocab) - len(lemmatized_vocab)}")
    report_lines.append("")
    report_lines.append("Interpretation:")
    report_lines.append("Stemming usually reduces the vocabulary more aggressively because multiple inflected forms are collapsed by suffix stripping.")
    report_lines.append("Lemmatization also reduces vocabulary, but usually less aggressively, because it tries to preserve valid dictionary forms.")
    report_lines.append("")

    report_lines.append("Examples of over-stemming:")
    if len(overstemming_examples) == 0:
        report_lines.append("No strong over-stemming examples were automatically detected using the selected heuristic.")
    else:
        for ex in overstemming_examples[:5]:
            report_lines.append(
                f"{ex['original']} -> stemmed: {ex['stemmed']} | lemmatized: {ex['lemmatized']}"
            )

    report_lines.append("")
    report_lines.append("Examples where lemmatization preserves semantic meaning better:")
    if len(semantic_examples) == 0:
        report_lines.append("No strong examples were automatically detected using the selected heuristic.")
    else:
        for ex in semantic_examples[:5]:
            report_lines.append(
                f"{ex['original']} -> stemmed: {ex['stemmed']} | lemmatized: {ex['lemmatized']}"
            )

    report_lines.append("")
    report_lines.append("Conclusion:")
    report_lines.append("Porter stemming is more aggressive and often produces truncated forms that are useful for matching but less interpretable.")
    report_lines.append("WordNet lemmatization preserves meaningful base forms better and is therefore linguistically cleaner, although it may reduce vocabulary less than stemming.")

    with open(os.path.join(output_dir, "report_notes.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(report_lines))

    print("Done.")
    print("All outputs saved in:", output_dir)
    print()
    print("Vocabulary Size Comparison")
    print("=" * 50)
    print(f"Original vocabulary size   : {len(original_vocab)}")
    print(f"Stemmed vocabulary size    : {len(stemmed_vocab)}")
    print(f"Lemmatized vocabulary size : {len(lemmatized_vocab)}")
    print()
    print("Top over-stemming examples:")
    for ex in overstemming_examples[:5]:
        print(f"  {ex['original']} -> {ex['stemmed']} | {ex['lemmatized']}")
    print()
    print("Top lemmatization-preserves-meaning examples:")
    for ex in semantic_examples[:5]:
        print(f"  {ex['original']} -> {ex['stemmed']} | {ex['lemmatized']}")


if __name__ == "__main__":
    main()