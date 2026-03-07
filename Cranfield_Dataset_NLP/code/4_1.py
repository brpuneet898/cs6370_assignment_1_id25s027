import os
import json
from sentenceSegmentation import SentenceSegmentation
from tokenization import Tokenization
from stopwordRemoval import StopwordRemoval

def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)

def count_tokens(tokenized_docs):
    total_tokens = 0
    for doc in tokenized_docs:
        for sent in doc:
            total_tokens += len(sent)
    return total_tokens

def count_sentences(segmented_docs):
    total_sentences = 0
    for doc in segmented_docs:
        total_sentences += len(doc)
    return total_sentences


def main():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(current_dir)

    dataset_path = os.path.join(project_root, "cranfield_dataset", "cran_docs.json")
    output_dir = os.path.join(project_root, "output_theory", "4_1")

    ensure_dir(output_dir)

    with open(dataset_path, "r", encoding="utf-8") as f:
        docs_json = json.load(f)

    doc_ids = []
    docs = []

    for item in docs_json:
        doc_ids.append(item.get("id"))
        docs.append(item.get("body", ""))

    sentence_segmenter = SentenceSegmentation()
    tokenizer = Tokenization()
    stopword_remover = StopwordRemoval()

    segmented_docs = []
    for doc in docs:
        segmented_docs.append(sentence_segmenter.punkt(doc))

    tokenized_docs = []
    for sent_list in segmented_docs:
        tokenized_docs.append(tokenizer.pennTreeBank(sent_list))

    stopword_removed_docs = []
    for doc in tokenized_docs:
        stopword_removed_docs.append(stopword_remover.fromList(doc))

    segmented_output = []
    for i in range(len(doc_ids)):
        segmented_output.append({
            "doc_id": doc_ids[i],
            "sentences": segmented_docs[i]
        })

    with open(os.path.join(output_dir, "segmented_docs_punkt.json"), "w", encoding="utf-8") as f:
        json.dump(segmented_output, f, indent=2, ensure_ascii=False)

    tokenized_output = []
    for i in range(len(doc_ids)):
        tokenized_output.append({
            "doc_id": doc_ids[i],
            "tokenized_sentences": tokenized_docs[i]
        })

    with open(os.path.join(output_dir, "tokenized_docs_ptb.json"), "w", encoding="utf-8") as f:
        json.dump(tokenized_output, f, indent=2, ensure_ascii=False)

    stopword_removed_output = []
    for i in range(len(doc_ids)):
        stopword_removed_output.append({
            "doc_id": doc_ids[i],
            "stopword_removed_sentences": stopword_removed_docs[i]
        })

    with open(os.path.join(output_dir, "stopword_removed_docs_nltk.json"), "w", encoding="utf-8") as f:
        json.dump(stopword_removed_output, f, indent=2, ensure_ascii=False)

    total_docs = len(docs)
    total_sentences = count_sentences(segmented_docs)
    total_tokens_before = count_tokens(tokenized_docs)
    total_tokens_after = count_tokens(stopword_removed_docs)
    total_stopwords_removed = total_tokens_before - total_tokens_after

    summary = {
        "number_of_documents": total_docs,
        "sentence_segmenter_used": "NLTK Punkt",
        "tokenizer_used": "NLTK Penn Treebank",
        "stopword_list_used": "NLTK English stopwords (curated list)",
        "statistics": {
            "total_sentences": total_sentences,
            "total_tokens_before_stopword_removal": total_tokens_before,
            "total_tokens_after_stopword_removal": total_tokens_after,
            "total_stopwords_removed": total_stopwords_removed
        }
    }

    with open(os.path.join(output_dir, "stopword_removal_summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    sample_comparisons = []
    sample_count = min(5, len(doc_ids))

    collected = 0
    for i in range(len(doc_ids)):
        if collected >= sample_count:
            break

        if len(tokenized_docs[i]) == 0:
            continue

        sample_comparisons.append({
            "doc_id": doc_ids[i],
            "before_stopword_removal": tokenized_docs[i][:3],   
            "after_stopword_removal": stopword_removed_docs[i][:3]
        })
        collected += 1

    with open(os.path.join(output_dir, "sample_stopword_removal_comparison.json"), "w", encoding="utf-8") as f:
        json.dump(sample_comparisons, f, indent=2, ensure_ascii=False)

    report_lines = []
    report_lines.append("PART 4 - QUESTION 1")
    report_lines.append("=" * 70)
    report_lines.append("")
    report_lines.append("Task performed:")
    report_lines.append("Stopwords were removed from the tokenized Cranfield documents using a curated stopword list.")
    report_lines.append("The curated list used here is the NLTK English stopword list.")
    report_lines.append("")
    report_lines.append("Pipeline used:")
    report_lines.append("1. Sentence segmentation: NLTK Punkt")
    report_lines.append("2. Word tokenization: NLTK Penn Treebank tokenizer")
    report_lines.append("3. Stopword removal: NLTK curated English stopword list")
    report_lines.append("")
    report_lines.append("Corpus statistics:")
    report_lines.append(f"Number of documents: {total_docs}")
    report_lines.append(f"Total sentences: {total_sentences}")
    report_lines.append(f"Total tokens before stopword removal: {total_tokens_before}")
    report_lines.append(f"Total tokens after stopword removal: {total_tokens_after}")
    report_lines.append(f"Total stopwords removed: {total_stopwords_removed}")
    report_lines.append("")
    report_lines.append("Interpretation:")
    report_lines.append("Common function words such as articles, prepositions, conjunctions, and auxiliary verbs are removed.")
    report_lines.append("This reduces noise in the document representation and keeps more content-bearing words for later IR tasks.")
    report_lines.append("")
    report_lines.append("Saved files:")
    report_lines.append("1. segmented_docs_punkt.json")
    report_lines.append("2. tokenized_docs_ptb.json")
    report_lines.append("3. stopword_removed_docs_nltk.json")
    report_lines.append("4. stopword_removal_summary.json")
    report_lines.append("5. sample_stopword_removal_comparison.json")
    report_lines.append("6. report_notes.txt")

    with open(os.path.join(output_dir, "report_notes.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(report_lines))

    print("Done.")
    print("All outputs saved in:", output_dir)
    print()
    print("Summary")
    print("=" * 50)
    print(f"Number of documents               : {total_docs}")
    print(f"Total sentences                  : {total_sentences}")
    print(f"Tokens before stopword removal   : {total_tokens_before}")
    print(f"Tokens after stopword removal    : {total_tokens_after}")
    print(f"Total stopwords removed          : {total_stopwords_removed}")


if __name__ == "__main__":
    main()