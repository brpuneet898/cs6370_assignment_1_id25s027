import os
import json
from statistics import mean

from sentenceSegmentation import SentenceSegmentation
from tokenization import Tokenization

## this file is created for question 2 part 3. 

def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)


def flatten_tokenized_docs(tokenized_docs):
    ## whatever the tokenization method, we need to calculate the total number of sentences, total number of tokens 
    ## and average tokens per sentence.
    total_sentences = 0
    total_tokens = 0

    for doc in tokenized_docs:
        total_sentences += len(doc)
        for sent in doc:
            total_tokens += len(sent)

    avg_tokens_per_sentence = 0.0
    if total_sentences > 0:
        avg_tokens_per_sentence = total_tokens / total_sentences

    return total_sentences, total_tokens, avg_tokens_per_sentence


def main():
    ## main function to perform sentence segmentation and tokenization on the Cranfield dataset and save the outputs 
    ## and statistics.
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(current_dir)

    dataset_path = os.path.join(project_root, "cranfield_dataset", "cran_docs.json")
    output_dir = os.path.join(project_root, "output_theory", "2_3")

    ensure_dir(output_dir)

    with open(dataset_path, "r", encoding="utf-8") as f:
        docs_json = json.load(f)

    doc_ids = []
    docs = []

    for item in docs_json:
        doc_id = item.get("id", None)
        body = item.get("body", "")

        doc_ids.append(doc_id)
        docs.append(body)

    sentence_segmenter = SentenceSegmentation()
    tokenizer = Tokenization()

    segmented_docs = []
    for doc in docs:
        segmented_docs.append(sentence_segmenter.punkt(doc))

    segmented_output = []
    for i in range(len(segmented_docs)):
        segmented_output.append({
            "doc_id": doc_ids[i],
            "sentences": segmented_docs[i]
        })

    with open(os.path.join(output_dir, "segmented_docs_punkt.json"), "w", encoding="utf-8") as f:
        json.dump(segmented_output, f, indent=2, ensure_ascii=False)

    naive_tokenized_docs = []
    ptb_tokenized_docs = []
    spacy_tokenized_docs = []

    for sent_list in segmented_docs:
        naive_tokenized_docs.append(tokenizer.naive(sent_list))
        ptb_tokenized_docs.append(tokenizer.pennTreeBank(sent_list))
        spacy_tokenized_docs.append(tokenizer.spacyTokenizer(sent_list))

    naive_output = []
    ptb_output = []
    spacy_output = []

    for i in range(len(segmented_docs)):
        naive_output.append({
            "doc_id": doc_ids[i],
            "tokenized_sentences": naive_tokenized_docs[i]
        })
        ptb_output.append({
            "doc_id": doc_ids[i],
            "tokenized_sentences": ptb_tokenized_docs[i]
        })
        spacy_output.append({
            "doc_id": doc_ids[i],
            "tokenized_sentences": spacy_tokenized_docs[i]
        })

    with open(os.path.join(output_dir, "tokenized_docs_naive.json"), "w", encoding="utf-8") as f:
        json.dump(naive_output, f, indent=2, ensure_ascii=False)

    with open(os.path.join(output_dir, "tokenized_docs_ptb.json"), "w", encoding="utf-8") as f:
        json.dump(ptb_output, f, indent=2, ensure_ascii=False)

    with open(os.path.join(output_dir, "tokenized_docs_spacy.json"), "w", encoding="utf-8") as f:
        json.dump(spacy_output, f, indent=2, ensure_ascii=False)

    naive_sentences, naive_tokens, naive_avg = flatten_tokenized_docs(naive_tokenized_docs)
    ptb_sentences, ptb_tokens, ptb_avg = flatten_tokenized_docs(ptb_tokenized_docs)
    spacy_sentences, spacy_tokens, spacy_avg = flatten_tokenized_docs(spacy_tokenized_docs)

    stats = {
        "number_of_documents": len(docs),
        "number_of_sentence_segmented_documents": len(segmented_docs),
        "sentence_segmenter_used": "Punkt",
        "tokenization_statistics": {
            "naive_top_down": {
                "total_sentences": naive_sentences,
                "total_tokens": naive_tokens,
                "average_tokens_per_sentence": naive_avg
            },
            "penn_treebank": {
                "total_sentences": ptb_sentences,
                "total_tokens": ptb_tokens,
                "average_tokens_per_sentence": ptb_avg
            },
            "spacy": {
                "total_sentences": spacy_sentences,
                "total_tokens": spacy_tokens,
                "average_tokens_per_sentence": spacy_avg
            }
        }
    }

    with open(os.path.join(output_dir, "tokenization_statistics.json"), "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)

    sample_count = min(5, len(segmented_docs))
    comparison_samples = []

    collected = 0
    for doc_index in range(len(segmented_docs)):
        if collected >= sample_count:
            break

        if len(segmented_docs[doc_index]) == 0:
            continue

        first_sentence = segmented_docs[doc_index][0]

        comparison_samples.append({
            "doc_id": doc_ids[doc_index],
            "sentence": first_sentence,
            "naive_tokens": tokenizer.naive([first_sentence])[0],
            "ptb_tokens": tokenizer.pennTreeBank([first_sentence])[0],
            "spacy_tokens": tokenizer.spacyTokenizer([first_sentence])[0]
        })

        collected += 1

    with open(os.path.join(output_dir, "sample_tokenization_comparison.json"), "w", encoding="utf-8") as f:
        json.dump(comparison_samples, f, indent=2, ensure_ascii=False)

    report_text = []
    report_text.append("PART 2 - QUESTION 3")
    report_text.append("=" * 60)
    report_text.append("")
    report_text.append("Task performed:")
    report_text.append("Word tokenization was performed on the sentence-segmented Cranfield documents using:")
    report_text.append("1. Proposed top-down tokenizer (naive regex-based tokenizer)")
    report_text.append("2. NLTK Penn Treebank tokenizer")
    report_text.append("3. spaCy tokenizer")
    report_text.append("")
    report_text.append("Sentence segmentation used before tokenization:")
    report_text.append("Punkt sentence segmenter")
    report_text.append("")
    report_text.append("Corpus statistics:")
    report_text.append(f"Number of documents: {len(docs)}")
    report_text.append(f"Total sentence units after segmentation: {len(segmented_docs)} document-level entries")
    report_text.append("")
    report_text.append("Tokenization statistics:")
    report_text.append(f"Naive tokenizer   -> total sentences: {naive_sentences}, total tokens: {naive_tokens}, average tokens/sentence: {naive_avg:.4f}")
    report_text.append(f"PTB tokenizer     -> total sentences: {ptb_sentences}, total tokens: {ptb_tokens}, average tokens/sentence: {ptb_avg:.4f}")
    report_text.append(f"spaCy tokenizer   -> total sentences: {spacy_sentences}, total tokens: {spacy_tokens}, average tokens/sentence: {spacy_avg:.4f}")
    report_text.append("")
    report_text.append("Scenario where the proposed top-down tokenizer performs better:")
    report_text.append("When only simple alphabetic word extraction is desired and punctuation should be discarded completely.")
    report_text.append("Example: 'engine-performance, fuel-flow, and pressure?'")
    report_text.append("Naive tokenizer output may be cleaner for a basic IR pipeline: ['engine', 'performance', 'fuel', 'flow', 'and', 'pressure']")
    report_text.append("PTB and spaCy typically preserve punctuation or split hyphenated forms into more pieces, which may be less convenient in a simplistic indexing setup.")
    report_text.append("")
    report_text.append("Scenario where the proposed top-down tokenizer performs worse:")
    report_text.append("When the text contains contractions, abbreviations, decimals, or hyphenated expressions.")
    report_text.append("Example: \"The U.S. aircraft can't maintain 3.5-mile stability.\"")
    report_text.append("Naive tokenizer may produce: ['The', 'U', 'S', 'aircraft', 'can', 't', 'maintain', '3', '5', 'mile', 'stability']")
    report_text.append("This loses important information. PTB and spaCy handle such cases more linguistically.")
    report_text.append("")
    report_text.append("Saved files:")
    report_text.append("1. segmented_docs_punkt.json")
    report_text.append("2. tokenized_docs_naive.json")
    report_text.append("3. tokenized_docs_ptb.json")
    report_text.append("4. tokenized_docs_spacy.json")
    report_text.append("5. tokenization_statistics.json")
    report_text.append("6. sample_tokenization_comparison.json")
    report_text.append("7. report_notes.txt")

    with open(os.path.join(output_dir, "report_notes.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(report_text))

    print("Done.")
    print("All outputs saved in:", output_dir)


if __name__ == "__main__":
    main()