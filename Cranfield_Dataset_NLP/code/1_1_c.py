import os
import json
import csv
from sentenceSegmentation import SentenceSegmentation


def normalize_segments(segments):
    return [s.strip() for s in segments if s and s.strip()]


def build_test_suite():
    return [
        {
            "id": 1,
            "category": "Abbreviation",
            "text": "Dr. Smith arrived late for the meeting.",
            "gold": ["Dr. Smith arrived late for the meeting."]
        },
        {
            "id": 2,
            "category": "MAbbreviation",
            "text": "Mr. and Mrs. Johnson live in Chennai.",
            "gold": ["Mr. and Mrs. Johnson live in Chennai."]
        },
        {
            "id": 3,
            "category": "Abbreviation",
            "text": "The meeting starts at 10 a.m. tomorrow.",
            "gold": ["The meeting starts at 10 a.m. tomorrow."]
        },
        {
            "id": 4,
            "category": "Decimal numbers",
            "text": "She earned 3.75 out of 4.0 in the exam.",
            "gold": ["She earned 3.75 out of 4.0 in the exam."]
        },
        {
            "id": 5,
            "category": "Initials",
            "text": "The U.S.A. team won the match easily.",
            "gold": ["The U.S.A. team won the match easily."]
        },
        {
            "id": 6,
            "category": "Initials",
            "text": "J.K. Rowling wrote several famous novels.",
            "gold": ["J.K. Rowling wrote several famous novels."]
        },
        {
            "id": 7,
            "category": "Ellipsis",
            "text": "Wait... I am not ready yet.",
            "gold": ["Wait...", "I am not ready yet."]
        },
        {
            "id": 8,
            "category": "Multiple punctuation",
            "text": "What?! Are you serious?",
            "gold": ["What?!", "Are you serious?"]
        },
        {
            "id": 9,
            "category": "Quotes",
            "text": "She said, \"I will come tomorrow.\"",
            "gold": ["She said, \"I will come tomorrow.\""]
        },
        {
            "id": 10,
            "category": "Quotes",
            "text": "Did he really say \"No.\" and leave?",
            "gold": ["Did he really say \"No.\" and leave?"]
        },
        {
            "id": 11,
            "category": "Decimal Numbers",
            "text": "Please refer to Sec. 3.2 for details.",
            "gold": ["Please refer to Sec. 3.2 for details."]
        },
        {
            "id": 12,
            "category": "Abbreviation",
            "text": "The price is Rs. 250 only.",
            "gold": ["The price is Rs. 250 only."]
        },
        {
            "id": 13,
            "category": "Link",
            "text": "My email is abc.xyz@example.com for official contact.",
            "gold": ["My email is abc.xyz@example.com for official contact."]
        },
        {
            "id": 14,
            "category": "Link",
            "text": "Visit www.example.com for more information.",
            "gold": ["Visit www.example.com for more information."]
        },
        {
            "id": 15,
            "category": "Abbreviation",
            "text": "He lives on St. Thomas Road near the church.",
            "gold": ["He lives on St. Thomas Road near the church."]
        }
    ]


def get_predictions(segmenter, text):
    return {
        "naive": normalize_segments(segmenter.naive(text)),
        "punkt": normalize_segments(segmenter.punkt(text)),
        "spacy": normalize_segments(segmenter.spacySegmenter(text)),
    }


def evaluate(test_suite, segmenter):
    detailed_rows = []

    summary = {
        "naive": {"total": 0, "correct": 0, "errors": 0},
        "punkt": {"total": 0, "correct": 0, "errors": 0},
        "spacy": {"total": 0, "correct": 0, "errors": 0},
    }

    for item in test_suite:
        gold = normalize_segments(item["gold"])
        preds = get_predictions(segmenter, item["text"])

        row = {
            "id": item["id"],
            "category": item["category"],
            "text": item["text"],
            "gold": gold,
            "naive_pred": preds["naive"],
            "punkt_pred": preds["punkt"],
            "spacy_pred": preds["spacy"],
            "naive_correct": preds["naive"] == gold,
            "punkt_correct": preds["punkt"] == gold,
            "spacy_correct": preds["spacy"] == gold,
        }
        detailed_rows.append(row)

        for method in ["naive", "punkt", "spacy"]:
            summary[method]["total"] += 1
            if preds[method] == gold:
                summary[method]["correct"] += 1
            else:
                summary[method]["errors"] += 1

    return detailed_rows, summary


def save_outputs(output_dir, test_suite, detailed_rows, summary):
    os.makedirs(output_dir, exist_ok=True)

    with open(os.path.join(output_dir, "adversarial_test_suite.json"), "w", encoding="utf-8") as f:
        json.dump(test_suite, f, indent=4, ensure_ascii=False)

    with open(os.path.join(output_dir, "adversarial_detailed_results.json"), "w", encoding="utf-8") as f:
        json.dump(detailed_rows, f, indent=4, ensure_ascii=False)

    csv_path = os.path.join(output_dir, "adversarial_detailed_results.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "id", "category", "text", "gold",
            "naive_pred", "naive_correct",
            "punkt_pred", "punkt_correct",
            "spacy_pred", "spacy_correct"
        ])
        for row in detailed_rows:
            writer.writerow([
                row["id"],
                row["category"],
                row["text"],
                " || ".join(row["gold"]),
                " || ".join(row["naive_pred"]),
                row["naive_correct"],
                " || ".join(row["punkt_pred"]),
                row["punkt_correct"],
                " || ".join(row["spacy_pred"]),
                row["spacy_correct"],
            ])

    summary_csv_path = os.path.join(output_dir, "adversarial_error_summary.csv")
    with open(summary_csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["Segmenter", "No. of Test Cases", "No. of Correct Segmentations", "No. of Errors"])
        writer.writerow(["Naive Top-Down", summary["naive"]["total"], summary["naive"]["correct"], summary["naive"]["errors"]])
        writer.writerow(["NLTK Punkt", summary["punkt"]["total"], summary["punkt"]["correct"], summary["punkt"]["errors"]])
        writer.writerow(["spaCy", summary["spacy"]["total"], summary["spacy"]["correct"], summary["spacy"]["errors"]])

    md_table_path = os.path.join(output_dir, "adversarial_error_summary.md")
    with open(md_table_path, "w", encoding="utf-8") as f:
        f.write("| Segmenter | No. of Test Cases | No. of Correct Segmentations | No. of Errors |\n")
        f.write("|---|---:|---:|---:|\n")
        f.write(f"| Naive Top-Down | {summary['naive']['total']} | {summary['naive']['correct']} | {summary['naive']['errors']} |\n")
        f.write(f"| NLTK Punkt | {summary['punkt']['total']} | {summary['punkt']['correct']} | {summary['punkt']['errors']} |\n")
        f.write(f"| spaCy | {summary['spacy']['total']} | {summary['spacy']['correct']} | {summary['spacy']['errors']} |\n")

    txt_path = os.path.join(output_dir, "adversarial_error_summary.txt")
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write("Adversarial Sentence Segmentation Evaluation Summary\n")
        f.write("=" * 55 + "\n\n")
        for method_name, label in [
            ("naive", "Naive Top-Down"),
            ("punkt", "NLTK Punkt"),
            ("spacy", "spaCy")
        ]:
            f.write(f"{label}\n")
            f.write(f"  Total Test Cases           : {summary[method_name]['total']}\n")
            f.write(f"  Correct Segmentations      : {summary[method_name]['correct']}\n")
            f.write(f"  Errors                     : {summary[method_name]['errors']}\n\n")

    return summary_csv_path, md_table_path, txt_path


def print_console_table(summary):
    print("\nAdversarial Test Suite Results")
    print("=" * 72)
    print(f"{'Segmenter':<20} {'Test Cases':<12} {'Correct':<12} {'Errors':<10}")
    print("-" * 72)
    print(f"{'Naive Top-Down':<20} {summary['naive']['total']:<12} {summary['naive']['correct']:<12} {summary['naive']['errors']:<10}")
    print(f"{'NLTK Punkt':<20} {summary['punkt']['total']:<12} {summary['punkt']['correct']:<12} {summary['punkt']['errors']:<10}")
    print(f"{'spaCy':<20} {summary['spacy']['total']:<12} {summary['spacy']['correct']:<12} {summary['spacy']['errors']:<10}")
    print("=" * 72)


def main():
    output_dir = os.path.join("..", "output_theory")

    test_suite = build_test_suite()
    segmenter = SentenceSegmentation()

    detailed_rows, summary = evaluate(test_suite, segmenter)
    summary_csv_path, md_table_path, txt_path = save_outputs(
        output_dir, test_suite, detailed_rows, summary
    )

    print_console_table(summary)

    print("\nSaved files:")
    print(f"1. Detailed JSON   : {os.path.abspath(os.path.join(output_dir, 'adversarial_detailed_results.json'))}")
    print(f"2. Detailed CSV    : {os.path.abspath(os.path.join(output_dir, 'adversarial_detailed_results.csv'))}")
    print(f"3. Summary CSV     : {os.path.abspath(summary_csv_path)}")
    print(f"4. Markdown table  : {os.path.abspath(md_table_path)}")
    print(f"5. Text summary    : {os.path.abspath(txt_path)}")


if __name__ == "__main__":
    main()