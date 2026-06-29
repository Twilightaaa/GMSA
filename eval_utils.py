import re
import string
from collections import Counter


def normalize_answer(text):
    def remove_articles(value):
        return re.sub(r"\b(a|an|the)\b", " ", value)

    def remove_punctuation(value):
        punctuation = set(string.punctuation)
        return "".join(char for char in value if char not in punctuation)

    return " ".join(
        remove_articles(remove_punctuation(str(text).lower())).split()
    )


def compute_exact(reference, prediction):
    return int(normalize_answer(prediction) == normalize_answer(reference))


def compute_f1(prediction, ground_truth):
    prediction_tokens = normalize_answer(prediction).split()
    ground_truth_tokens = normalize_answer(ground_truth).split()
    if not prediction_tokens or not ground_truth_tokens:
        return float(prediction_tokens == ground_truth_tokens)

    common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
    same_count = sum(common.values())
    if same_count == 0:
        return 0.0
    precision = same_count / len(prediction_tokens)
    recall = same_count / len(ground_truth_tokens)
    return 2 * precision * recall / (precision + recall)


def score_prediction(prediction, references):
    references = references if isinstance(references, list) else [references]
    references = [
        reference
        for reference in references
        if isinstance(reference, str) and reference.strip()
    ]
    if not references:
        raise ValueError("answer must contain at least one non-empty string")

    exact = max(compute_exact(reference, prediction) for reference in references)
    f1 = max(compute_f1(prediction, reference) for reference in references)
    return exact, f1
