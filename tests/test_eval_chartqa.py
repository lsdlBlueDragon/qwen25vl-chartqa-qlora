from src.eval_chartqa import EvaluationConfig, evaluate_records, relaxed_numeric_match


def test_percent_scale_still_counts_as_relaxed_numeric():
    assert relaxed_numeric_match("72", "0.72", EvaluationConfig())


def test_year_question_does_not_use_relaxed_numeric_tolerance():
    records = [
        {
            "question": "In which year did the blue line peak?",
            "answer": "2006",
            "reference_answer": "2018",
        }
    ]

    metrics, evaluated, errors = evaluate_records(records)

    assert metrics["exact_match"] == 0
    assert metrics["relaxed_correct"] == 0
    assert evaluated[0]["eval_relaxed_numeric_match"] is False
    assert len(errors) == 1


def test_plain_four_digit_year_reference_is_strict_even_without_year_word():
    records = [
        {
            "question": "What is the label at the peak?",
            "answer": "2006",
            "reference_answer": "2018",
        }
    ]

    metrics, evaluated, errors = evaluate_records(records)

    assert metrics["relaxed_correct"] == 0
    assert evaluated[0]["eval_relaxed_numeric_match"] is False
    assert len(errors) == 1
