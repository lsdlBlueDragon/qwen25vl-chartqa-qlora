# ChartQA 22C Codex visual review - 2026-07-03

## Scope

This is a local, no-GPU visual review of the 34 high-value samples from Module 22C.
The review reads the mounted Google Drive images and existing Module 21 / 22B outputs.
No model inference, training, or full-val run was performed.

Source queue:

```text
outputs/chartqa_staged_extraction_quality_audit_22c/chartqa_22c_high_value_manual_review_queue.csv
```

Drive image source:

```text
G:\我的云端硬盘\qwen25vl-chartqa-qlora\data\processed\chartqa_val_full_sft_1920_images
```

## Executive Finding

The high-value queue is not a single failure mode. It splits into four useful groups:

| group | samples | interpretation |
|---|---:|---|
| clear 22B/staged-extraction useful recoveries | 4-6 | staged table helps on counting, color/line mapping, local comparison, and arithmetic when the table is trusted |
| true visual / semantic / arithmetic failures | ~17 | still need better visual grounding, strict threshold handling, range aggregation, spatial grounding, or QA computation |
| answer normalization / evaluator issues | ~4 | model output is essentially correct but evaluator rejects wording, punctuation, star suffix, or close numeric format |
| likely reference/question issues | ~6 | chart contradicts the reference or the question is ambiguous enough that the sample should be excluded or separately tagged |

The current conclusion remains: Module 22B should be kept as a diagnostic and complementary route, but it should not replace Module 21 style image-only / one-shot runs. The next improvement target is not more JSON formatting. It is failure-layer separation:

- value extraction / color mapping / axis grounding;
- strict comparison and range aggregation;
- QA prompt forcing short final answers;
- evaluator normalization and reference cleanup.

## Reviewed Samples

| sample | relation | visual review judgment |
|---:|---|---|
| 13 | Module21 unique | Reference 155 is visually correct: peaks are about 80, 68, and 7. 22B loses peak values or sum reasoning. True staged extraction / aggregation failure. |
| 24 | Module21 unique | Iraqi dependents is the 12% slice, while 22B picks 19% Iraqi principal applicants. True role/group disambiguation failure. |
| 186 | 22B unique | X-axis shows 9 labeled years. 22B table-only correctly answers 9; image+table drifts to all years. Staged table is useful here. |
| 190 | Module21 unique | Only Southern Asia is greater than 60%; Eastern Asia is approximately equal to 60%, not greater. 22B fails strict inequality. |
| 245 | 22B unique | Peak year is visually close/ambiguous between 1980 and 2000. Reference is 1980, but 2000 appears at least comparable. Mark as ambiguous peak/reference-sensitive. |
| 255 | Module21 unique | Only 1998 is clearly below 15 dollars/barrel. 22B counts near-threshold years. True strict threshold / visual precision failure. |
| 362 | Module21 unique | Chart countries are Czech Republic and New Zealand. 22B outputs the same labels but evaluator rejects list format. Normalization/evaluator issue. |
| 369 | Module21 unique | Bars are 1.2k and 4.5k; average is 2.85k, not greater than 3k. 22B says Yes, likely unit-scale reasoning failure. |
| 419 | 22B unique | Ukraine 34.5 is less than the sum of the other four countries, 36.6. 22B table-only answers No correctly; image+table is distracted by longest bar. |
| 441 | 22B unique | Red line is below blue only in 2009. 22B table-only gets this right. Useful staged color/line mapping case. |
| 677 | 22B unique | Poland 21.8 vs Austria 21.5 differs by 0.3. 22B table-only gets this right; image+table confuses nearby Sweden. |
| 832 | 22B unique | Light-blue shortest is 21 and dark-blue tallest is 38. Reference 17 treats subtraction as absolute difference; literal subtraction is -17. Mark as sign/wording ambiguity. |
| 882 | Module21 unique | Chart legend has 2028*, reference is 2028. 22B predicts 2028*. Normalization issue, not visual failure. |
| 1190 | axis/legend JSON issue | Reference 2008 is questionable: increases occur in several later years, while 2008 has no previous comparison. Mark as question/reference ambiguity. |
| 28 | table contains reference but QA failed | Dark grey segment corresponds to Both. 22B answers Don't know. True color/legend mapping failure. |
| 138 | table contains reference but QA failed | Smallest value is South Sudan 0.53%, not Angola 41.39%. Reference appears wrong; 22B answer is visually correct. |
| 154 | table contains reference but QA failed | Correct answer is orange. 22B image+table says the answer is orange inside a sentence, but evaluator rejects it. Final-answer normalization issue. |
| 189 | table contains reference but QA failed | Four true age groups are 15-49, 50-69, 70+, and 5-14. All ages / age-standardized should not count. 22B semantic filtering failure. |
| 229 | table contains reference but QA failed | Male series peak is 2018, slightly higher than 2016. 22B picks 2016. True near-peak date-axis reading failure. |
| 250 | table contains reference but QA failed | Agriculture series peaks in 1999; 2000 is slightly lower. 22B picks 2000. True neighboring-year visual precision failure. |
| 281 | table contains reference but QA failed | Highest navy blue is 47; median light blue is about 28; sum is 75, not greater than 100. QA arithmetic/median failure. |
| 317 | table contains reference but QA failed | All voters: Very=57 and Somewhat=34, so answer should be Yes. Reference says No. Reference error. |
| 326 | table contains reference but QA failed | Rightmost bar in the middle row is 50. 22B picks 63 from another row. Spatial grounding failure. |
| 424 | table contains reference but QA failed | Brown/orange bar is Northwest Europe marker price. 22B returns either value or wrong label. Color-to-label mapping failure. |
| 529 | table contains reference but QA failed | YellowPages is 12%; three times is 36%, matching Industry specific. 22B fails multiplicative target lookup. |
| 675 | table contains reference but QA failed | People over 55 means 55-64 plus 65+, 14+8=22. 22B only takes 65+. Range aggregation failure. |
| 778 | table contains reference but QA failed | Very likely is highest for 30-44 at 17%, slightly above 18-29 at 16%. 22B fails close-value max selection. |
| 779 | table contains reference but QA failed | Minimum difference between Not likely at all and Very likely is not 65+; 65+ is the largest gap. Reference likely wrong. |
| 781 | table contains reference but QA failed | Biggest gender wage gap is Some college or associate's degree, about 248. 22B chooses Advanced degree, likely using max value rather than max gap. |
| 877 | table contains reference but QA failed | Twitter has the smallest male/female gap, about 1 point. 22B picks Pinterest, likely using minimum single value instead of minimum difference. |
| 946 | table contains reference but QA failed | Grocers average is higher than General Merchandisers. Reference says General Merchandisers; reference likely wrong. |
| 977 | table contains reference but QA failed | Question asks for 2020 ICT sector value; chart value is 11.8 for ICT services. Reference 2022 is not consistent with the question. Reference/question mismatch. |
| 987 | table contains reference but QA failed | Question wording is ambiguous. Reference 2013 looks like a start year, while 22B picks 2019 as peak/latest increase. Needs reference/question cleanup. |
| 1065 | table contains reference but QA failed | Home furnishings is about 65.3%; 22B says 65 or 65%. This should be accepted under relaxed numeric matching. Evaluator/normalization issue. |

## Recommended Action

1. Add a cleanup tag list for samples that should not be counted as model failures:
   `138, 317, 362, 779, 882, 946, 977, 987, 1065, 1190`.
   Sample `245` and `832` should be marked ambiguous/reference-sensitive rather than clean failures.

2. Add answer normalization for:
   - list answers such as `Czech Republic, New Zealand`;
   - starred years such as `2028* -> 2028`;
   - sentence answers that contain the exact categorical answer, such as `... orange`;
   - percentage strings and close numeric values such as `65%` vs `65.3`.

3. For true model failures, prioritize targeted diagnostics:
   - strict inequality and near-threshold comparisons: `190, 255, 778`;
   - date-axis peak/neighbor precision: `229, 250`;
   - unit/scale reasoning: `369`;
   - range aggregation: `675`;
   - spatial grounding: `326`;
   - color/legend mapping: `28, 424`;
   - arithmetic/difference/median reasoning: `281, 529, 781, 877`.

4. Do not launch full-val yet. First convert this review into a cleaned denominator and a small prompt/evaluator ablation, otherwise metric changes will mix real model gains with dataset/evaluator cleanup.
