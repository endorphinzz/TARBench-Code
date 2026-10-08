# IAA Summary (E2/A2)

Coverage: 500 samples, annotators A, B, C.

## Binary label (Benign vs Anomalous)
| pair | kappa | agree | 95% CI |
|---|---|---|---|
| A-B | 0.7411 | 0.872 | 0.6852-0.7971 |
| A-C | 0.9176 | 0.962 | 0.8801-0.9516 |
| B-C | 0.7614 | 0.882 | 0.7036-0.8164 |

Krippendorff's alpha (3 raters): **0.803** (CI [0.7605, 0.8456])

## Full label (Benign + 14 types)
- A-B: kappa = 0.7616, agree = 0.814
- A-C: kappa = 0.8835, agree = 0.906
- B-C: kappa = 0.7759, agree = 0.826
- Krippendorff's alpha (3 raters): 0.8073

## Subclass agreement (co-anomalous samples, n=258)
- A-B: kappa = 0.8658, agree = 0.8876, n=258
- A-C: kappa = 0.893, agree = 0.9097, n=310
- B-C: kappa = 0.8709, agree = 0.8923, n=260
- alpha = 0.8744

## Per-class one-vs-rest kappa
| class | votes | A-B | A-C | B-C |
|---|---|---|---|---|
| BEN | 601 | 0.7411 | 0.9176 | 0.7614 |
| AC | 248 | 0.9327 | 0.915 | 0.8779 |
| AD | 128 | 0.3056 | 0.922 | 0.2957 |
| AR | 94 | 0.7808 | 0.7878 | 0.9078 |
| EL | 134 | 0.9756 | 0.9756 | 1.0 |
| AM | 35 | 1.0 | 0.9555 | 0.9555 |
| IS | 39 | 0.6585 | 0.6568 | 0.7569 |
| BE | 26 | 1.0 | 0.8869 | 0.8869 |
| TM | 50 | 0.9391 | 1.0 | 0.9391 |
| AB | 3 | 0.0 | 0.0 | 1.0 |
| CV | 30 | 0.6245 | 0.5919 | 0.7568 |
| PE | 104 | 0.6634 | 0.7611 | 0.5988 |
| PD | 8 | 0.0 | 1.0 | 0.0 |

## vs ground truth
| annotator | acc | P | R | F1 | type-in-set | FP | FN |
|---|---|---|---|---|---|---|---|
| A | 0.932 | 0.9938 | 0.9086 | 0.9493 | 0.7516 | 2 | 32 |
| B | 0.812 | 0.9923 | 0.7371 | 0.8459 | 0.8372 | 2 | 92 |
| C | 0.922 | 0.9875 | 0.9 | 0.9417 | 0.7587 | 4 | 35 |
| majority | 0.916 | 0.9936 | 0.8857 | 0.9366 | 0.7677 | 2 | 40 |

## Adjudication load
- label disagreements: 71/500
- type disagreements among all-anomalous: 39
- majority type ties: 7
