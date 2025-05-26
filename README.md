# OPJ-Recenzije

Sentiment Analysis Project using Python for our Natural Language Processing Course.
- 6 annotators
- 3800 sentences for learning, 1500 sentences for testing (all in their respective spreadsheets/tsv files)


# Dataset - 3 label system
## Annotation guideline  
|Label|Sentiment |
|--|--|
| 0 | Positive |
| 1 | Neutral |
| 2 | Negative |

The original implementation used a 5 label system that was subsequently reworked into a 3 label system. During this, labels for sarcasm and mixed sentiment were dropped.

## Our dataset

Label Distribution:
- Positive (0): 789 (25.1%)
- Neutral (1): 1960 (62.4%)
- Negative (2): 390 (12.4%)
- Total sentences: 3139

Sentence Length Statistics:
- Average words per sentence: 22.57
- Minimum words: 1
- Maximum words: 104

## Combined dataset of all 3 groups

Label Distribution:
- Positive (0): 2753 (27.2%)
- Neutral (1): 6236 (61.7%)
- Negative (2): 1115 (11.0%)
- Total sentences: 10107

Sentence Length Statistics:
- Average words per sentence: 22.41
- Minimum words: 1
- Maximum words: 152

The dataset was created using the stratify option on all three corpora to create their respective test and train sets.

Group 1 (Test-1 and Train-1) originally used 1 for negative and 2 for positive. This has been adjusted to match the sets from other groups.


# Machine Learning results

| # | method | algorithm | train | Test 1 | Test 2 | Test 3 (Ours)|
|--------|------------------|--------------------------|--------|---------------------------------------------------------|---------------------------------------------------------|---------------------------------------------------------|
| 1.a.i | Machine learning | Logistic Regression | Train3 | Precision=0.593, Recall=0.616, F1=0.602, Accuracy=0.616 | Precision=0.595, Recall=0.632, F1=0.600, Accuracy=0.632 | Precision=0.640, Recall=0.665, **F1=0.636**, Accuracy=0.665 |
| 1.a.ii | Machine learning | Logistic Regression | TRAIN | Precision=0.617, Recall=0.629, F1=0.620, Accuracy=0.629 | Precision=0.611, Recall=0.632, F1=0.619, Accuracy=0.632 | Precision=0.659, Recall=0.678, **F1=0.660**, Accuracy=0.678 |
| 1.b.i | Machine learning | Multinomial Naive Bayes | Train3 | Precision=0.598, Recall=0.616, F1=0.606, Accuracy=0.616 | Precision=0.593, Recall=0.626, F1=0.602, Accuracy=0.626 | Precision=0.654, Recall=0.671, **F1=0.640**, Accuracy=0.671 |
| 1.b.ii | Machine learning | Multinomial Naive Bayes | TRAIN | Precision=0.606, Recall=0.659, F1=0.618, Accuracy=0.659 | Precision=0.637, Recall=0.673, F1=0.621, Accuracy=0.673 | Precision=0.676, Recall=0.694, **F1=0.646**, Accuracy=0.694 |

# Deep Learning results
