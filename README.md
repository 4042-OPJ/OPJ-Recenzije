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

----------------------------------------------------------------------------------------------------------------------------------
You can find all the models metioned below on [our HuggingFace profile](https://huggingface.co/InfZnDipl). (Work in progress...)
----------------------------------------------------------------------------------------------------------------------------------

# Machine Learning results

| # | method | algorithm | train | Test 1 | Test 2 | Test 3 (Ours)|
|--------|------------------|--------------------------|--------|---------------------------------------------------------|---------------------------------------------------------|---------------------------------------------------------|
| 1.a.i | Machine learning | Logistic Regression | Train3 | Precision=0.593, Recall=0.616, F1=0.602, Accuracy=0.616 | Precision=0.595, Recall=0.632, F1=0.600, Accuracy=0.632 | Precision=0.640, Recall=0.665, **F1=0.636**, Accuracy=0.665 |
| 1.a.ii | Machine learning | Logistic Regression | TRAIN | Precision=0.617, Recall=0.629, F1=0.620, Accuracy=0.629 | Precision=0.611, Recall=0.632, F1=0.619, Accuracy=0.632 | Precision=0.659, Recall=0.678, **F1=0.660**, Accuracy=0.678 |
| 1.b.i | Machine learning | Multinomial Naive Bayes | Train3 | Precision=0.598, Recall=0.616, F1=0.606, Accuracy=0.616 | Precision=0.593, Recall=0.626, F1=0.602, Accuracy=0.626 | Precision=0.654, Recall=0.671, **F1=0.640**, Accuracy=0.671 |
| 1.b.ii | Machine learning | Multinomial Naive Bayes | TRAIN | Precision=0.606, Recall=0.659, F1=0.618, Accuracy=0.659 | Precision=0.637, Recall=0.673, F1=0.621, Accuracy=0.673 | Precision=0.676, Recall=0.694, **F1=0.646**, Accuracy=0.694 |

# Deep Learning results
CNN Results

Model trained on TRAIN (train1 + train2 + train3), hyperparameters adjusted for test-1

parameters: {'lr': 0.0045992912985175025, 'dropout': 0.45827226824902384, 'num_filters': 350, 'batch_size': 16}

model: CNN1-T-t11.pt

| train set | test set | Accuracy | Precision | Recall | Macro F1 score |
|-----------|----------|----------|-----------|--------|----------------|
| TRAIN     | test1    | 0.6401   | 0.5088    | 0.5256 | 0.5158         |
| TRAIN     | test2    | 0.6032   | 0.5134    | 0.5139 | 0.5135         |
| TRAIN     | test3    | 0.7856   | 0.8008    | 0.7857 | 0.7822         |

Model trained on train-3, hyperparameters adjusted for test-1

parameters: {'lr': 4.6049000018608994e-05, 'dropout': 0.16486342956301311, 'num_filters': 250, 'batch_size': 16}

model: CNN1-t3-t1.pt

| train set | test set | Accuracy | Precision | Recall | Macro F1 score |
|-----------|----------|----------|-----------|--------|----------------|
| test3     | test1    | 0.4472   | 0.4363    | 0.4938 | 0.4067         |
| test3     | test2    | 0.5317   | 0.4874    | 0.5117 | 0.4817         |
| test3     | test3    | 0.8096   | 0.8107    | 0.8097 | 0.8075         |

Other model results can be found in the results folder under Method2/DL/results

Models such as CNN1-T-t3, which had their hyperparams adjusted for the test-3, obviously show better results for the test-3 set, but test-1 scores seems to drop significantly when the hyperparams are adjusted for any other set.

| train set | test set | accuracy | Macro F1 score |
|-----------|----------|----------|----------------|
| TRAIN     | test1    | 0.5743   | 0.4858         |
| TRAIN     | test2    | 0.6059   | 0.5455         |
| TRAIN     | test3    | 0.8108   | 0.8096         |


RNN Results

Model trained on TRAIN (train1 + train2 + train3), hyperparameters adjusted for test-3

Model: RNN-T-t3.pt

| train set | test set | Weighted F1 score |
|-----------|----------|-------------------|
| TRAIN     | test1    | 58.62             |
| TRAIN     | test2    | 59.07             |
| TRAIN     | test3    | 77.79             |

Model trained on train-3, hyperparameters adjusted for test-2

Model: RNN-t3-t2.pt

| train set | test set | Weighted F1 score |
|-----------|----------|-------------------|
| TRAIN     | test1    | 44.58             |
| TRAIN     | test2    | 49.07             |
| TRAIN     | test3    | 80.67             |


# Fine-Tuned Transformer Results

Fine-tuned Transformer models made using pre-trained **BERTić** and **CroSloEngual BERT** models.  
Fine-tuned models are available on Huggingface in `.zip` files (each `.zip` contains models trained on two datasets - TRAIN and TRAIN3).

### CroSloEngual BERT - TRAIN3

| Test set | Accuracy | Precision | Recall | F1     |
|----------|----------|-----------|--------|--------|
| TEST1    | 0.6801   | 0.7111    | 0.6801 | 0.6911 |
| TEST2    | 0.7840   | 0.7845    | 0.7840 | 0.7842 |
| TEST3    | 0.8051   | 0.8017    | 0.8051 | 0.8002 |

### CroSloEngual BERT - TRAIN

| Test set | Accuracy | Precision | Recall | F1     |
|----------|----------|-----------|--------|--------|
| TEST1    | 0.7169   | 0.7109    | 0.7169 | 0.7134 |
| TEST2    | 0.7905   | 0.7861    | 0.7905 | 0.7800 |
| TEST3    | 0.8255   | 0.8298    | 0.8255 | 0.8182 |

### BERTić - TRAIN3

| Test set | Accuracy | Precision | Recall | F1     |
|----------|----------|-----------|--------|--------|
| TEST1    | 0.6471   | 0.7061    | 0.6471 | 0.6597 |
| TEST2    | 0.7430   | 0.7734    | 0.7430 | 0.7481 |
| TEST3    | 0.8280   | 0.8283    | 0.8280 | 0.8270 |

### BERTić - TRAIN

| Test set | Accuracy | Precision | Recall | F1     |
|----------|----------|-----------|--------|--------|
| TEST1    | 0.7034   | 0.7199    | 0.7034 | 0.7099 |
| TEST2    | 0.8089   | 0.8103    | 0.8089 | 0.8093 |
| TEST3    | 0.8433   | 0.8428    | 0.8433 | 0.8429 |
