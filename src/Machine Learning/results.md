# Main ML algorithms

| # | method | algorithm | train | Test 1 | Test 2 | Test 3 (Ours)|
|--------|------------------|--------------------------|--------|---------------------------------------------------------|---------------------------------------------------------|---------------------------------------------------------|
| 1.a.i | Machine learning | Logistic Regression | Train3 | Precision=0.599, Recall=0.623, F1=0.608, Accuracy=0.623 | Precision=0.595, Recall=0.632, F1=0.600, Accuracy=0.632 | Precision=0.640, Recall=0.665, **F1=0.636**, Accuracy=0.665 |
| 1.a.ii | Machine learning | Logistic Regression | TRAIN | Precision=0.617, Recall=0.629, F1=0.620, Accuracy=0.629 | Precision=0.611, Recall=0.632, F1=0.619, Accuracy=0.632 | Precision=0.659, Recall=0.678, **F1=0.660**, Accuracy=0.678 |
| 1.b.i | Machine learning | Multinomial Naive Bayes | Train3 | Precision=0.598, Recall=0.616, F1=0.606, Accuracy=0.616 | Precision=0.593, Recall=0.626, F1=0.602, Accuracy=0.626 | Precision=0.654, Recall=0.671, **F1=0.640**, Accuracy=0.671 |
| 1.b.ii | Machine learning | Multinomial Naive Bayes | TRAIN | Precision=0.606, Recall=0.659, F1=0.618, Accuracy=0.659 | Precision=0.637, Recall=0.673, F1=0.621, Accuracy=0.673 | Precision=0.676, Recall=0.694, **F1=0.646**, Accuracy=0.694 |

# Other ML algorithms

| # | model type | algorithm | train | Test 1 | Test 2 | Test 3 (Ours)|
|--------|------------------|--------------------------|--------|---------------------------------------------------------|---------------------------------------------------------|---------------------------------------------------------|
| 1.a.i | Linear model | RidgeClassifier | Train3 | Precision=0.601, Recall=0.631, F1=0.610, Accuracy=0.631| Precision=0.589, Recall=0.635, F1=0.593, Accuracy=0.635 | Precision=0.659, Recall=0.679, **F1=0.640**, Accuracy=0.679 |
| 1.a.ii | Linear model | RidgeClassifier | TRAIN | Precision=0.634, Recall=0.661, F1=0.642, Accuracy=0.661 | Precision=0.627, Recall=0.662, F1=0.635, Accuracy=0.662 | Precision=0.662, Recall=0.687, **F1=0.660**, Accuracy=0.687 |
| 1.b.i | Linear model | LinearRegression | Train3 | Precision=0.546, Recall=0.640, F1=0.558, Accuracy=0.640| Precision=0.648, Recall=0.640, F1=0.554, Accuracy=0.640 | Precision=0.661, Recall=0.656, **F1=0.574**, Accuracy=0.656 |
| 1.b.ii | Linear model | LinearRegression | TRAIN | Precision=0.636, Recall=0.673, F1=0.630, Accuracy=0.673 | Precision=0.642, Recall=0.668, F1=0.618, Accuracy=0.668 | Precision=0.680, Recall=0.688, **F1=0.643**, Accuracy=0.688 |
| 1.c.i | Linear model | SGDClassifier | Train3 | Precision=0.592, Recall=0.570, F1=0.579, Accuracy=0.570| Precision=0.607, Recall=0.612, F1=0.608, Accuracy=0.612 | Precision=0.632, Recall=0.647, **F1=0.639**, Accuracy=0.647 |
| 1.c.ii | Linear model | SGDClassifier | TRAIN | Precision=0.610, Recall=0.607, F1=0.608, Accuracy=0.607 | Precision=0.621, Recall=0.623, F1=0.622, Accuracy=0.623 | Precision=0.638, Recall=0.655, **F1=0.648**, Accuracy=0.655 |
| 2.a.i | Tree | DecisionTreeClassifier | Train3 | Precision=0.561, Recall=0.605, F1=0.572, Accuracy=0.605| Precision=0.525, Recall=0.581, F1=0.536, Accuracy=0.581 | Precision=0.578, Recall=0.624, **F1=0.553**, Accuracy=0.624 |
| 2.a.ii | Tree | DecisionTreeClassifier | TRAIN | Precision=0.598, Recall=0.636, F1=0.605, Accuracy=0.636 | Precision=0.560, Recall=0.605, F1=0.567, Accuracy=0.605 | Precision=0.625, Recall=0.654, **F1=0.595**, Accuracy=0.654 |
| 3.a.i | SVM | SVC | Train3 | Precision=0.583, Recall=0.641, F1=0.581, Accuracy=0.641 | Precision=0.610, Recall=0.641, F1=0.567, Accuracy=0.641 | Precision=0.649, Recall=0.660, **F1=0.588**, Accuracy=0.660 |
| 3.a.ii | SVM | SVC | TRAIN | Precision=0.634, Recall=0.673, F1=0.641, Accuracy=0.673 | Precision=0.634, Recall=0.666, F1=0.629, Accuracy=0.666 | Precision=0.688, Recall=0.690, **F1=0.655**, Accuracy=0.690 |
| 4.a.i | XGBoost | XGBClassifier | Train3 | Precision=0.596, Recall=0.643, F1=0.598, Accuracy=0.643 | Precision=0.579, Recall=0.632, F1=0.572, Accuracy=0.632 | Precision=0.622, Recall=0.654, **F1=0.604**, Accuracy=0.654 |
| 4.a.ii | XGBoost | XGBClassifier | TRAIN | Precision=0.594, Recall=0.661, F1=0.601, Accuracy=0.661 | Precision=0.626, Recall=0.652, F1=0.587, Accuracy=0.652 | Precision=0.678, Recall=0.670, **F1=0.610**, Accuracy=0.670 |
