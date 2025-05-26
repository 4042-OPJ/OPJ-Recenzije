# Models & Hyperparameters

Main models:
**LogisticRegression** >>> C=10, penalty='l2', solver='lbfgs', random_state=42
**MultinomialNB** >>> alpha=0.1

Other models:
**RidgeClassifier** >>> alpha=0.8, solver='auto'
**LinearRegression** >>> default hyperparameters
**DecisionTreeClassifier** >>> criterion='entropy', max_depth=None, min_samples_split=0.5
**SGDClassifier** >>> alpha=0.0001, loss='hinge', penalty=None
**SVC** >>> C=10, gamma='scale', kernel='rbf', random_state=42
**XGBClassifier** >>> n_estimators=500, learning_rate=0.1, max_depth=3, random_state=42, gamma=0.1

Linear models have shown best results, as well as Multinomial Naive Bayes.
  
