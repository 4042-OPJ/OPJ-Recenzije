
# Annotation guideline:
Originally, implementation 1 used a 5-label system.

3-label system
|Label|Sentiment |
|--|--|
| 0 | Positive |
| 1 | Neutral |
| 2 | Negative |

  

# Hyperparameters
Hyperparameters were determined using GridSearchCV.

**Support Vector Machines**
C=10, gamma='scale', kernel='rbf', random_state=42

**XGBoost**
n_estimators=500, learning_rate=0.1, max_depth=5, random_state=42, gamma=0


# Additional notes:

Test-1 and Train-1 Labels originally used 1 for negative and 2 for positive. This has been corrected to match the sets from other groups.

After the label system was reworked from 5 labels to 3, the F1 scores lowered to an average of about 0.53 due to a significant imbalance in the label counts. Because of this, we added data from the Cro-FiReDa dataset to balance the sentiment label counts. This improved the F1 score for our data, but didn't have much affect on the data from Team 1 and Team 2 (when tested on Train-3) as that data isn't balanced.
Added a column to Test-3 and Train-3 that specifies whether the data was added from Cro-FiReDa or not; 0 signifies the data isn't from Cro-FiReDa, 1 signifies that it is.

Link to Cro-FiReDa:
https://github.com/cleopatra-itn/Cro-FiReDa-A-Sentiment-Annotated-Dataset-of-Film-Reviews
