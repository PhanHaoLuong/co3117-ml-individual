Dataset:
UCI Human Activity Recognition Using Smartphones
Version: 1.0

Task:
Multiclass classification of human activity.

Target:
Activity label.

Primary metric:
Macro-F1.

Secondary metrics:
Accuracy
Confusion matrix

Data split:
- The original dataset has already been split into a 70/30 split. 70% of volunteers were selected for training data and the other 30% for testing data
- Due to that initial split, we'll keep the original 30% untouched but we'll further split the 70% into training and validation data.
- With that being said, there would be 21 train subjects and 9 test subjects. In those 21 train subjects, 17 would be used for training and the 4 left would be used for validation.

Validation subjects:
2, 9, 14, 19

Training subjects:
1, 3, 4, 5, 6, 7, 8, 10, 11, 12, 13, 15, 16, 17, 18, 20, 21

Test set:
The original 9 test subjects will remain untouched during model development and will only be used for the final evaluation

Random seed:
42