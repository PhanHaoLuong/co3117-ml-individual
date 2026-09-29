## W6 - Tuesday 29 September 2026 : Perceptron
- First naive implementation of Perceptron, the baseline after training and validating for 60 epochs with the learning rate of 0.1 is:
    Training Macro-F1: 0.9642, Training Accuracy: 0.9622, Training Confusion Matrix:
            [[1115    0    0    0    0    0]
             [   2  957   20    0    0    0]
             [   0    0  902    0    0    0]
             [   0    0    0 1113   46    0]
             [   0    0    0  182 1059    0]
             [   0    0    0    2    0 1271]]
    Validation Macro-F1: 0.8388, Validation Accuracy: 0.8624, Validation Confusion Matrix:
            [[ 39  59  13   0   0   0]
             [  3  91   0   0   0   0]
             [  0   3  81   0   0   0]
             [  0   0   0 116   8   3]
             [  0   0   0   3 130   0]
             [  0   0   0   2   0 132]]

- We then do a basic experiment with it, shuffling the order of the samples in each epoch using the random rng seed. We then get the result:
    Training Macro-F1: 0.9879, Training Accuracy: 0.9874, Training Confusion Matrix:
            [[1115    0    0    0    0    0]
             [   0  969   10    0    0    0]
             [   0    0  902    0    0    0]
             [   0    0    0 1139   20    0]
             [   0    0    0   54 1187    0]
             [   0    0    0    0    0 1273]]
    Validation Macro-F1: 0.8381, Validation Accuracy: 0.8579, Validation Confusion Matrix:
            [[ 46  54  11   0   0   0]
             [  3  91   0   0   0   0]
             [  0  12  72   0   0   0]
             [  0   0   0 114   1  12]
             [  0   0   0   2 131   0]
             [  0   0   2   0   0 132]]