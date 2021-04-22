# ES654-2021 Assignment 3

*Aditya Pusalkar* - *18110009*

------

Confusion Matrix
[[177.   0.   0.   0.   0.   1.   0.   0.   0.   0.]
 [  0. 177.   1.   0.   0.   0.   1.   0.   2.   1.]
 [  0.   1. 176.   0.   0.   0.   0.   0.   0.   0.]
 [  0.   0.   0. 178.   0.   0.   0.   1.   3.   1.]
 [  1.   0.   0.   0. 172.   0.   3.   1.   0.   4.]
 [  0.   0.   0.   0.   0. 176.   1.   0.   0.   5.]
 [  0.   1.   0.   0.   0.   1. 178.   0.   1.   0.]
 [  0.   0.   0.   0.   0.   0.   0. 177.   0.   2.]
 [  0.   6.   1.   0.   0.   2.   1.   0. 164.   0.]
 [  0.   1.   0.   1.   0.   2.   0.   0.   0. 176.]]

The easiest number to predict is 0 followed by 2, 7 and 6 (based F-score calculation), 8 and 9 are toughest to predict well.
From the confusion matrix we can see that 5 and 9 are the digits which get confused the most. Following them are the pairs 1-8 and 4-9.

From the PCA plot we can see that the numbers which are not confused with others are very distinctly represented by the plot; for example 0 or 6. Numbers like 8 and 9 which are confused the most are not distinctly represented and their plots overlap a lot indicating indecisiveness and hence inaccuracy. 

