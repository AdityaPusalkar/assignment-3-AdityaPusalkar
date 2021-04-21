# ES654-2021 Assignment 3

*Aditya Pusalkar* - *18110009*

------

** Time and Space Complexity while training:
--> Time Complexity of Logistic Regression while training is O(ND) where N is the number of samples and D is the dimension of the input data. in Logistic regression we have to find a vector w and bias b such that it will help us classify the input data into two categories (2 class problem). These quantities are trained by going over each input data point and updating according to the update rules. This take time complexity O(ND).
--> Space Complexity of Logistic Regression while training is O(ND). We need to store the input data, that is the feature vector and the output vector (X and y), as well create the vector w and bias b. This takes O(ND + N + D) space.

** Time and Space Complexity while predicting:
--> Time Complexity of Logistic Regression while predicting is O(D). The test vector will be multiplied with the vector w and bias b will be added. Then based the sign of this quantity we classify it into either of the 2 classes. This multiplication takes time O(D).
--> Space Complexity of Logistic Regression while training is O(D). Once we train the model the only quantity we need is the vector w and the bias b. This takes space O(D) + O(1) = O(D).

Note: If we have a multi-class problem then the space and time will be multipliedd by a factor of O(K); where K is the number of classes in the input data. This complexity increase is because of the training of a separate vector w and bias b for each of the classes in the input data. Hence a O(K) increase. 

