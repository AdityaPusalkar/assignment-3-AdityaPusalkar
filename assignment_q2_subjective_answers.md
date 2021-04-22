# ES654-2021 Assignment 3

*Aditya Pusalkar* - *18110009*

------

** Noting down the top 3 best regularizing coefficients based on accuracy:
L1 Regularisation
[0.97894, 0.97894, 0.97894]
[1.15, 1.1, 1.05]
L2 Regularisation
[0.97894, 0.97894, 0.97894]
[0.15, 0.1, 0.05]

Important features using L1 regularisation
The final theta is for lambda = 3.5
The values which are in power e+0 for this lambda are the important features. The more important the feature the higher its coeffiecient will be. The features which are not that important will gradually vanish away and tend towards 0 (indicated by the graph plotted).

[ 5.58012037e+00 -2.61208308e-01 -3.19326602e-02 -3.86459659e-01
 -2.00672757e-01  5.64252387e-03 -4.52034820e-03 -6.86922521e-01
 -2.99195513e+00  1.73639932e-02  2.05562395e-02  5.65521872e-03
  2.18061833e-02 -6.20046488e-03  1.34531518e-02  7.73291915e-03
  1.77978531e-02  1.35637405e-02 -7.66927683e-03 -1.66886395e-02
  2.32954817e-02 -2.58142663e+00 -1.63765710e+00 -2.16304869e+00
 -1.00211519e+00  1.46065374e-02  4.08411413e-03 -9.00285892e-03
 -4.14687273e+00  1.00226902e-02  8.98581617e-03]

Thus the bias coefficient (theta[0]) and theta[8,21,22,23,24,28] are the most important features, after that theta[1,3,4,7] are the next most important features.

