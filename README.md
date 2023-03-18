# Interpolation-ResNet
Invited paper for WSA 2021, called 'Low Complexity Channel estimation with Neural Network Solutions'. 

Low complexity residual convolutional neural network for channel estimation

Conpared with the ReEsNet from the repo Residual_CNN, it has slightly improved performance and the number of parameters is reduced by 82% (when pruning is not applied). I planed to release the code when I sorted out the files. 

Run Demonstration_of_H_regression_48_CommuRayleigh.m to test. 

Run ResNN_pilot_regression.m for training the neural network. 

One thing you need to keep in mind is that, the pruning is applied without retraining because it aims to minimize the computations/latency for low-complexity solutions. Typically, after pruning there should be complex procedures (Learning both weights and connections for efficient neural networks, arXivpreprint) to compensate, but that is not realistic for low-latency and low-complexity online implementation. For sure, that pruning degrades the performance significantly. I just remove the retraining and feedback loop.

The authors gratefully acknowledge the funding of this research by Huawei.
