# Interpolation-ResNet
Invited paper for WSA 2021

The paper is 'Low Complexity Channel estimation with Neural Network Solutions'.

Low complexity residual convolutional neural network for channel estimation

Conpared with the ReEsNet from the repo Residual_CNN, it has slightly improved performance and the number of parameters is reduced by 82% (pruningis not applied)

Yes, I use the neural network for both time interpolation and frequency interpolation, and for denoising.

Almost all the papers are really masters when investigating the research, because using the complete channel matrix, even the channel matrix on the data symbols, for training. We use a data-driven method and even process the channel matrix of the data symbols in? Everyone know that is trash and impractical compeletely, but no one stands up and says you have no cloths on. How can we achieve online training and if my memory is correct each neural network must have a fix output at least a fixed output size? We are celebrating the unprecedented success we got by training with the complete channel matrix, compared with conventional methods which can access channel matrix at the reference signal. Should I ask if we have already know the channel matrix of complete package, why we estimate it? solar flashlight? If we have already known the channel matrix at the data symbols then we can train the neural network to estimate it, if not then we cannot? It reminds me of the bitcoin, no one really cares if they did is correct. We neither, as long as can publish paper on top journal/conference. I can understand some people might submit papers but get rejection because they said "oh why u dont use channel matrix of complete package and it seems like the proposed neural network can not achieve a LMMSE performance lol, you need to do something like us. Althrough that is not practical for training with data channel matrix, it overkills.". If you use MSE as loss function, have a careful think on why you can achieve a LMMSE performance and even outperforms LMMSE because LMMSE method with perfect channel knowledge hold the global optimization of MSE loss. A neural network can achieve global minimum or even lower than global minimum? Wow...It is amazing that I dont really understand what is global minimum. Or a neural network without published code? Or both? What I want to mean is that,

For the people, who with name, and without. For the real engineers.
