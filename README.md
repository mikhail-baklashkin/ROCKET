A time series classification method presented in [Dempster et al: exceptionally fast and accurate time series classification using random convolutional kernels](https://arxiv.org/abs/1910.13051) is tested on [Ham](https://www.timeseriesclassification.com/description.php?Dataset=Ham) dataset. The results are compared to the reference test $accuracy = 0.7257.$

The idea of the method is to generate a feature vector based on a large number of random convolutions (which are freezed during training) and feed it to a linear classifier.

I considered 3 ways of generating random convolution weights: based on 1) normal distribution, 2) random choice out of $[-1, 1]$, and 3) random choice out of $[-1, 0, 1].$ The other parameters are identical to those in the original paper.
