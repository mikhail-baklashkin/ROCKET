import numpy as np

import torch
import torch.nn as nn


class FeatureExtractor(nn.Module):
    """Applies randomly generated convolutions, then applies global max pooling &
        extracts the proportion of positive values in feature maps in parallel.
        The convolutions are only generated on init.
    """

    def __init__(self, input_size, num_conv, kernel_sampling_type):
        """Initializes a FeatureExtractor object.

            Args:
                input_size: input size (length of the feature vector).
                num_conv: number of random convolutions.
                kernel_sampling_type: specifies the distribution to sample
                    kernel weights from, must be one of "normal" (standard normal distribution,
                    then centered), "binary" (random choice from [-1, 1]), or "tertiary" 
                    (random choice from [-1, 0, 1]).
        """
        super(FeatureExtractor, self).__init__()
        # for initialization
        self.input_size = input_size
        self.sampling_function = self._get_sampling_function(kernel_sampling_type)

        self.convolutions = self._generate_random_convolutions(num_conv)

    def forward(self, input):
        """Applies feature extractor to input.
        
            Args:
                input: input tensor.

            Returns:
                output tensor.
        """
        with torch.no_grad():
            # apply convolutions
            output = [conv(input) for conv in self.convolutions]
            output = [feature_map.reshape(feature_map.shape[0], feature_map.shape[-1])
                      for feature_map in output]
            
            # extract global max & proportion of positive values 
            output = [(
                feature_map.max(axis=1).values,
                (feature_map >= 0).sum(axis=1) / feature_map.shape[-1]
            ) for feature_map in output]

            # concatenate
            output = [torch.dstack((max_pool, ppv)) for max_pool, ppv in output]
            output = [features.reshape((features.shape[1], features.shape[0], features.shape[-1])) for features in output]
            output = torch.cat(output, dim=-1)
            return output
    
    def _get_sampling_function(self, kernel_sampling_type):
        """Returns a sampling function based on specified kernel sampling type.
        
            Args:
                kernel_sampling_type: specifies the distribution to sample
                    kernel weights from, must be one of "normal" (standard normal distribution,
                    then centered), "binary" (random choice from [-1, 1]), or "tertiary" 
                    (random choice from [-1, 0, 1]).
            
            Returns:
                lambda function.
        """
        if kernel_sampling_type == "normal":
            return lambda x: np.random.normal(0, 1, x)
        if kernel_sampling_type == "binary":
            return lambda x: np.random.choice([-1, 1], x)
        if kernel_sampling_type == "tertiary":
            return lambda x: np.random.choice([-1, 0, 1], x)

        message = f"kernel_sampling type must be one of\
            'normal', 'binary', 'tertiary'; got {kernel_sampling_type} instead."
        raise ValueError(message)

    def _generate_random_convolutions(self, num_conv):
        """Generates a specified number of random kernels.

            Args:
                num_conv: number of convolutions to generate.

            Returns:
                list of convolutions.
        """
        params = self._generate_random_params(num_conv)
        convolutions = [
            nn.Conv1d(1, 1, kernel_size=kernel_size, padding=padding, dilation=dilation) for
            kernel_size, padding, dilation in params
        ]

        with torch.no_grad():
            for conv in convolutions:
                weight = self.sampling_function(conv.kernel_size)
                weight = weight - weight.mean()
                bias = np.random.uniform(-1, 1, 1)

                conv.weight.copy_(torch.from_numpy(weight).float())
                conv.bias.copy_(torch.from_numpy(bias).float())
            
        return convolutions
    
    def _generate_random_params(self, num_conv):
        """Generates random parameters for a specified number of kernels.

            Args:
                num_conv: number of convolutions to generate.

            Returns:
                np.ndarray of kernel parameters.
        """
        kernel_sizes = np.random.choice((7, 9, 11), num_conv)
        dilations = np.asarray([2 ** np.random.uniform(0, np.log2((self.input_size - 1) / (kernel_size - 1)))
                     for kernel_size in kernel_sizes])
        paddings = np.asarray([((kernel_size - 1) * dilation) // 2 if np.random.randint(2) else 0
                    for kernel_size, dilation in zip(kernel_sizes, dilations)])
        
        return np.dstack([kernel_sizes, paddings, dilations])[0].astype(int)
