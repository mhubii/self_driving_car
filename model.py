import torch
import torch.nn as nn


class CNN(nn.Module):
    """
        Convolutional neural net for behavioural cloning.
    """
    def __init__(self, input_shape, batch_size):
        """
            Initialize the CNN.
        """
        super(CNN, self).__init__()
        self.batch_size = batch_size
        self.features = nn.Sequential(
            nn.Conv2d(3, 8, 5, 2),
            nn.ReLU(),
            nn.Conv2d(8, 16, 5, 2),
            nn.ReLU(),
            nn.Conv2d(16, 32, 3, 2),
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, 2),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, 2),
            nn.ReLU()
        )

        n = self._get_conv_output(input_shape)

        self.classification = nn.Sequential(
            nn.Linear(n, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 8),
            nn.ReLU(),
            nn.Linear(8, 1)
        )

    def _get_conv_output(self, shape):
        """
            Determine the dimension of the feature space.
        """
        # Unsqueeze to obtain 1x(shape) as dimensions.
        input = torch.rand(shape).unsqueeze(0)
        input = torch.autograd.Variable(input)
        output = self.features(input)
        n = output.numel()
        return n

    def forward(self, input):
        """
            Forward through the CNN.
        """
        # Convolutional layers for feature extraction.
        output = self.features(input)

        # Flatten.
        output = output.view(self.batch_size, int(output.numel()/self.batch_size))

        # Linear layers for classification.
        output = self.classification(output)
        return output