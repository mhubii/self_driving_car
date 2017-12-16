import torch
import torch.nn as nn


class CNN(nn.Module):
    """
        Convolutional neural net for behavioural cloning.
    """
    def __init__(self, input_shape):
        """
            Initialize the CNN.
        """
        super(CNN, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 5, 5, 2),
            nn.ReLU(),
            nn.Conv2d(5, 5, 5, 2),
            nn.ReLU(),
            nn.Conv2d(5, 5, 5, 2),
            nn.ReLU(),
            nn.Conv2d(5, 5, 5, 2),
            nn.ReLU(),
            nn.Conv2d(5, 5, 5, 2),
            nn.ReLU()
        )

        n = self._get_conv_output(input_shape)

        self.classification = nn.Sequential(
            nn.Linear(n, 100),
            nn.ReLU(),
            nn.Linear(100, 50),
            nn.ReLU(),
            nn.Linear(50, 10),
            nn.ReLU(),
            nn.Linear(10, 1)
        )

    def _get_conv_output(self, shape):
        """
            Determine the dimension of the feature space.
        """
        input = torch.autograd.Variable(torch.rand(shape))
        output = self.features(input)
        n = torch.numel(output)
        return n

    def forward(self, input):
        """
            Forward through the CNN.
        """
        # Convolutional layers for feature extraction.
        output = self.features(input)

        # Flatten.
        output = output.view(torch.numel(output))

        # Linear layers for classification.
        output = self.classification(output)
        return output