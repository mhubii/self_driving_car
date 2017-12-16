import argparse
from torchvision import transforms
from torch.utils.data import DataLoader
import model
import utils


def train():
    """
        Load data, build  and train model.
    """

    # Get command line arguments.
    parser = argparse.ArgumentParser(description='Behavioral Cloning Training Program')
    parser.add_argument('-d', type=str, help='data directory', default='data', dest='data_dir')
    parser.add_argument('-l', type=float, help='learning rate', dest='learning_rate')
    parser.add_argument('-b', type=int, help='batch size', dest='batch_size')
    parser.add_argument('-e', type=int, help='number of epochs', dest='epochs')

    args = parser.parse_args()

    # Load, pre-process and augment data.
    data_set = utils.DataSetGenerator(data_dir=args.data_dir,
                                      transform=transforms.Compose([
                                          utils.PreProcessData,
                                          utils.AugmentData
                                      ]))

    # Data loader for batch generation.
    data_loader = DataLoader(data_set, batch_size=args.batch_size,)

    # Build model.
    model = model.CNN()

    # train model

    # save weights


if __name__ == '__main__':
    train()
