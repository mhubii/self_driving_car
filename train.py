import argparse
import torch
import torch.nn as nn
from torchvision import transforms
from torch.utils.data import DataLoader
from torch.autograd import Variable
from model import CNN
import utils


def train():
    """
        Load data, build  and train model.
    """

    # Get command line arguments.
    parser = argparse.ArgumentParser(description='Behavioral Cloning Training Program')
    parser.add_argument('-d', type=str,   help='data directory',   default='data', dest='data_dir')
    parser.add_argument('-l', type=float, help='learning rate',    default=0.001,  dest='learning_rate')
    parser.add_argument('-b', type=int,   help='batch size',       default=20000,  dest='batch_size')
    parser.add_argument('-e', type=int,   help='number of epochs', default=10,     dest='epochs')

    args = parser.parse_args()

    # Load, pre-process and augment data.
    data_set = utils.DataSetGenerator(data_dir=args.data_dir,
                                      transform=transforms.Compose([
                                          utils.PreProcessData,
                                          utils.AugmentData,
                                          transforms.ToTensor()
                                      ]))

    # Data loader for batch generation.
    data_loader = DataLoader(data_set, batch_size=args.batch_size)

    # Build model.
    model = CNN(utils.INPUT_SHAPE)

    # Loss and optimizer.
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)

    # Train model.
    for epoch in range(args.epochs):
        for idx, sample in enumerate(data_loader):
            img = Variable(sample['img'])
            steering_angle = Variable(sample['steering_angle'])
            optimizer.zero_grad()
            steering_angle_out = model(img)
            loss = criterion(steering_angle, steering_angle_out)
            loss.backward()
            optimizer.step()

    # Save weights.
    torch.save(model.state_dict(), 'train.pk1')


if __name__ == '__main__':
    train()