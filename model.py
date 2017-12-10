"""
    Model of the behavioral cloning.
"""

import argparse
import helper
from helper import INPUT_SHAPE
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Conv2D, Flatten, Dense
from keras.optimizers import Adam


def main():
    """
        Load data, build  and train model.
    """

    # get command line arguments
    parser = argparse.ArgumentParser(description='Behavioral Cloning Training Program')
    parser.add_argument('-d', type=str, help='data directory', default='data', dest='data_dir')
    parser.add_argument('-t', type=float, help='train size fraction', default=0.8, dest='train_size')
    parser.add_argument('-l', type=float, help='learning rate', dest='learning_rate')
    parser.add_argument('-s', type=int, help='samples per epoch', dest='samples_per_epoch')
    parser.add_argument('-e', type=int, help='number of epochs', dest='epochs')

    args = parser.parse_args()

    # load the data
    x, y = helper.load_data(args)
    print(x)

    # split it into train and validation set
    x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=args.train_size, random_state=0)

    # build model
    model = build_model()

    # train model

    # save weights


def build_model():
    model = Sequential()
    model.add(Conv2D(filters=5, kernel_size=5, strides=(2, 2), activation='relu', input_shape=INPUT_SHAPE))
    model.add(Conv2D(filters=5, kernel_size=5, strides=(2, 2), activation='relu'))
    model.add(Conv2D(filters=5, kernel_size=5, strides=(2, 2), activation='relu'))
    model.add(Conv2D(filters=5, kernel_size=3, activation='relu'))
    model.add(Conv2D(filters=5, kernel_size=3, activation='relu'))
    model.add(Flatten())
    model.add(Dense(100, activation='relu'))
    model.add(Dense(50, activation='relu'))
    model.add(Dense(10, activation='relu'))
    model.add(Dense(1))

    model.summary()

    return model


def train_model(args, model, x_train, x_test, y_train, y_test):
    model.compile(loss='mean_squared_error', optimizer=Adam(lr=args.learning_rate))

    #model.fit_generator(batch_generator(),
    #                    args.samples_per_epoch,
    #                    args.epochs)

if __name__ == '__main__':
    main()
