import torch
import torch.nn as nn


class CNN(nn.Module):
    """Convolutional neural net."""
    def __init__(self):
        super(CNN, self).__init__()
        self.c1 = nn.Conv2d(in_channels=1, out_channels=5, kernel_size=5, stride=2)
        self.c2 = nn.Conv2d(in_channels=5, out_channels=5, kernel_size=5, stride=2)
        self.c3 = nn.Conv2d(in_channels=5, out_channels=5, kernel_size=5, stride=2)
        self.c4 = nn.Conv2d(in_channels=5, out_channels=5, kernel_size=5, stride=2)
        self.c5 = nn.Conv2d(in_channels=5, out_channels=5, kernel_size=5, stride=2)
        self.fc1 = nn.Linear(in_features=())
        self.fc2 = nn.Linear()
        self.fc3 = nn.Linear()

    def forward(self, *input):




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
