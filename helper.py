"""
    Helper functions.
"""

import pandas as pd
import numpy as np
import cv2
import os

IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANNELS = 66, 200, 3
INPUT_SHAPE = (IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANNELS)


def batch_generator():
    """
        Generates batches of samples.

    :return:
        (x, y): x = images, y = steering angle
                Generated tuples for the fit_generator().
    """


def load_data(args):
    """
        Loads the input data and separates it into the images X
        and the steering angle y.

    :return:
        X: [center, left, right]
           Location of recorded images.
        y: float
           Steering angle.
    """
    data_df = pd.read_csv(os.path.join(os.getcwd(), args.data_dir, 'driving_log.csv'), names=['center', 'left', 'right',
                                                                                              'steering', 'throttle',
                                                                                              'break', 'velocity'])
    x = data_df[['center', 'left', 'right']].values
    y = data_df['steering'].values
    return x, y


def select_image(data_dir, left, center, right, steering_angle):
    """
        Randomly select an image from left, center or right. Steering
        angle, corresponding to images which were taken from the
        off-center, need to be corrected.

    :return:
        image: array
               RGB values of the image.
        steering_angle: rad
                        Steering angle corresponding to image.
    """
    choice = np.random.choice(3)
    if choice == 0:
        return load_image(data_dir, left), steering_angle + 0.2
    elif choice == 1:
        return load_image(data_dir, center), steering_angle
    else:
        return load_image(data_dir, right), steering_angle - 0.2


def load_image(data_dir, image_file):
    """
        Load RGB image from a file.
    """
    return cv2.imread(os.path.join(data_dir, image_file.strip()))


def pre_process(image):
    image = crop(image)
    image = resize(image)
    return image


def crop(image):
    """
        Crop of the sky and parts of the care since
        it does not add useful information for the
        training.
    """
    return image


def resize(image):
    """
        Resize the image size to the input format of
        the network.
    """
    return cv2.resize(image, dsize=(IMAGE_WIDTH, IMAGE_HEIGHT))


def random_brightness():
    pass


def random_flip(image, steering_angle):
    """
        Flips an image as there mainly left turns
        on the training ground.

    :return:
        image: array
               Flipped image.
        steering_angle: rad
                        Sign reverted steering angle.
    """
    choice = np.random.choice(2)
    if choice == 0:
        return cv2.flip(image, 1), -1*steering_angle
    else:
        return image, steering_angle


def random_rotation():
    """
        Apply a random rotation to the input image.

    :return:
        image: array
               Rotated version of the input image.
        steering_angle: rad
                        Adjusted steering angle.
    """
    pass
