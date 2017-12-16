import pandas as pd
import numpy as np
from torch.utils.data import Dataset
import cv2
import os

IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANNELS = 64, 64, 3
INPUT_SHAPE = (IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANNELS)


class DataSetGenerator(Dataset):
    def __init__(self, data_dir, transform=None):
        """
            Load all image paths and steering angles
            from the .csv file.
        """
        self.data_dir = data_dir
        self.image_paths, self.steering_angles = load_data(data_dir)
        self.transform = transform

    def __getitem__(self, index):
        """
            Get an image and the steering angle by index.
            Outputs the adjusted steering angle.
        """
        img, steering_angle = select_image(self.data_dir,
                                           self.image_paths[index],
                                           self.steering_angles[index])

        sample = {'img': img, 'steering_angle': steering_angle}

        if self.transform is not None:
            sample = self.transform(sample)

        return sample

    def __len__(self):
        """
            Return the lenght of the whole data set
        """
        return self.steering_angles.shape[0]


class PreProcessData(object):
    """
        Pre-process the data.
    """
    def __call__(self, sample):
        img, steering_angle = sample['img'], sample['steering_angle']

        img = crop(img)
        img = resize(img)

        return {'img': img, 'steering_angle': steering_angle}


class AugmentData(object):
    """
        Augment data and apply some simple checks.
    """
    def __call__(self, sample):
        img, steering_angle = sample['img'], sample['steering_angle']

        img = random_brightness(img)
        img, steering_angle = random_flip(img, steering_angle)

        return {'img': img, 'steering_angle': steering_angle}


def load_data(data_dir):
    """
        Loads the input data and separates it into image_paths
        and steering_angles.

    :return:
        image_paths: np.ndarray
                     Location of recorded images.
        labels: float/rad
                Steering angle.
    """
    data_df = pd.read_csv(os.path.join(os.getcwd(),
                          data_dir, 'driving_log.csv'),
                          names=['center', 'left', 'right',
                                 'steering', 'throttle',
                                 'break', 'velocity'])

    image_paths = data_df[['center', 'left', 'right']].values
    steering_angles = data_df['steering'].values
    return image_paths, steering_angles


def select_image(data_dir, image_file, steering_angle):
    """
        Load an image and adjust the steering angle accordingly
        to the location of the camera (center, left or right).

    :return:
        image: np.array
               RGB values of the image.
        steering_angle: float/rad
                        Steering angle corresponding to image.
    """
    img = load_image(data_dir, image_file)

    if 'left' in image_file:
        steering_angle += 0.25
    elif 'right' in image_file:
        steering_angle -= 0.25

    return img, steering_angle


def load_image(data_dir, image_file):
    """
        Load RGB image from a file.
    """
    return cv2.imread(os.path.join(os.getcwd(), data_dir, image_file))


def crop(image):
    """
        Crop of the sky and parts of the care since
        it does not add useful information for the
        training.
    """
    return image[65:, :]


def resize(image):
    """
        Resize the image size to the input format of
        the network.
    """
    return cv2.resize(image, dsize=(IMAGE_WIDTH, IMAGE_HEIGHT))


def random_brightness(image):
    """
        Adjust brightness of image randomly by applying
        a gamma correction
            image = image^(1/gamma)
        randomly.
    """

    gamma = np.random.random_sample() + 0.5
    inv_gamma = 1/gamma

    # Build a lookup table mapping the pixel values
    # [0, 255] to their adjusted gamma values.
    table = np.array([((i/255.)**inv_gamma)*255
                      for i in range(256)]).astype('uint8')
    return cv2.LUT(image, table)


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
        return cv2.flip(image, 1), -steering_angle
    else:
        return image, steering_angle


