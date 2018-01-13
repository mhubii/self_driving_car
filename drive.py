import argparse
import base64
from datetime import datetime
import os
import shutil

import numpy as np
import socketio
import eventlet
import eventlet.wsgi
from PIL import Image
from flask import Flask
from io import BytesIO

import torch
import utils
import model

sio = socketio.Server()
app = Flask(__name__)
prev_image_array = None

MAX_SPEED = 25
MIN_SPEED = 10

speed_limit = MAX_SPEED


@sio.on('telemetry')
def telemetry(sid, data):
    if data:
        # The current speed of the   car.
        speed = float(data["speed"])
        # The current image from the center camera of the car.
        imgString = data["image"]
        image = Image.open(BytesIO(base64.b64decode(imgString)))
        image_array = np.asarray(image)

        # Crop the image.
        image_array = utils.crop(image_array)
        image_array = utils.normalize(image_array)

        # Change HxWxC to CxHxW.
        image_array = np.swapaxes(image_array, 0, 2)
        image_array = np.swapaxes(image_array, 1, 2)

        # Convert to torch.autograd.Variable.
        image_array = torch.from_numpy(image_array).float().unsqueeze(0)
        image_array = torch.autograd.Variable(image_array)

        # Predict steering angle.
        steering_angle = model(image_array)
        steering_angle = steering_angle.data[0, 0]

        global speed_limit
        if speed > speed_limit:
            speed_limit = MIN_SPEED  # slow down
        else:
            speed_limit = MAX_SPEED

        throttle = 1.0 - steering_angle ** 2 - (speed / speed_limit) ** 2

        print('{} {} {}'.format(steering_angle, throttle, speed))

        send_control(steering_angle, throttle)

    else:
        # NOTE: DON'T EDIT THIS.
        sio.emit('manual', data={}, skip_sid=True)


@sio.on('connect')
def connect(sid, environ):
    print("connect ", sid)
    send_control(0, 0)


def send_control(steering_angle, throttle):
    sio.emit(
        "steer",
        data={
            'steering_angle': steering_angle.__str__(),
            'throttle': throttle.__str__()
        },
        skip_sid=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Remote Driving')
    parser.add_argument(
        'image_folder',
        type=str,
        nargs='?',
        default='',
        help='Path to image folder. This is where the images from the run will be saved.'
    )
    args = parser.parse_args()

    # Load the model.
    model = model.CNN(utils.INPUT_SHAPE, batch_size=1)
    model.load_state_dict(torch.load('train.pth'))

    if args.image_folder != '':
        print("Creating image folder at {}".format(args.image_folder))
        if not os.path.exists(args.image_folder):
            os.makedirs(args.image_folder)
        else:
            shutil.rmtree(args.image_folder)
            os.makedirs(args.image_folder)
        print("RECORDING THIS RUN ...")
    else:
        print("NOT RECORDING THIS RUN ...")

    # wrap Flask application with engineio's middleware
    app = socketio.Middleware(sio, app)

    # deploy as an eventlet WSGI server
    eventlet.wsgi.server(eventlet.listen(('', 4567)), app)
