import sys
import tkinter
from tkinter import filedialog
import os

import cv2
import keyboard
import matplotlib.pyplot as plt
import numpy as np
import pyautogui
from cv2.data import *
from keras.layers import *
from keras.models import *

root = tkinter.Tk()
root.withdraw()  # use to hide tkinter window
current_dir = os.getcwd()
rootdir = filedialog.askdirectory(parent=root, initialdir=current_dir,
                                  title='Please select a directory to train CNN')
# Checking if root dir is selected. If not None is returned.

if rootdir is None:
    sys.exit()

cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')

video_capture = cv2.VideoCapture(0)


def normalize(x):
    minn, maxx = x.min(), x.max()
    return (x - minn) / (maxx - minn)


def scan(image_size=(32, 32)):
    _, frame = video_capture.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    boxes = cascade.detectMultiScale(gray, 1.3, 10)
    if len(boxes) == 2:
        eyes = []
        for box in boxes:
            x, y, w, h = box
            eye = frame[y:y + h, x:x + w]
            eye = cv2.resize(eye, image_size)
            eye = normalize(eye)
            eye = eye[10:-10, 5:-5]
            eyes.append(eye)
        return (np.hstack(eyes) * 255).astype(np.uint8)
    else:
        return None


width, height = 1919, 1079

check_model = os.path.isfile(rootdir + '/iris_mouse.h5')

if not check_model:
    opersys = pyautogui.prompt("What operating system are you using? \nWindows/Linux/Mac: ", 'OS')
    if opersys is None:
        sys.exit()
    if opersys == "Windows":
        root = rootdir + "\\"
    elif opersys == "Mac":
        root = rootdir + "/"
    else:
        root = rootdir + "/"

    filepaths = os.listdir(root)
    X, Y = [], []
    # Split the filepath to x, y, and unwanted jpeg end.
    # Then use x and y
    for filepath in filepaths:
        x, y, _ = filepath.split(' ')
        x = float(x) / width
        y = float(y) / height
        X.append(cv2.imread(root + filepath))
        Y.append([x, y])
    X = np.array(X) / 255.0
    Y = np.array(Y)
    # print(X.shape, Y.shape)

    pyautogui.alert('To exit the mouse system, press shift and x', 'Press Shift and x to exit.')
    # Build the model
    model = Sequential()
    model.add(Conv2D(32, 3, 2, activation='relu', input_shape=(12, 44, 3)))
    model.add(Conv2D(64, 2, 2, activation='relu'))
    model.add(Flatten())
    model.add(Dense(32, activation='relu'))
    model.add(Dense(2, activation='sigmoid'))
    model.compile(optimizer="adam", loss="mean_squared_error")
    # model.summary()

    # Train the model and store the history object
    history = model.fit(X, Y, epochs=210, batch_size=32, verbose=0)
    # Get the training loss from the history object
    train_loss = history.history['loss']

    '''
    epochs = 210
    
    for epoch in range(epochs):
        model.fit(X, Y, batch_size=32)
    '''
    # Plot the training loss
    plt.plot(train_loss, label='Training Loss')

    # Add title and labels
    plt.title('Training Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')

    # Show the plot
    plt.show()
    # larger N value == more smoothing == more lag.
    # Model did not exist so we reached here.
    # Since it did not exist, we will be saving it in the cwd.
    model.save(rootdir + '/iris_mouse.h5')

else:
    # Since model is found, we will load it...
    pyautogui.alert('Model exists. Not training model.', 'Model found!. To exit press Shift and x')
    model = load_model(rootdir + '/iris_mouse.h5')
N = 3  # size of the filter scale
weights = [0.1, 0.3, 0.6]  # weights for the weighted average
x_buffer = [0] * N
y_buffer = [0] * N

while True:

    if keyboard.is_pressed('shift') and keyboard.is_pressed('x'):
        break
    eyes = scan()
    if not eyes is None:
        eyes = np.expand_dims(eyes / 255.0, axis=0)
        x, y = model.predict(eyes, verbose=0)[0]

        # add new values to buffer
        x_buffer.append(x)
        y_buffer.append(y)

        # remove oldest values from buffer
        x_buffer.pop(0)
        y_buffer.pop(0)

        # compute weighted average of coordinates
        x_smooth = sum(w * x for w, x in zip(weights, x_buffer))
        y_smooth = sum(w * y for w, y in zip(weights, y_buffer))

        pyautogui.moveTo(x_smooth * width, y_smooth * height)

        pyautogui.FAILSAFE = False

    else:
        pyautogui.alert('Your lighting conditions are bad. Make sure there is no source of light behind you, '
                        'but lots of light in front of you.')
