import hashlib
import os
import random
import tkinter
from tkinter import filedialog

import cv2
import numpy as np
import pyautogui
import pygame
from cv2.data import *

pyautogui.alert('This is a highly elevated event. Do not edit the training data without reason. '
                'This can lead'
                'to unwanted changes in your system. For your security, password will be required to continue',
                'Warning!')
# password = '1r1s@PRAnav@)!)'
password_os_256 = os.environ.get('PASSWORD_FOR_IRISMOUSE')

input_pwd = pyautogui.password(text='(ssh) enter password: ', title='Password', default='', mask='*')

if input_pwd is None:
    exit()

if password_os_256 == hashlib.sha256(input_pwd.encode()).hexdigest():
    pyautogui.alert('CNN training activated.')

    root = tkinter.Tk()
    root.withdraw()  # use to hide tkinter window
    current_dir = os.getcwd()
    rootdir = filedialog.askdirectory(parent=root, initialdir=current_dir, title='Please select a directory to store images')
    # Checking if root dir is selected. If not None is returned.

    if rootdir is None:
        exit()
    rootdir = rootdir + '/'
    if os.path.isdir(rootdir):
        print('ok')
        print('dir already exists. mkdir invalid, continuing...')
        model_path = rootdir + 'iris_mouse.h5'
        if os.path.exists(rootdir + 'iris_mouse.h5'):
            os.remove(model_path)
            pyautogui.alert('An existing model file was found. It was deleted.', 'Existing Model Found')

    else:
        print("dir does not exist. mkdir valid, continuing...")
        os.mkdir(rootdir)
else:
    print('incorrect')
    exit()

# Initialize Pygame
pygame.init()

# Set screen size

pygame.init()

info = pygame.display.Info()
w = info.current_w
h = info.current_h


screen = pygame.display.set_mode((0, 0), pygame.FULLSCREEN, pygame.RESIZABLE)

# Set dot color and position
dot_color = (255, 0, 0)

dot_pos = [249, 401]

# Set dot speed
dot_speed = 5

# Set direction
direction = [1, 1]


# Normalization helper function
def normalize(x):
    minn, maxx = x.min(), x.max()
    return (x - minn) / (maxx - minn)


# Eye cropping function
def scan(image_size=(32, 32)):
    _, frame = video_capture.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    boxes = cascade.detectMultiScale(gray, 1.3, 10)
    if len(boxes) == 2:
        eyesl = []
        for box in boxes:
            x, y, w, h = box
            eye = frame[y:y + h, x:x + w]
            eye = cv2.resize(eye, image_size)
            eye = normalize(eye)
            eye = eye[10:-10, 5:-5]
            eyesl.append(eye)
        return (np.hstack(eyesl) * 255).astype(np.uint8)
    else:
        return None


# Load Haar cascade classifier
cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')

# Open video capture
video_capture = cv2.VideoCapture(0)

# Set running flag
running = True

# Start loop
while running:
    # Handle events
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        elif event.type == pygame.KEYDOWN:
            if event.key == pygame.K_x:
                running = False

    dot_pos[0] = random.randint(0, screen.get_width())
    dot_pos[1] = random.randint(0, screen.get_height())

    # Clear screen
    screen.fill((0, 0, 0))

    # Draw dot
    pygame.draw.circle(screen, dot_color, dot_pos, 10)

    # Update display
    pygame.display.flip()

    # Wait for a half second so eyes can adjust.
    pygame.time.wait(950)
    # Capture eyes image
    eyes = scan()

    # If eyes were detected
    if not eyes is None:
        x_dot = dot_pos[0]
        y_dot = dot_pos[1]
        print('eyes')  # Print 'eye' if an eye is detected.
        # Save image to disk. Add 'eyes' so it can be split later.
        filename = rootdir + "{} {} eyes.jpeg".format(x_dot, y_dot, rootdir)
        cv2.imwrite(filename, eyes)

# Quit Pygame and release video capture
pygame.quit()
video_capture.release()
