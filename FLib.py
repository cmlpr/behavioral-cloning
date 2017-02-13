import sys
import os
from six.moves.urllib.request import urlretrieve
import zipfile
import pandas as pd
import cv2
import numpy as np
import matplotlib.pyplot as plt

from keras.models import Sequential
from keras.layers.core import Lambda, Dense, Activation, Flatten, Dropout
from keras.layers.convolutional import Convolution2D
from keras.layers.pooling import MaxPooling2D
from keras.optimizers import Adam

# Gathering Data
#######################################################################################################################

last_percent_reported = None


def download_progress_hook(count, block_size, total_size):
    """
    A hook to report the progress of a download. This is mostly intended for users with
    slow internet connections. Reports every 5% change in download progress.
    :param count:
    :param block_size:
    :param total_size:
    :return:
    """

    global last_percent_reported
    percent = int(count * block_size * 100 / total_size)

    if last_percent_reported != percent:
        if percent % 5 == 0:
            sys.stdout.write("%s%%" % percent)
            sys.stdout.flush()
        else:
            sys.stdout.write(".")
            sys.stdout.flush()

        last_percent_reported = percent


def download_data(url, file_name, expected_bytes, force=False):
    """
    Download a file if not present, and make sure it's the right size.
    :param url:
    :param file_name:
    :param expected_bytes:
    :param force: by default is it False
    :return:
    """

    if force or not os.path.exists(file_name):
        print('Attempting to download:', file_name)
        file_name, _ = urlretrieve(url + file_name, file_name, reporthook=download_progress_hook)
        print('\nDownload Complete!')
    if not os.path.exists(file_name):
        raise Exception(
            'Failed to verify ' + file_name + '. Can you check the link or file name?')
    stat_info = os.stat(file_name)
    if stat_info.st_size == expected_bytes:
        print('Found and verified', file_name)
    else:
        raise Exception(
            'Failed to verify ' + file_name + '. Can you get to it with a browser?')
    return file_name


def extract_data(file_name, force=False):
    """
    This function extracts the data if it is not already extracted
    Optionally you can force the function to overwrite the current extraction
    :param file_name:
    :param force:
    :return:
    """

    # Remove file extension (.zip)

    root = os.path.splitext(file_name)[0]

    if os.path.isdir(root) and not force:
        # You may override by setting force=True.
        print('%s already present - Skipping extraction of %s.' % (root, file_name))
    else:
        print('Extracting data for %s. This may take a while. Please wait.' % root)
        zip_ref = zipfile.ZipFile(file_name, 'r')
        sys.stdout.flush()
        zip_ref.extractall()
        zip_ref.close()
        print('\nExtraction Complete!')

    # Find the file with extension csv to assign as the log file
    log_file = [file_loc for file_loc in os.listdir(root) if file_loc.endswith('.csv')]

    path_to_log = os.path.join(root, log_file[0])

    return path_to_log, root


def use_online_data(data_url, training_data, data_size):
    """
    When this function is called  - it downloads the data from a link that is provided,
    extracts the contents and reads the csv log file.
    :param data_url:
    :param training_data:
    :param data_size:
    :return:
    """

    # Download the training data from the link provided above
    _ = download_data(data_url, training_data, data_size)

    # Extract the data and get the paths for csv and image files
    log_path, img_path = extract_data(training_data)

    # Make sure the path to log file is correct
    print('Path to the CSV log file is : ', log_path)

    # Make sure the images path is correct
    print('Images are located in : ', img_path)

    # Read the CSV File
    img_list = pd.read_csv(log_path)

    # join the extract folder path with the image path in the log file
    img_list['center'] = img_list['center'].apply(lambda x: os.path.join(img_path, x.strip()))
    img_list['left'] = img_list['left'].apply(lambda x: os.path.join(img_path, x.strip()))
    img_list['right'] = img_list['right'].apply(lambda x: os.path.join(img_path, x.strip()))

    return img_path, img_list


def use_local_recording(img_path):
    """
    This function reads data from a local recording
    :param img_path:
    :return:
    """

    log_file = [file_loc for file_loc in os.listdir(img_path) if file_loc.endswith('.csv')]
    path_to_log = os.path.join(img_path, log_file[0])
    print(path_to_log)

    # Local recording doesn't have the csv header so we need to add it manually below
    img_list = pd.read_csv(path_to_log,
                           names=["center", "left", "right", "steering", "throttle", "brake", "speed"])
    return img_list


# Reading and plotting image - data exploration functions
#######################################################################################################################

def read_img(img_list, camera_position, array_position):
    """
    This function reads an image/steering angle from a list of images with given camera position and row position
    :param img_list:
    :param camera_position:
    :param array_position:
    :return:
    """
    # Note we need to strip left whitespace for left images - only a problem in downloaded file
    # global img_files, img_path
    if camera_position in ('center', 'left', 'right'):
        img = cv2.imread(os.path.join(img_list[camera_position][array_position].strip()))
        # Need to convert the images from bgr to rgb
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        # Also read the steering angle
        steer_angle = float(img_list['steering'][array_position])
        return img, steer_angle
    else:
        sys.exit('Wrong camera position defined in read_img')


def plot_random_image(img_list, camera_position, img_out_folder):
    """
    A plotting function for data exploration
    Randomly selects an image from a list of images (pandas data frame) with given camera position
    Saves a plot of this image in the output folder provided
    :param img_list:
    :param camera_position:
    :param img_out_folder:
    :return:
    """
    total_count = len(img_list)
    # randomly select an integer between 0 and total count of images
    img_number = int(np.random.randint(0, total_count, 1))
    # call the function to read images. we won't use the steering angle
    im, _ = read_img(img_list, camera_position, img_number)
    print("Plotting image #%s-%s image" % (img_number, camera_position))
    print('Shape of the image: ', im.shape)
    plt.axis("off")
    plt.imshow(im)
    plt.title('Random %s Image: %s' % (camera_position, img_number))
    fname = 'random_' + camera_position + '_image'
    plt.savefig(os.path.join(img_out_folder, fname))
    print('Completed plotting\n')
    plt.close()


def plot_random_image_set(img_list, img_out_folder):
    """
    This function randomly selects an image set (left, center and right) from a list of images
    and saves a plot of these into the output folder
    :param img_list:
    :param img_out_folder:
    :return:
    """
    total_count = len(img_list)
    img_number = int(np.random.randint(0, total_count, 1))
    fig, axes = plt.subplots(1, 3, figsize=(8, 3), subplot_kw={'xticks': [], 'yticks': []})
    fig.subplots_adjust(hspace=0.2, wspace=0.05)
    fig.suptitle(('Random Image Set: %s' % img_number), fontsize=15)
    # Subplot left image
    axes[0].imshow(read_img(img_list, 'left', img_number)[0])
    axes[0].set_title('left_camera', fontsize=10)
    # Subplot for center image
    axes[1].imshow(read_img(img_list, 'center', img_number)[0])
    axes[1].set_title('center_camera', fontsize=10)
    # Subplot for right image
    axes[2].imshow(read_img(img_list, 'right', img_number)[0])
    axes[2].set_title('right_camera', fontsize=10)
    plt.savefig(os.path.join(img_out_folder, 'random_image_set'))
    plt.close()
    print('Completed plotting left, center, right images for #%s\n' % img_number)


def plot_driving_data(img_list, img_out_folder):
    """
    Data set includes steering angle, throttle, brake and speed information
    It is a good practice to visualize them
    :param img_list:
    :param img_out_folder:
    :return:
    """
    plt.subplot(221)
    plt.plot(img_list['steering'])
    plt.ylabel('Steering Angle')
    plt.title('Steering Data')
    plt.grid(True)

    plt.subplot(222)
    plt.plot(img_list['throttle'])
    plt.ylabel('Throttle')
    plt.title('Throttle Data')
    plt.grid(True)

    plt.subplot(223)
    plt.plot(img_list['brake'])
    plt.ylabel('Brake')
    plt.title('Brake Data')
    plt.grid(True)

    plt.subplot(224)
    plt.plot(img_list['speed'])
    plt.ylabel('Speed')
    plt.title('Speed Data')
    plt.grid(True)

    plt.subplots_adjust(top=0.92, bottom=0.08, left=0.12, right=0.95, hspace=0.55,
                        wspace=0.35)
    plt.savefig(os.path.join(img_out_folder, 'drive_data'))
    plt.close()
    print('Completed plotting drive data\n')


def plot_steering_histogram(img_list, title, img_out_folder, steering_weights=None):
    """
    This function plots the histogram of the steering angle.
    There is an option to assign weights to each data point.
    :param img_list:
    :param title:
    :param img_out_folder:
    :param steering_weights:
    :return:
    """

    if steering_weights:
        plt.hist(img_list['steering'], weights=steering_weights, bins=40)
    else:
        plt.hist(img_list['steering'], bins=40)
    plt.title("Histogram of Steering Data")
    plt.xlabel("Value")
    plt.ylabel("Frequency")
    if steering_weights:
        plt.savefig(os.path.join(img_out_folder, title))
    else:
        plt.savefig(os.path.join(img_out_folder, title))
    plt.close()
    print('Completed plotting histogram of steering data\n')


def compare_2_images(img_no, camera_pos, left_img, right_img, title, img_out_folder):
    """
    This function plots two images side by side.

    :param img_no: integer
    :param camera_pos: string
    :param left_img: tuple (img, steering angle)
    :param right_img: tuple (img, steering angle)
    :param title: desired title and file name
    :param img_out_folder: output folder
    :return:
    """

    plt.figure(figsize=(7, 4))
    plt.subplot(121)
    plt.axis("off")
    plt.imshow(left_img[0])
    plt.title(('Before - Str: %.3f' % left_img[1]), fontsize=10)

    plt.subplot(122)
    plt.axis("off")
    plt.imshow(right_img[0])
    plt.title(('After - Str: %.3f' % right_img[1]), fontsize=10)
    plt.suptitle((title + ' - %s img: %s' % (camera_pos, img_no)), fontsize=15)
    plt.savefig(os.path.join(img_out_folder, title))
    plt.close()
    print('Completed comparison plot for: %s\n' % title)


# Data Augmentation Helper Functions
#######################################################################################################################

def normalize_scales(img):
    """
    Normalize images by subtracting mean and dividing by the range so that pixel values are between -0.5 and 0.5
    :param img:
    :return:
    """
    # normalized_image = np.divide(img - 125.0, 255.0)
    normalized_image = (img - 125.0) / 255.0
    return normalized_image


def adjust_brightness(img, bright_limit=(-0.5, 0.15)):
    """
    Adjust brightness of the image by randomly selecting from a uniform distribution between the limits provided
    The selected number will be added to 1 and multiplied with the V channel of the image
    Requires RGB to HSV conversion and then back to RGB conversion
    :param img:
    :param bright_limit: tuple needs to between -1 and 1
    :return:
    """
    # by default the lower limit is -0.5 and higher limit is 0.15
    brightness_multiplier = 1.0 + np.random.uniform(low=bright_limit[0], high=bright_limit[1])
    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    hsv[:, :, 2] = hsv[:, :, 2] * brightness_multiplier
    adjusted = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
    return adjusted


def crop_image(img, cut_info=(40, 25, 0, 0)):
    """
    Crops images from 4 edges with desired px quantities
    # By default cut_info will be set to 40px from top, 25 px from bottom, 0 px from left and 0px from right
    # Crop 40px from top for removing the horizon
    # Crop 25px from bottom for removing the car body
    :param img:
    :param cut_info: tuple of 4 integer pixel values - top, bottom, left, right order
    :return:
    """
    crop_top = cut_info[0]
    crop_bottom = cut_info[1]
    crop_left = cut_info[2]
    crop_right = cut_info[3]
    img_shape = img.shape
    cropped = img[crop_top:img_shape[0]-crop_bottom, crop_left:img_shape[1]-crop_right]
    return cropped


def resize_image(img, new_dim=(64, 64)):
    """
    If desired images can be resized using this function.
    # By default this function will reduce the image to 64px by 64px - first cols then rows
    # (new_cols, new_rows)
    :param img:
    :param new_dim: tuple of integer pixel values for desired columns and rows
    :return:
    """

    resized = cv2.resize(img, new_dim, interpolation=cv2.INTER_AREA)
    return resized


def flip_image(img, steering_angle):
    """
    Flips the image and changes the steering angle if the image is flipped.
    Flipping is random and based on binomial distribution (coin flip)
    :param img:
    :param steering_angle:
    :return:
    """
    flip_random = int(np.random.binomial(1, 0.5, 1))
    if flip_random:
        return cv2.flip(img, 1), -1.0 * float(steering_angle)
    else:
        return img, float(steering_angle)


def rotate_image(img, steering_angle, rot_info=(10.0, 0.0)):
    """
    Function to rotate an image. Rotation angle and center of rotation are based on random numbers
    The rot_angle tuple has the first constraint for the max rotation angle and second constraint for displacement
    from the middle of the image that will be used as the center of rotation.
    Steering angle is also adjusted based on the rotation
    Rotation is fixed to +25 and -25 so if the steering angle + selected random rotation angle are bigger than these
    numbers, rotation will only be performed by the amount that would bring us to the maximums
    By default the off-center displacement is set to 0 but it is possible to set it to some small values
    Setting it to large values is not desirable.
    For each rotation angle steering angle is also adjusted in the rotation direction
    :param img:
    :param steering_angle: tuple (max rotation, max off center distance)
    :param rot_info:
    :return:
    """

    act_steering_angle = float(steering_angle) * 25.0

    max_rotation_angle = rot_info[0]  # degrees
    max_center_translation = rot_info[1]  # pixels

    # Randomly pick a rotation angle
    angle = np.random.uniform(low=-max_rotation_angle, high=max_rotation_angle)

    # Check if the total angle is greater than 25 or smaller than -25. These are the max rotations possible
    # Then adjust the rotation angle
    if act_steering_angle + angle < -25.0:
        total_rotation = - 25.0 - act_steering_angle
    elif act_steering_angle + angle > 25.0:
        total_rotation = 25 - act_steering_angle
    else:
        total_rotation = angle

    # Update the steering angle by the rotation angle
    new_steering_angle = float(steering_angle) + (total_rotation / 25.0)

    rows, cols = img.shape[0:2]

    # Determine the center of rotation - it doesn't have to be rotated around the center of the image
    center = (cols / 2.0 + np.random.uniform(low=-max_center_translation, high=max_center_translation),
              rows / 2.0 + np.random.uniform(low=-max_center_translation, high=max_center_translation))
    # positive values in CCW, negative in CW, therefore multiply by -1
    rot_mat = cv2.getRotationMatrix2D(center, -total_rotation, 1.0)
    img_rotated = cv2.warpAffine(img, rot_mat, (cols, rows), flags=cv2.INTER_LINEAR)
    return img_rotated, new_steering_angle


def translate_image(img, steering_angle, trans_info=(40, 5)):
    """
    Function to translate the image in x and y directions.
    By default images will be translated up to 20x in x direction and 5px in y direction - (x, y)
    :param img:
    :param steering_angle: tuple (max x, max y)
    :param trans_info:
    :return:
    """

    rows, cols = img.shape[0:2]
    x_translation = np.random.uniform(low=-trans_info[0], high=trans_info[0])
    y_translation = np.random.uniform(low=-trans_info[1], high=trans_info[1])
    translation_matrix = np.float32([[1, 0, x_translation],
                                     [0, 1, y_translation]])
    img_trans = cv2.warpAffine(img, translation_matrix, (cols, rows))
    new_steering_angle = max(min(float(steering_angle) + (x_translation / trans_info[0]) * 0.25, 1.0), -1.0)
    return img_trans, new_steering_angle


# Helper functions for training and validation data generation
#######################################################################################################################

def get_weights(img_list):
    """
    Since data is biased toward zero steering angle images, using higher weights for non-zero angle images could be
    a good practice in training.
    This function assigns higher weights to higher steering angle images based on the absolute value of the steering angle
    One can modify the function so that weights are assigned based on the square root of the steering angle to increase
    the weights for the higher steering angle images
    A small value is added to all angles so that 0 steering angle images are not assigned 0 weights
    Weights add up to 1
    :param img_list:
    :return:
    """
    wghts = []
    steering_list = img_list['steering'].tolist()
    total = sum([abs(steer) + 0.05 for steer in steering_list])
    for steer in steering_list:
        str_angle = abs(steer)
        wghts.append((str_angle + 0.05)/total)
    return tuple(wghts)


def select_random_image(img_list, steering_weights, pr=(0.33, 0.34, 0.33)):
    """
    This functions selects a random image from a list of images
    Randomly picks a value between 0 and total image count with replacement
    Random choice function also uses the weights for each image to increase the probability of picking high-steering
    angle images
    After row location is picked, function randomly picks a camera position with the probability provided
    For left images steering angle is adjusted by +0.25 and for right images by -0.25
    :param img_list:
    :param steering_weights: tuple of weights for each row position (weights depend on steering angle)
    :param pr: tuple of weights for each camera position (left, center, right)
    :return:
    """
    total_count = len(img_list)
    # img_number = int(np.random.randint(0, total_count, 1))
    img_number = np.random.choice(total_count, 1, replace=True, p=steering_weights)[0]
    options_arr = ['left', 'center', 'right']
    camera_position = np.random.choice(options_arr, 1, p=pr)[0]
    im, steering_angle = read_img(img_list, camera_position, img_number)
    if camera_position == 'left':
        # Adjust left images by +0.25 to simulate recovery from left
        steering_angle = min(float(steering_angle) + 0.25, 1.0)
    elif camera_position == 'right':
        # Adjust right images by -0.25 to simulate recovery from right
        steering_angle = max(float(steering_angle) - 0.25, -1.0)
    else:
        pass
    return img_number, camera_position, im, steering_angle


def data_generator(case, param, img_list):
    """
    Data generator for training and validation sets
    Provides a python generator function for Keras with batches of data
    :param case: Training or validation
    :param param: a dictionary with various information like batch size, resize information,
                camera position probabilities, brightness, image cropping,
    :param img_list:
    :return:
    """

    batch_size = param['batch_size']
    final_img_cols, final_img_rows, chn = param['resize_image']

    while 1:
        # Create the arrays for features and labels for the batch
        batch_features = np.zeros((batch_size, final_img_rows, final_img_cols, chn), dtype=np.float32)
        batch_labels = np.zeros((batch_size,), dtype=np.float32)
        # batch_weights = np.zeros((batch_size,), dtype=np.float32)

        steering_weights = get_weights(img_list)

        for i in range(batch_size):

            rnd_nbr, rnd_pos, rnd_img, rnd_str = select_random_image(img_list,
                                                                     steering_weights,
                                                                     param['camera_pos_pr'])

            if case.upper() in ('TRAIN', 'TRAINING'):

                # For training data adjust brightness, crop images, flip images, resize images and return the image
                new_img = adjust_brightness(rnd_img, param['brightness'])
                new_img = crop_image(new_img, param['image_crop'])
                new_img, new_str = flip_image(new_img, rnd_str)
                new_img = resize_image(new_img, (final_img_cols, final_img_rows))
                batch_features[i], batch_labels[i] = new_img, new_str

            elif case.upper() in ('VALID', 'VALIDATION'):

                # For validation set, crop images, flip images and resize images
                new_img = crop_image(rnd_img, param['image_crop'])
                new_img, new_str = flip_image(new_img, rnd_str)
                new_img = resize_image(new_img, (final_img_cols, final_img_rows))
                batch_features[i], batch_labels[i] = new_img, new_str

            else:

                sys.exit("unknown case in generator")

        yield batch_features, batch_labels


def training_histogram(batch_size, total_size, generator, out_folder):
    """
    Using this function we can see the distribution of steering angles that will be used in the training
    Functions call the generator and plots the histogram
    :param batch_size:
    :param total_size:
    :param generator:
    :param out_folder:
    :return:
    """
    itr = 0
    y_out = np.zeros((total_size,))
    for g in generator:
        print(itr)
        if itr * batch_size == total_size:
            break
        y_out[itr * batch_size:(itr + 1) * batch_size, ] = g[1]
        itr += 1

    plt.hist(y_out, bins=40)
    plt.title("Histogram of Steering Data")
    plt.xlabel("Value")
    plt.ylabel("Frequency")
    plt.savefig(os.path.join(out_folder, 'training_data_histogram'))
    plt.close()
    sys.exit("just saved training data histogram")


# CNN model
#######################################################################################################################

def mynet(img_shape):
    """
    My model for training.
    :param img_shape:
    :return:
    """
    final_img_cols, final_img_rows, chn = img_shape
    my_net = Sequential()
    # Lambda layer for normalizing pixel values from -0.5 to +0.5
    my_net.add(Lambda(lambda x: (x - 125.0) / 255.0, input_shape=(final_img_rows, final_img_cols, chn)))
    # 5 convolution layers with drop out after 4th and 5th layers
    # Relu activation function
    # (5,5) filtering and (2,2) subsampling in the first 3 layers
    # (3,3) filtering and (1,1) subsampling for the last 2 layers
    # Valid padding for all
    my_net.add(Convolution2D(24, 5, 5, border_mode='valid', activation='relu', subsample=(2, 2)))
    my_net.add(Convolution2D(36, 5, 5, border_mode='valid', activation='relu', subsample=(2, 2)))
    my_net.add(Convolution2D(48, 5, 5, border_mode='valid', activation='relu', subsample=(2, 2)))
    my_net.add(Convolution2D(64, 3, 3, border_mode='valid', activation='relu', subsample=(1, 1)))
    my_net.add(Dropout(0.2))
    my_net.add(Convolution2D(64, 3, 3, border_mode='valid', activation='relu', subsample=(1, 1)))
    my_net.add(Dropout(0.2))
    # Flatten it to transition for FCs
    my_net.add(Flatten())
    # 5 FC layers with drop out only after the first one since it has the highest number of parameters
    my_net.add(Dense(1164, activation='relu'))
    my_net.add(Dropout(0.2))
    my_net.add(Dense(100, activation='relu'))
    my_net.add(Dense(50, activation='relu'))
    my_net.add(Dense(10, activation='relu'))
    # Output depth is 1 since this is a regression problem
    # Activation function is TanH like the NVIDIA model.
    my_net.add(Dense(1, activation='tanh'))

    # Adam optimizer with 0.0001 learning rate
    adam_opt = Adam(lr=1.0e-4)
    # Loss function is mean squared error
    my_net.compile(optimizer=adam_opt, loss='mse')

    return my_net