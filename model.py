import os
import sys
from six.moves.urllib.request import urlretrieve
import zipfile
import pandas as pd
import numpy as np
import cv2
# from PIL import Image
import matplotlib.pyplot as plt
# import matplotlib.gridspec as gridspec
from keras.models import Sequential
from keras.layers.core import Lambda, Dense, Activation, Flatten, Dropout
from keras.layers.convolutional import Convolution2D
from keras.layers.pooling import MaxPooling2D
from keras.layers.advanced_activations import ELU
from keras.optimizers import Adam
from keras.layers import BatchNormalization
from keras.callbacks import EarlyStopping, ModelCheckpoint
import json

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

    path_to_log = os.path.join(root, os.listdir(root)[1])
    path_to_images = os.path.join(root)

    return path_to_log, path_to_images


def read_img(camera_position, array_position):
    # Note we need to trip left whitespace for left images
    # global img_files, img_path
    if camera_position in ('center', 'left', 'right'):
        img = cv2.imread(os.path.join(img_path, img_files[camera_position][array_position].strip()))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        steer_angle = img_files['steering'][array_position]
        return img, steer_angle
    else:
        sys.exit('Wrong camera position defined in read_img')


def plot_random_image(camera_position):
    img_number = int(np.random.randint(0, img_count, 1))
    im, _ = read_img(camera_position, img_number)
    print("Plotting image #%s-%s image" % (img_number, camera_position))
    print('Shape of the image: ', im.shape)
    plt.axis("off")
    plt.imshow(im)
    plt.title('Random %s Image: %s' % (camera_position, img_number))
    fname = 'random_' + camera_position + '_image'
    plt.savefig(os.path.join(img_out_folder, fname))
    print('Completed plotting\n')
    plt.close()


def plot_random_image_set():
    img_number = int(np.random.randint(0, img_count, 1))
    fig, axes = plt.subplots(1, 3, figsize=(8, 3), subplot_kw={'xticks': [], 'yticks': []})
    fig.subplots_adjust(hspace=0.2, wspace=0.05)
    fig.suptitle(('Random Image Set: %s' % img_number), fontsize=15)
    axes[0].imshow(read_img('left', img_number)[0])
    axes[0].set_title('left_camera', fontsize=10)
    axes[1].imshow(read_img('center', img_number)[0])
    axes[1].set_title('center_camera', fontsize=10)
    axes[2].imshow(read_img('right', img_number)[0])
    axes[2].set_title('right_camera', fontsize=10)
    plt.savefig(os.path.join(img_out_folder, 'random_image_set'))
    plt.close()
    print('Completed plotting left, center, right images for #%s\n' % img_number)


def plot_driving_data():
    plt.subplot(221)
    plt.plot(img_files['steering'])
    plt.ylabel('Steering Angle')
    plt.title('Steering Data')
    plt.grid(True)

    plt.subplot(222)
    plt.plot(img_files['throttle'])
    plt.ylabel('Throttle')
    plt.title('Throttle Data')
    plt.grid(True)

    plt.subplot(223)
    plt.plot(img_files['brake'])
    plt.ylabel('Brake')
    plt.title('Brake Data')
    plt.grid(True)

    plt.subplot(224)
    plt.plot(img_files['speed'])
    plt.ylabel('Speed')
    plt.title('Speed Data')
    plt.grid(True)

    plt.subplots_adjust(top=0.92, bottom=0.08, left=0.12, right=0.95, hspace=0.55,
                        wspace=0.35)
    plt.savefig(os.path.join(img_out_folder, 'drive_data'))
    plt.close()
    print('Completed plotting drive data\n')


def plot_steering_histogram():
    plt.hist(img_files['steering'], bins=40)
    plt.title("Histogram of Steering Data")
    plt.xlabel("Value")
    plt.ylabel("Frequency")
    plt.savefig(os.path.join(img_out_folder, 'steering_histogram'))
    plt.close()
    print('Completed plotting histogram of steering data\n')


def compare_2_images(img_no, camera_position, steering_angle, left, right, title):

    global img_out_folder

    plt.figure(figsize=(7, 4))
    plt.subplot(121)
    plt.axis("off")
    plt.imshow(left)
    plt.title(('Before - Str: %.3f' % steering_angle), fontsize=10)

    plt.subplot(122)
    plt.axis("off")
    plt.imshow(right[0])
    new_steering_angle = float(steering_angle) * right[1]
    plt.title(('After - Str: %.3f' % new_steering_angle), fontsize=10)
    plt.suptitle((title + ' - %s img: %s' % (camera_position, img_no)), fontsize=15)
    plt.savefig(os.path.join(img_out_folder, title))
    plt.close()
    print('Completed comparison plot for: %s\n' % title)


# Data Augmentation Helper Functions

def get_weights():
    wghts = []
    total = (abs(img_files['steering'])+0.05).sum()
    for img_loc in range(img_count):
        str_angle = abs(float(img_files['steering'][img_loc]))
        wghts.append((str_angle + 0.05)/total)
    return tuple(wghts)


def select_random_image(pr=(0.25, 0.50, 0.25)):
    # img_number = int(np.random.randint(0, img_count, 1))
    img_number = np.random.choice(img_count, 1, replace=True, p=steering_weights)[0]
    options_arr = ['left', 'center', 'right']
    camera_position = np.random.choice(options_arr, 1, p=pr)[0]
    im, steering_angle = read_img(camera_position, img_number)
    if camera_position == 'left':
        steering_angle = min(float(steering_angle) + 0.15, 1.0)
    elif camera_position == 'right':
        steering_angle = max(float(steering_angle) - 0.15, -1.0)
    else:
        pass
    return img_number, camera_position, im, steering_angle


def normalize_scales(img):
    """
    Normalize images by subtracting mean and dividing by the range so that pixel values are between -0.5 and 0.5
    :param img:
    :return:
    """
    # normalized_image = np.divide(img - 125.0, 255.0)
    normalized_image = (img - 125.0) / 255.0
    return normalized_image


def adjust_brightness(img):
    brightness_multiplier = np.random.uniform(low=-0.5, high=0.15)
    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    hsv[:, :, 2] = hsv[:, :, 2] * (1.0 + brightness_multiplier)
    adjusted = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
    return adjusted


def crop_image(img):
    # Crop 40px from top for removing the horizon
    # Crop 25px from bottom for removing the car body
    crop_top = 40
    crop_bottom = 25
    crop_left = 0
    crop_right = 0
    img_shape = img.shape
    cropped = img[crop_top:img_shape[0]-crop_bottom, crop_left:img_shape[1]-crop_right]
    return cropped


def resize_image(img):
    new_height = 64
    new_width = 64
    resized = cv2.resize(img, (new_width, new_height), interpolation=cv2.INTER_AREA)
    return resized


def flip_image(img):
    flip_random = int(np.random.binomial(1, 0.5, 1))
    if flip_random:
        return cv2.flip(img, 1), -1
    else:
        return img, 1


def select_pre_process_image(display=False, case='TRAIN'):
        if case == 'TRAIN':
            pr = (0.33, 0.34, 0.33)
        else:
            pr = (0.1, 0.8, 0.1)
        img_number, img_position, im, img_steer = select_random_image(pr)
        im_post = adjust_brightness(im)
        # im_post = normalize_scales(im_post)
        im_post = crop_image(im_post)
        im_post = resize_image(im_post)
        im_post, multiplier = flip_image(im_post)
        if display:
            compare_2_images(img_number, img_position, img_steer, im, (im_post, multiplier),
                             'all_preprocessing_steps')
        else:
            img_steer = float(img_steer) * multiplier
            return im_post, img_steer


def data_generator(case):

    while 1:
        batch_features = np.zeros((batch_size, 64, 64, 3), dtype=np.float32)
        batch_labels = np.zeros((batch_size,), dtype=np.float32)
        batch_weights = np.zeros((batch_size,), dtype=np.float32)

        for i in range(batch_size):
            batch_features[i], batch_labels[i] = select_pre_process_image(False, case.upper())
            batch_weights[i] = min(abs(batch_labels[i]) + 0.05, 1.0)

        if case.upper() == 'TRAIN':
            yield batch_features, batch_labels, batch_weights
        else:
            yield batch_features, batch_labels


def commaai_model():

    cai_model = Sequential()
    cai_model.add(Lambda(lambda x: (x - 125.0) / 255.0, input_shape=(64, 64, 3)))
    cai_model.add(Convolution2D(16, 8, 8, subsample=(4, 4), border_mode="same", init='glorot_normal'))
    cai_model.add(Activation('elu'))
    cai_model.add(Convolution2D(32, 5, 5, subsample=(2, 2), border_mode="same", init='glorot_normal'))
    cai_model.add(Activation('elu'))
    cai_model.add(Convolution2D(64, 5, 5, subsample=(2, 2), border_mode="same", init='glorot_normal'))
    cai_model.add(Flatten())
    cai_model.add(Dropout(.2))
    cai_model.add(Activation('elu'))
    cai_model.add(Dense(512, init='glorot_normal'))
    cai_model.add(Dropout(.5))
    cai_model.add(Activation('elu'))
    cai_model.add(Dense(1, init='glorot_normal'))

    return cai_model


def nvidia_model():

    nvd_model = Sequential()
    nvd_model.add(BatchNormalization(epsilon=0.001, mode=1, axis=3, input_shape=(64, 64, 3)))
    nvd_model.add(Convolution2D(24, 5, 5, border_mode='valid', activation='relu', subsample=(2, 2)))
    nvd_model.add(Convolution2D(36, 5, 5, border_mode='valid', activation='relu', subsample=(2, 2)))
    nvd_model.add(Convolution2D(48, 5, 5, border_mode='valid', activation='relu', subsample=(2, 2)))
    nvd_model.add(Convolution2D(64, 3, 3, border_mode='valid', activation='relu', subsample=(1, 1)))
    nvd_model.add(Convolution2D(64, 3, 3, border_mode='valid', activation='relu', subsample=(1, 1)))
    nvd_model.add(Flatten())
    nvd_model.add(Dense(1164, activation='relu'))
    nvd_model.add(Dense(100, activation='relu'))
    nvd_model.add(Dense(50, activation='relu'))
    nvd_model.add(Dense(10, activation='relu'))
    nvd_model.add(Dense(1, activation='tanh'))

    return nvd_model


def mynet1():

    my_net1 = Sequential()
    my_net1.add(Lambda(lambda x: (x - 125.0) / 255.0, input_shape=(64, 64, 3)))
    my_net1.add(Convolution2D(32, 3, 3, subsample=(1, 1), border_mode='same'))
    my_net1.add(ELU())
    my_net1.add(MaxPooling2D((2, 2)))
    my_net1.add(Convolution2D(16, 3, 3, subsample=(1, 1), border_mode='same'))
    my_net1.add(ELU())
    my_net1.add(MaxPooling2D((2, 2)))
    my_net1.add(Flatten())
    my_net1.add(Dropout(0.2))
    my_net1.add(Dense(512))
    my_net1.add(Dropout(0.4))
    my_net1.add(ELU())
    my_net1.add(Dense(128))
    my_net1.add(ELU())
    my_net1.add(Dense(1))

    return my_net1


def mynet2():

    my_net2 = Sequential()
    my_net2.add(Lambda(lambda x: (x - 125.0) / 255.0, input_shape=(64, 64, 3)))
    my_net2.add(Convolution2D(24, 3, 3, border_mode='valid', activation='elu', subsample=(2, 2)))
    my_net2.add(Dropout(0.5))
    my_net2.add(Convolution2D(36, 3, 3, border_mode='valid', activation='elu', subsample=(2, 2)))
    my_net2.add(Dropout(0.5))
    my_net2.add(Convolution2D(48, 3, 3, border_mode='valid', activation='elu', subsample=(2, 2)))
    my_net2.add(Dropout(0.5))
    my_net2.add(Convolution2D(64, 3, 3, border_mode='valid', activation='elu', subsample=(1, 1)))
    my_net2.add(Dropout(0.5))
    my_net2.add(Convolution2D(128, 3, 3, border_mode='valid', activation='elu', subsample=(1, 1)))
    my_net2.add(Dropout(0.5))
    my_net2.add(Flatten())
    my_net2.add(Dense(512, activation='elu'))
    my_net2.add(Dropout(0.25))
    my_net2.add(Dense(100, activation='elu'))
    my_net2.add(Dropout(0.25))
    my_net2.add(Dense(50, activation='elu'))
    my_net2.add(Dropout(0.25))
    my_net2.add(Dense(10, activation='elu'))
    my_net2.add(Dense(1, activation='tanh'))

    return my_net2


def mynet3():

    my_net3 = Sequential()
    my_net3.add(BatchNormalization(epsilon=0.001, mode=1, axis=3, input_shape=(64, 64, 3)))
    my_net3.add(Convolution2D(24, 5, 5, border_mode='valid', activation='relu', subsample=(2, 2)))
    my_net3.add(Convolution2D(36, 5, 5, border_mode='valid', activation='relu', subsample=(2, 2)))
    my_net3.add(Convolution2D(48, 5, 5, border_mode='valid', activation='relu', subsample=(2, 2)))
    my_net3.add(Convolution2D(64, 3, 3, border_mode='valid', activation='relu', subsample=(1, 1)))
    my_net3.add(Convolution2D(64, 3, 3, border_mode='valid', activation='relu', subsample=(1, 1)))
    my_net3.add(Flatten())
    my_net3.add(Dense(1164, activation='relu'))
    my_net3.add(Dropout(0.2))
    my_net3.add(Dense(100, activation='relu'))
    my_net3.add(Dense(50, activation='relu'))
    my_net3.add(Dense(10, activation='relu'))
    my_net3.add(Dense(1, activation='tanh'))

    return my_net3

###########################
###########################
###########################


training_data = 'data.zip'
data_url = 'https://d17h27t6h515a5.cloudfront.net/topher/2016/December/584f6edd_data/'

project_data = download_data(data_url, training_data, 333137665)

log_path, img_path = extract_data(training_data)
# Make sure the path to log file is correct
print(log_path)
# Make sure the images path is correct
print(img_path)

# Read the CSV File
img_files = pd.read_csv(log_path)

# Explore data
###########################################################################################################

# First create an output folder if not already present
img_out_folder = 'images'
if os.path.isdir(img_out_folder):
    print("\nImage output directory is already present. Please see output images in <%s> folder" % img_out_folder)
else:
    os.makedirs(img_out_folder)
    print("\nImage output directory created! Please see output images in <%s> folder" % img_out_folder)

# Data Size
img_count = len(img_files)
print('Length of data %s' % img_count)

# Data Headers
print('Data Headers: ', list(img_files), '\n')

# Print the fist 5 columns
# print(img_files.head())

# Plot a random center image
plot_random_image('center')

# Plot a random left image
plot_random_image('left')

# Plot a random right image
plot_random_image('right')

# Plot a random image set (left-center-right)
plot_random_image_set()

# Plot the driving data
plot_driving_data()

# Plot histogram of steering data
plot_steering_histogram()

# Test Pre-processing Functions
###########################################################################################################

# Select a random image with equal probability of left, center, right
steering_weights = get_weights()
rnd_img_number, rnd_position, rnd_img, rnd_str = select_random_image()

compare_2_images(rnd_img_number, rnd_position, rnd_str, rnd_img, (adjust_brightness(rnd_img), 1), 'brightness_adjustment')

compare_2_images(rnd_img_number, rnd_position, rnd_str, rnd_img, (normalize_scales(rnd_img), 1), 'scale_normalization')

compare_2_images(rnd_img_number, rnd_position, rnd_str, rnd_img, (crop_image(rnd_img), 1), 'cropped_image')

compare_2_images(rnd_img_number, rnd_position, rnd_str, rnd_img, (resize_image(rnd_img), 1), 'resized_image')

compare_2_images(rnd_img_number, rnd_position, rnd_str, rnd_img, flip_image(rnd_img), 'flipped_image')


# Combine all pre-processing steps in one function and compare the input and output
select_pre_process_image(display=True)

# Model
###########################################################################################################

# First create an output folder if not already present
model_out_folder = 'models'
if os.path.isdir(model_out_folder):
    print("\nModel output directory is already present. Please see models in <%s> folder" % model_out_folder)
else:
    os.makedirs(model_out_folder)
    print("\nModel output directory created! Please see models in <%s> folder" % model_out_folder)

model_name = 'model20'
epoch = 5
batch_size = 256
samples_in_each_epoch = 20480
samples_in_validation = 1024

# model = commaai_model()
# model = nvidia_model()
# model = mynet1()
# model = mynet2()
model = mynet3()
adam_opt = Adam(lr=1.0e-4)
model.compile(optimizer=adam_opt, loss='mse')
model.summary()

# Generators

training_generator = data_generator('train')
validation_generator = data_generator('valid')

# Plot histogram for data used in an epoch
# iter = 0
# batch_size = batch_size
# total_size = samples_in_each_epoch
# y_out = np.zeros((total_size, ))
# w_out = np.zeros((total_size, ))
# for g in training_generator:
#     print(iter)
#     if iter * batch_size == total_size:
#         break
#     y_out[iter*batch_size:(iter+1)*batch_size, ] = g[1]
#     w_out[iter*batch_size:(iter+1)*batch_size, ] = g[2]
#     iter += 1
#
# plt.hist(y_out, weights=w_out, bins=40)
# plt.title("Histogram of Steering Data")
# plt.xlabel("Value")
# plt.ylabel("Frequency")
# plt.savefig(os.path.join(img_out_folder, 'training_data_histogram'))
# plt.close()
# sys.exit()

model_json = os.path.join(model_out_folder, model_name + '.json')
model_h5 = os.path.join(model_out_folder, model_name + '.h5')

print(model_json, model_h5)
if os.path.isfile(model_json):
    os.remove(model_json)
if os.path.isfile(model_h5):
    os.remove(model_h5)

checkpoint = ModelCheckpoint(model_h5, monitor='val_loss', verbose=1, save_best_only=True, save_weights_only=False)

early_stopping = EarlyStopping(monitor='val_loss', min_delta=0, patience=0, verbose=1)

history = model.fit_generator(training_generator,
                              samples_per_epoch=samples_in_each_epoch,
                              nb_epoch=epoch,
                              callbacks=[checkpoint, early_stopping],
                              verbose=1,
                              validation_data=validation_generator,
                              nb_val_samples=samples_in_validation)

# Save model

with open(model_json, 'w') as outfile:
    json.dump(model.to_json(), outfile)
model.save_weights(model_h5)

