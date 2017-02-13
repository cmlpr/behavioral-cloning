import os
import sys
import FLib
import numpy as np
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split

from keras.callbacks import EarlyStopping, ModelCheckpoint

# Get Data
###########################################################################################################

# Option 1
# Downloading the data provided
# training_data = 'data.zip'
# data_url = 'https://d17h27t6h515a5.cloudfront.net/topher/2016/December/584f6edd_data/'
# data_size = 333137665
#
# img_path, img_list = FLib.use_online_data(data_url, training_data, data_size)

# Option 2
# Using data I recorded - beta simulator
img_path = 'my_recordings/betasim_trial1'
img_list = FLib.use_local_recording(img_path)


# Exploring the data
###########################################################################################################

# First create an output folder if not already present
img_out_folder = 'images'
if os.path.isdir(img_out_folder):
    print("\nImage output directory is already present. Please see output images in <%s> folder" % img_out_folder)
else:
    os.makedirs(img_out_folder)
    print("\nImage output directory created! Please see output images in <%s> folder" % img_out_folder)

# Data Size
img_count = len(img_list)
print('Length of data %s' % img_count)

# Data Headers
print('Data Headers: ', list(img_list), '\n')

# Print the fist 5 columns
# print(img_list.head())

# Plot a random center image
FLib.plot_random_image(img_list, 'center', img_out_folder)

# Plot a random left image
FLib.plot_random_image(img_list, 'left', img_out_folder)

# Plot a random right image
FLib.plot_random_image(img_list, 'right', img_out_folder)

# Plot a random image set (left-center-right)
FLib.plot_random_image_set(img_list, img_out_folder)

# Plot the driving data
FLib.plot_driving_data(img_list, img_out_folder)

# Plot histogram of steering data
FLib.plot_steering_histogram(img_list, 'steering_histogram', img_out_folder)

# Assign weights to each image depending on the steering angle and then plot histogram
steering_weights = FLib.get_weights(img_list)
FLib.plot_steering_histogram(img_list, 'weighted_steering_histogram', img_out_folder, steering_weights)


# Test Pre-processing Functions
###########################################################################################################

# Probability of center, left, right images
position_pr_array = (0.33, 0.34, 0.33)

# Select a random image
rnd_nbr, rnd_pos, rnd_img, rnd_str = FLib.select_random_image(img_list,
                                                              steering_weights,
                                                              position_pr_array)
# Adjust brightness
new_img = FLib.adjust_brightness(rnd_img, (-0.5, 0.15))
new_str = rnd_str
FLib.compare_2_images(rnd_nbr, rnd_pos, (rnd_img, rnd_str), (new_img, new_str),
                      'brightness_adjustment', img_out_folder)

# Normalize the pixel values
new_img = FLib.normalize_scales(rnd_img)
new_str = rnd_str
FLib.compare_2_images(rnd_nbr, rnd_pos, (rnd_img, rnd_str), (new_img, new_str),
                      'scale_normalization', img_out_folder)

# Crop image to eliminate unnecessary parts of the image and reduce memory and processor requirements
new_img = FLib.crop_image(rnd_img, (40, 25, 0, 0))
new_str = rnd_str
FLib.compare_2_images(rnd_nbr, rnd_pos, (rnd_img, rnd_str), (new_img, new_str),
                      'cropped_image', img_out_folder)

# Resize image to desired shape
new_img = FLib.resize_image(rnd_img, (64, 64))  # cols, rows
new_str = rnd_str
FLib.compare_2_images(rnd_nbr, rnd_pos, (rnd_img, rnd_str), (new_img, new_str),
                      'resized_image', img_out_folder)

# Flip images
new_img, new_str = FLib.flip_image(rnd_img, rnd_str)
FLib.compare_2_images(rnd_nbr, rnd_pos, (rnd_img, rnd_str), (new_img, new_str),
                      'flipped_image', img_out_folder)

# Rotate images
new_img, new_str = FLib.rotate_image(rnd_img, rnd_str, rot_info=(10.0, 0.0))
FLib.compare_2_images(rnd_nbr, rnd_pos, (rnd_img, rnd_str), (new_img, new_str),
                      'rotated_image', img_out_folder)

# Translate images
new_img, new_str = FLib.translate_image(rnd_img, rnd_str, trans_info=(40.0, 5.0))
FLib.compare_2_images(rnd_nbr, rnd_pos, (rnd_img, rnd_str), (new_img, new_str),
                      'translated_image', img_out_folder)

# An example with multiple steps
# Apply multiple steps
new_img = FLib.adjust_brightness(rnd_img, (-0.5, 0.15))
new_img = FLib.crop_image(new_img, (40, 25, 0, 0))
new_img, new_str = FLib.flip_image(new_img, rnd_str)
new_img = FLib.resize_image(new_img, (64, 64))
new_img, new_str = FLib.translate_image(new_img, new_str, trans_info=(40.0, 5.0))
new_img = FLib.normalize_scales(new_img)

FLib.compare_2_images(rnd_nbr, rnd_pos, (rnd_img, rnd_str), (new_img, new_str),
                      'all_preprocessing_steps', img_out_folder)

# Model
###########################################################################################################

# x_train = shuffle(img_list)
# x_train, x_valid = train_test_split(x_train, test_size=0.20, random_state=832289)

# weights = FLib.get_weights(x_valid)
# FLib.plot_steering_histogram(x_valid, 'steering_histogram_x_valid', img_out_folder)
# FLib.plot_steering_histogram(x_valid, 'weighted_steering_histogram_x_valid', img_out_folder, weights)

# First create an output folder if not already present
model_out_folder = 'models'
if os.path.isdir(model_out_folder):
    print("\nModel output directory is already present. Please see models in <%s> folder" % model_out_folder)
else:
    os.makedirs(model_out_folder)
    print("\nModel output directory created! Please see models in <%s> folder" % model_out_folder)

model_name = 'new_model_1'
epoch = 5
batch_size = 256
samples_in_each_epoch = batch_size*80
samples_in_validation = batch_size*4

brightness = (-0.5, 0.15)
crop_dim = (40, 25, 0, 0)
img_size = (96, 64, 3)

train_preprocess_spec = {
    'batch_size': batch_size,
    'camera_pos_pr': (0.33, 0.34, 0.33),
    'brightness': brightness,
    'image_crop': crop_dim,
    'resize_image': img_size,
    'rotate_image': (10.0, 0.0),
    'translate_image': (40.0, 5.0)
}

valid_preprocess_spec = {
    'batch_size': batch_size,
    'camera_pos_pr': (0.0, 1.0, 0.0),
    'brightness': brightness,
    'image_crop': crop_dim,
    'resize_image': img_size,
}

train_generator = FLib.data_generator('train', train_preprocess_spec, img_list)
valid_generator = FLib.data_generator('valid', valid_preprocess_spec, img_list)

# Plot histogram for data used in an epoch
plot_training_data_histogram = False
if plot_training_data_histogram:
    FLib.training_histogram(batch_size, samples_in_each_epoch, train_generator, img_out_folder)

model = FLib.mynet(img_size)
model.summary()

model_h5 = os.path.join(model_out_folder, model_name + '.h5')

print(model_h5)
if os.path.isfile(model_h5):
    os.remove(model_h5)

checkpoint = ModelCheckpoint(model_h5, monitor='val_loss', verbose=1, save_best_only=True, save_weights_only=False)

early_stopping = EarlyStopping(monitor='val_loss', min_delta=0, patience=0, verbose=1)

history = model.fit_generator(train_generator,
                              samples_per_epoch=samples_in_each_epoch,
                              nb_epoch=epoch,
                              callbacks=[checkpoint, early_stopping],
                              verbose=1,
                              validation_data=valid_generator,
                              nb_val_samples=samples_in_validation)

# Save model
model.save(model_h5)
