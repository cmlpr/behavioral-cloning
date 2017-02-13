---
layout: post
title: "Behavioral Clonning for Predicting Steering Angles"
date: 2017-02-11
mathjx: false
hlighters: htmlcode
image: behavioral_cloning.jpg
imagealt: "Behavioral Cloning"
description: "Using a simulator and Convolutional Neural Network (CNN) model to predict steering angles for Self Driving Cars "
category: Neural Networks
tags: behavioral_cloning tensorflow keras neural_networks deep_neural_networks convolutional_neural_networks python opencv image_processing self_driving_cars matplotlib sklearn generators
published: true
comments: true
---

In this blog post we will teach a car drive itself in autonomous mode. The overall methodology is called "Behavioral Cloning" in which a model learns from imitating human behavior. 

<!--more-->

[//]: # (Image References)

[image1]: /images/behavioral_cloning/random_image_set.png "A random image set with all three camera views"
[image2]: /images/behavioral_cloning/drive_data.png "Recorded drive data"
[image3]: /images/behavioral_cloning/steering_histogram.png "Steering angle histogram"
[image4]: /images/behavioral_cloning/brightness_adjustment.png "Example image with brightness adjustment"
[image5]: /images/behavioral_cloning/cropped_image.png "Example image after cropping"
[image6]: /images/behavioral_cloning/resized_image.png "Example image after resizing"
[image7]: /images/behavioral_cloning/flipped_image.png "Example image after flipping"
[image8]: /images/behavioral_cloning/rotated_image.png "Example image after rotation"
[image9]: /images/behavioral_cloning/translated_image.png "Example image after translation"
[image10]: /images/behavioral_cloning/weighted_steering_histogram.png "Histogram of raw steering data with weights"
[image11]: /images/behavioral_cloning/training_data_histogram.png "Histogram of the final training data"
[image12]: /images/behavioral_cloning/CNN_model_arch.png "Convolutional Model Architecture"


Here are the steps we will follow from high-level perspective:

- Use the simulator that Udacity provides to record images and driving data 
- Prepare data and make it ready for a deep learning exercise
- Build a Convolutional Neural Network (CNN) in Keras using TensorFlow backend and train it using the pre-processed data
- Test the model in the simulator in autonomus mode

Files associated with this post are located in this **[REPO](https://github.com/cmlpr/behavioral-cloning)**. Here is the list:

- model_new.py includes the main function used in training the model
- FLib.py includes all the helper functions used in model.py - I prefer collecting functions in another file as model.py gets too long and confusing
- drive.py contains the script to drive the car in autonomous mode
- new_model_1.h5 is the saved convolutional neural network
- 2017-02-11-behavioral-cloning.md is this markdown file you are reading now


### Training data

This section will go through the steps in more detail. 

---


#### Getting data

There are two options to get training data: 

+   Use the sample data provided 
+   Drive the car around the track and record images 

I employed both methodologies but found that my model worked much better with the data I recorded in which I drove around the track multiple times and tried to stay on the road at all times. The functions used for getting the data into `model.py` are in `FLib.py`.


| Function               | Definition                                                          |
| :--------------------- | :-------------------------------------------------------------------|
| use_online_data        | Function that returns downloaded and extracted driving data         |
| use_local_recording    | Function that returns driving data recorded locally - no extraction |
| download_data          | Function to download data from the provided link                    |
| download_progress_hook | Function that shows the progress during download                    |
| extract_data           | Function to extract `.zip` files                                    |
|                        |                                                                     |

Techniques that can help with getting a good solution would require feeding as many data points as possible with all the possible combinations of things such as recovering the car from road sides. Due to limited time I only recorded data with regular driving and used a few data augmentation techniques to improve the data set. 

There are two tracks in the simulator. I used the first track to record training data in the beta simulator. One of the characteristics of this track is that it is mostly straight driving with occasional left turn, by default driving is counter clock wise (CCW). 

The simulator saves images and a driving log in a folder that you specify at the beginning of the recording. What I found out is that if you have any problems during autonomous driving and want to get more data with new recordings, the simulator stores new images to the same folder and appends the new drive log at the end of the existing drive log. 

#### Exploring Data

The beta simulator (MACOS version) records images at a rate of 13hz. There are 3 sets of images: left camera image, center camera image and right camera image. In addition to the images, we have access to the steering angle, throttle, brake and speed information. All these information are recorded in the drive log in CSV file format. Each row in the drive log represents instantaneous data. The first three columns are the paths for each image. Image file name specifies which camera it belongs to and the timestamp. Images have 3 channels (BGR) and they are 160px in height and 320px in width. 

```python
Length of data: 12872
Data Headers:  ['center', 'left', 'right', 'steering', 'throttle', 'brake', 'speed'] 
Shape of each image:  (160, 320, 3)
```

Here is a randomly selected image set. It shows all three images from left, center and right cameras.

![Simulator output][image1]

It is also a good idea to visualize other driving data to understand what we are up against. So let's plot other data from the driving log:

![Drive Data][image2]

Among all these 4 different data tags we are interested in the steering angle as it is going to be the parameter we will predict in this project. Throttle can also be important but we'll set it to some reasonable value during autonomous driving. 

It looks like steering angle changes from -1 to +1 in the drive log. In the simulator steering angle is displayed at the upper left corner of the screen and it chanegs from -25 to +25 degrees so we now know that this data has already been normalized with the max possible value. Another intresting observation is that the steering data is usually small and negative. This is information is not surprising as the track is CCW and it only requires small adjustments in steering angle most of the time. This information is very critical in model training as we can guess that if we only use this data our predictions will be bias toward small negative numbers. Now let's look at how steering angle data is distributed using a histogram plot which further emphasizes this point. 

![Steering angle histogram][image3]

This table shows the functions I used in this part:

| Function                | Definition                                                                                |
| :---------------------  | :-----------------------------------------------------------------------------------------|
| read_img                | Reads an image and steering angle with given camera and array position in the driving log |
| plot_random_image       | Plots a random image from the drive log with given camera position                        |
| plot_random_image_set   | Plots all three camera images at a random array position in the driving log               |
| plot_driving_data       | Plots all other drive data: steering angle, throttle, brake and speed                     |
| plot_steering_histogram | Plots the histogram of the steering angle                                                 |
| compare_2_images        | Compares two images side by side                                                          |
|                         |                                                                                           |

#### Pre-processing data

After exploring the data and being unsuccessful in a few autonomous driving trials, I created a few functions that helped my model worked better and faster. I did a lot of trial-error runs and finally used only a few of them. 

* Normalizing images

Normalization (mean zero and equal variance) helps us to have a more numerically stable problem and provides a well-conditioned system for the optimization algorithm. For images this process is very simple as we already know that image pixels can get values from 0 to 255. We can subtract 125 and divide by 255 to reduce the range of pixes to -0.5 to 0.5.

```python
def normalize_scales(img):
    """
    Normalize images by subtracting mean and dividing by the range so that pixel values are between -0.5 and 0.5
    :param img:
    :return:
    """
    normalized_image = (img - 125.0) / 255.0
    return normalized_image
```

* Adjusting brightness

Adjusting the brightness of the images is a pre-processing step that we can use to generate extra images that would simulate brighter or darker cases. This process hopefully makes our model more robust. `HSV` color space provides the "Value" channel which is the brigtness therefore I prefered to multiply the V value with a random number between -0.5 to 1.15. The upper range is not 1.5 as the images are already bright enough and it is best to use the function to get darker images. 

```python
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
```

A side by side comparison of a randomly selected image before and after brightness adjustment.

![Brightness Adjustment][image4]

* Cropping

To reduce image size and speed up the training, we can crop the images from top (sky) and bottom (car). I prefered to crop 40px from top and 25px from bottom. 

```python
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
```

A side by side comparison of a randomly selected image before and after cropping.

![Cropped Image][image5]

* Resize Images

This process may not be necessary but I decided to apply it anyway. 

```python
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
```

A side by side comparison of a randomly selected image before and after resizing.

![Resized Image][image6]

* Flipping Images

This pre-processing technique is very useful in this study. We already talked about the bias in the steering angles. We can flip images first to increase our data size and then to generate positive steering angle images. This process will balance the data set with respect to 0 angle.  

```python
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
```

A side by side comparison of a randomly selected image before and after flipping.

![Flipped Image][image7]

* Rotating Images

I also created a script for rotating the images at a given position measured from the center of the image. Although I tried to use a few time, I couldn't get improvement in the results. I can't really say that this is not needed as there are many hyperparameters that might be impacting the results. I haven't fully followed a good design of experiments (DEO) study here so I'll figure if rotated images could help with training at a later time.

In a nutshell this function pick a random rotation angle and rotates the images and updates the steering angle in the same direction as the rotation. More information can be found in the code snippet below. 

```python
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
```

A side by side comparison of a randomly selected image before and after rotating.

![Rotated Image][image8]

* Translating Images

This function translates the images in x and y directions. Like rotation, I didn't observe improvement in the model when this function is used. I'll investigate further at a later time. 

```python
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
```

A side by side comparison of a randomly selected image before and after translation.

![Translated Image][image9]

* Using left and right camera images 

We can use left and right camera images to simulate the recorvery of the car from sides. Using left and right images also helps us to increase our training data. When a left camera image is used we can assume that a steering angle needs to be increase by +0.25 (6.25deg) to bring the car to the cente line. We subtract 0.25 from the steering angle for the right camera images to simulate steering left to recover the car from the right of the centerline. These angle adjustemnts are found by trial and method and trough discussion.  

```python
if camera_position == 'left':
    # Adjust left images by +0.25 to simulate recovery from left
    steering_angle = min(float(steering_angle) + 0.25, 1.0)
elif camera_position == 'right':
    # Adjust right images by -0.25 to simulate recovery from right
    steering_angle = max(float(steering_angle) - 0.25, -1.0)
else:
    pass
```

* Increasing the probability of high steering angle images 

The histogram of the recorded steering angle data shows that there is a peak around 0 degrees. When all these images are used without adjustment there is a risk that our model would predict steering angles that are close to 0. To prevent that we can assign higher weights to high steering angles. The methodoly I followed here is very simple and can be improved. I used the normalized steering angles as the weights in random image selection code. The absolute value of the steering angle plus a small adder divided by the sum of all steering angles was used as the probability of that image. Images with higher steering angles would have higher wwights using this method. I added a small value to each steering angle as 0 steering angle would yield 0 weight which would prevent us from selecting 0 steering angle images.  

```python
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
```

When weights are assigned, histogram of the steering angle improves as shown in the plot below. 

![Histogram of steering data with weights][image10]

The final data submitted to the model for training looks like this:

![Histogram of final training data][image11]

There are three peaks in the histogram: -0.25, 0, +0.25. Two new peaks at -0.25 and 0.25 are due to using left and right images and correcting 0 degree images with 0.25 in the `Using left and right camera images` section above. 



### Model Architecture

---

Here is the model function `mynet` I used for this task. This can also be found in `FLIB.py` function library. It takes the image shape and returns the Keras Model.

```python
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
```

This model is based on NVIDIA's model published in **[End to End Learning for Self-Driving Cars](http://images.nvidia.com/content/tegra/automotive/images/2016/solutions/pdf/end-to-end-dl-using-px.pdf)**

I used  `TensorFlow` with `Keras` high level library which is easier to read and prototype with. My model consists of 5 layers of convolution neural networks with 5x5 and 3x3 filter sizes, depths between 24 and 64, and 2x2 and 1x1 subsampling. The model includes RELU layers to introduce nonlinearity, and the data is normalized in the model using a Keras lambda layer. I later prefered to use Lambda layer rather than using my own function as normalization done during training would be parallel. Convolutional layers are followed by a flatten layer and 4 fully connected layers. Output layer size is 1. The model contains dropout layers in order to reduce overfitting. Dropouts layers are used at the end of last 2 convolutional layers and first fully connected layers as those layers are bigger in size and dropout will be more effective. 

Here is a visualization of the architecture:

![Model Architecture][image12]

This model uses `Adam` optimizer so that learning rate is tuned automatically during the optimization process. I started with a learning rate of `0.0001`. Since this is a regression problem it employs `Mean Squared Error - MSE` as the loss function. 

I used simpler models in my trials but then found that NVIDIA's model is a good starting point. The only problem was that although the model was generating good results with during training (low mse in validation set) it wasn't performing well during testing mode (using the simulator). This implied that the model was overfitting and adding the dropout layers solved the problem. 

I got a good performance in the track with this model so I didn't feel the need to get extra images or recovery data from the simulator. 


### Some More Detail

---

#### Model Parameters

I used 5 epoches with 256*80 batches. To save memory and perform parallel image processing, I used the `model.fit_generator` function in Keras. I feed 256 images in each call of the generator. Validation size was 1024. During training I randomly picked left right and center camera images with almost equal probabilities but during validation I only picked the **center image only** as simulator only uses center image. I also didn't use the brightness augmentation for validation images. 

Input image size was set to (96, 64, 3). 

```python
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
}

valid_preprocess_spec = {
    'batch_size': batch_size,
    'camera_pos_pr': (0.0, 1.0, 0.0),
    'image_crop': crop_dim,
    'resize_image': img_size,
}
```

#### Python Generator

This is the generator function I used not to run out of memory during training and perform image processing in parallel fashion. 

```python

train_generator = FLib.data_generator('train', train_preprocess_spec, img_list)
valid_generator = FLib.data_generator('valid', valid_preprocess_spec, img_list)

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
```

#### Runnning and Saving the Model

We can use `ModelCheckpoint` and `EarlyStopping` function provided in `Keras` to store model checkpoints and terminate the run based on `validation loss`. These features were helpful during my trial-errors runs. 

```python
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
```

#### Changes in Drive.py

Since we pre-processed the images `drive.py` needs to be updated with necessary processing changes. Among the pre-processing steps that were highlighted above only `resizing` and `cropping` will be necessary in the testing phase. 

List of changes

* Line 20: import FLib.py
* Line 41: resize and crop simulator images before feeding to the model
* Line 45-52: small changes in speed and throttle values - increased the speed a little

#### How to run autonomously

Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 

```sh
python drive.py new_model_1.h5
```

### Video of Autonomous Driving



<video controls="controls">
  <source type="video/mp4" src="/videos/new_model_1_out.mp4"></source>
  <p>Your browser does not support the video element.</p>
</video>