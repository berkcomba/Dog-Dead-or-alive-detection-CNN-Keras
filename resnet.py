
import xml.etree.ElementTree as ET
import cv2 
import matplotlib.pyplot as plt
from PIL import Image 
import numpy as np
from sklearn.utils import shuffle
import numpy as np
import matplotlib.pyplot as plt
import pickle
from keras.utils import to_categorical
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession
import keras.backend
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D, BatchNormalization
from keras import backend as K
from keras import regularizers
from keras import Input
import tensorflow as tf
import keras
from keras import layers
from keras import models
from keras import regularizers

#Gpu mamory allocation for reduce some errors couse by Cuda Version you can skip it.
"""
config = ConfigProto()
#config.gpu_options.allow_growth = True
config.gpu_options.per_process_gpu_memory_fraction = 0.6
session = InteractiveSession(config=config)
"""

#Data preprocessing
labelsc = []
xcmin=[]
xcmax=[]
ycmin=[]
ycmax=[]
labelst = []
xtmin=[]
xtmax=[]
ytmin=[]
ytmax=[]
isimc=[]
isimt=[]
imagesc=[]
imagest=[]
croppedimagesc=[]
croppedimagest=[]

for i in range(1,201):
    a=str(i)
    b="canliveri/"+a+".xml"
    c="telefveri/"+a+".xml"

    tree = ET.parse(b)
    root = tree.getroot()
    isimc.append(root[1].text)
    d="canliveri/"+isimc[i-1]
    labelsc.append(root[6][0].text)
    xcmin.append(int(root[6][4][0].text))
    xcmax.append(int(root[6][4][2].text))
    ycmin.append(int(root[6][4][1].text))
    ycmax.append(int(root[6][4][3].text))
    img = cv2.imread(d)
    cimg= img [ycmin[i-1]:ycmax[i-1],xcmin[i-1]:xcmax[i-1]]
    croppedimagesc.append(cv2.resize(cimg,(64,64),interpolation = cv2.INTER_AREA))    
    
    
    tree = ET.parse(c)
    root = tree.getroot()
    isimt.append(root[1].text)
    e="telefveri/"+isimt[i-1]
    labelst.append(root[6][0].text)
    xtmin.append(int(root[6][4][0].text))
    xtmax.append(int(root[6][4][2].text))
    ytmin.append(int(root[6][4][1].text))
    ytmax.append(int(root[6][4][3].text))
    img = cv2.imread(e)
    cimg= img [ytmin[i-1]:ytmax[i-1],xtmin[i-1]:xtmax[i-1]]
    croppedimagest.append(cv2.resize(cimg,(64,64),interpolation = cv2.INTER_AREA))  

train_images = []
test_images = []  
tumveri=[]
train_data=croppedimagesc[0:140]+croppedimagest[0:140]
train_data=croppedimagesc[0:140]+croppedimagest[0:140]
test_data=croppedimagesc[140:201]+croppedimagest[140:201]
train_labels=labelsc[0:140]+labelst[0:140]
test_labels=labelsc[140:201]+labelst[140:201]
train_images = np.asarray(train_data)
test_images  = np.asarray(test_data)



for i in range(280):
    if train_labels[i]=="canli":
        train_labels[i]=int(1)
    else:
        train_labels[i]=int(0)

for i in range(120):
    if test_labels[i]=="canli":
        test_labels[i]=int(1)
    else:
        test_labels[i]=int(0)
        
        
train_images=train_images.astype("float32")/255
test_images= test_images.astype("float32")/255

train_labelss = to_categorical(train_labels)
test_labelss  = to_categorical(test_labels)


inp = train_images[0].shape
train_images , train_labelss = shuffle(train_images,train_labelss)
test_images , test_labelss   = shuffle(test_images,test_labelss)
testshow = test_images.copy()




#model--

import tensorflow.keras
from tensorflow.keras.layers import Dense, Conv2D, BatchNormalization, Activation
from tensorflow.keras.layers import AveragePooling2D, Input, Flatten
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, LearningRateScheduler
from tensorflow.keras.callbacks import ReduceLROnPlateau
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.regularizers import l2
from tensorflow.keras import backend as K
from tensorflow.keras.models import Model
import numpy as np

BATCH_SIZE = 8  # orig paper trained all networks with batch_size=128
EPOCHS = 200 # 200
USE_AUGMENTATION = True
NUM_CLASSES = 2 # 10
COLORS = 3

# Subtracting pixel mean improves accuracy
SUBTRACT_PIXEL_MEAN = True

# Model version
# Orig paper: version = 1 (ResNet v1), Improved ResNet: version = 2 (ResNet v2)
VERSION = 1

# Computed depth from supplied model parameter n
if VERSION == 1:
    DEPTH = COLORS * 6 + 2
elif VERSION == 2:
    DEPTH = COLORS * 9 + 2

def lr_schedule(epoch):
    """Learning Rate Schedule

    Learning rate is scheduled to be reduced after 80, 120, 160, 180 epochs.
    Called automatically every epoch as part of callbacks during training.

    # Arguments
        epoch (int): The number of epochs

    # Returns
        lr (float32): learning rate
    """
    lr = 1e-3
    if epoch > 180:
        lr *= 0.5e-3
    elif epoch > 160:
        lr *= 1e-3
    elif epoch > 120:
        lr *= 1e-2
    elif epoch > 80:
        lr *= 1e-1
    print('Learning rate: ', lr)
    return lr

def resnet_layer(inputs,
                 num_filters=16,
                 kernel_size=3,
                 strides=1,
                 activation='relu',
                 batch_normalization=True,
                 conv_first=True):
    """2D Convolution-Batch Normalization-Activation stack builder

    # Arguments
        inputs (tensor): input tensor from input image or previous layer
        num_filters (int): Conv2D number of filters
        kernel_size (int): Conv2D square kernel dimensions
        strides (int): Conv2D square stride dimensions
        activation (string): activation name
        batch_normalization (bool): whether to include batch normalization
        conv_first (bool): conv-bn-activation (True) or
            bn-activation-conv (False)

    # Returns
        x (tensor): tensor as input to the next layer
    """
    conv = Conv2D(num_filters,
                  kernel_size=kernel_size,
                  strides=strides,
                  padding='same',
                  kernel_initializer='he_normal',
                  kernel_regularizer=l2(1e-4))

    x = inputs
    if conv_first:
        x = conv(x)
        if batch_normalization:
            x = BatchNormalization()(x)
        if activation is not None:
            x = Activation(activation)(x)
    else:
        if batch_normalization:
            x = BatchNormalization()(x)
        if activation is not None:
            x = Activation(activation)(x)
        x = conv(x)

    return x


def resnet_v1(input_shape, depth, num_classes=2):
    """ResNet Version 1 Model builder [a]

    Stacks of 2 x (3 x 3) Conv2D-BN-ReLU
    Last ReLU is after the shortcut connection.
    At the beginning of each stage, the feature map size is halved (downsampled)
    by a convolutional layer with strides=2, while the number of filters is
    doubled. Within each stage, the layers have the same number filters and the
    same number of filters.
    Features maps sizes:
    stage 0: 32x32, 16
    stage 1: 16x16, 32
    stage 2:  8x8,  64
    The Number of parameters is approx the same as Table 6 of [a]:
    ResNet20 0.27M
    ResNet32 0.46M
    ResNet44 0.66M
    ResNet56 0.85M
    ResNet110 1.7M

    # Arguments
        input_shape (tensor): shape of input image tensor
        depth (int): number of core convolutional layers
        num_classes (int): number of classes (CIFAR10 has 10)

    # Returns
        model (Model): Keras model instance
    """
    if (depth - 2) % 6 != 0:
        raise ValueError('depth should be 6n+2 (eg 20, 32, 44 in [a])')
    # Start model definition.
    num_filters = 16
    num_res_blocks = int((depth - 2) / 6)

    inputs = Input(shape=input_shape)
    x = resnet_layer(inputs=inputs)
    # Instantiate the stack of residual units
    for stack in range(3):
        for res_block in range(num_res_blocks):
            strides = 1
            if stack > 0 and res_block == 0:  # first layer but not first stack
                strides = 2  # downsample
            y = resnet_layer(inputs=x,
                             num_filters=num_filters,
                             strides=strides)
            y = resnet_layer(inputs=y,
                             num_filters=num_filters,
                             activation=None)
            if stack > 0 and res_block == 0:  # first layer but not first stack
                # linear projection residual shortcut connection to match
                # changed dims
                x = resnet_layer(inputs=x,
                                 num_filters=num_filters,
                                 kernel_size=1,
                                 strides=strides,
                                 activation=None,
                                 batch_normalization=False)
            x = tensorflow.keras.layers.add([x, y])
            x = Activation('relu')(x)
        num_filters *= 2

    # Add classifier on top.
    # v1 does not use BN after last shortcut connection-ReLU
    x = AveragePooling2D(pool_size=8)(x)
    y = Flatten()(x)
    outputs = Dense(num_classes,
                    activation='softmax',
                    kernel_initializer='he_normal')(y)

    # Instantiate model.
    model = Model(inputs=inputs, outputs=outputs)
    return model

def resnet_v2(input_shape, depth, num_classes=2):
    """ResNet Version 2 Model builder [b]

    Stacks of (1 x 1)-(3 x 3)-(1 x 1) BN-ReLU-Conv2D or also known as
    bottleneck layer
    First shortcut connection per layer is 1 x 1 Conv2D.
    Second and onwards shortcut connection is identity.
    At the beginning of each stage, the feature map size is halved (downsampled)
    by a convolutional layer with strides=2, while the number of filter maps is
    doubled. Within each stage, the layers have the same number filters and the
    same filter map sizes.
    Features maps sizes:
    conv1  : 32x32,  16
    stage 0: 32x32,  64
    stage 1: 16x16, 128
    stage 2:  8x8,  256

    # Arguments
        input_shape (tensor): shape of input image tensor
        depth (int): number of core convolutional layers
        num_classes (int): number of classes (CIFAR10 has 10)

    # Returns
        model (Model): Keras model instance
    """
    if (depth - 2) % 9 != 0:
        raise ValueError('depth should be 9n+2 (eg 56 or 110 in [b])')
    # Start model definition.
    num_filters_in = 16
    num_res_blocks = int((depth - 2) / 9)

    inputs = Input(shape=input_shape)
    # v2 performs Conv2D with BN-ReLU on input before splitting into 2 paths
    x = resnet_layer(inputs=inputs,
                     num_filters=num_filters_in,
                     conv_first=True)

    # Instantiate the stack of residual units
    for stage in range(3):
        for res_block in range(num_res_blocks):
            activation = 'relu'
            batch_normalization = True
            strides = 1
            if stage == 0:
                num_filters_out = num_filters_in * 4
                if res_block == 0:  # first layer and first stage
                    activation = None
                    batch_normalization = False
            else:
                num_filters_out = num_filters_in * 2
                if res_block == 0:  # first layer but not first stage
                    strides = 2    # downsample

            # bottleneck residual unit
            y = resnet_layer(inputs=x,
                             num_filters=num_filters_in,
                             kernel_size=1,
                             strides=strides,
                             activation=activation,
                             batch_normalization=batch_normalization,
                             conv_first=False)
            y = resnet_layer(inputs=y,
                             num_filters=num_filters_in,
                             conv_first=False)
            y = resnet_layer(inputs=y,
                             num_filters=num_filters_out,
                             kernel_size=1,
                             conv_first=False)
            if res_block == 0:
                # linear projection residual shortcut connection to match
                # changed dims
                x = resnet_layer(inputs=x,
                                 num_filters=num_filters_out,
                                 kernel_size=1,
                                 strides=strides,
                                 activation=None,
                                 batch_normalization=False)
            x = tensorflow.keras.layers.add([x, y])

        num_filters_in = num_filters_out

    # Add classifier on top.
    # v2 has BN-ReLU before Pooling
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = AveragePooling2D(pool_size=8)(x)
    y = Flatten()(x)
    outputs = Dense(num_classes,
                    activation='softmax',
                    kernel_initializer='he_normal')(y)

    # Instantiate model.
    model = Model(inputs=inputs, outputs=outputs)
    return model


# Input image dimensions.
input_shape = train_images.shape[1:]

# Normalize data.

# If subtract pixel mean is enabled
if SUBTRACT_PIXEL_MEAN:
    x_train_mean = np.mean(train_images, axis=0)
    train_images -= x_train_mean
    test_images -= x_train_mean

print('x_train shape:', train_images.shape)
print(train_images.shape[0], 'train samples')
print(train_images.shape[0], 'test samples')
print('y_train shape:', train_images.shape)

# Convert class vectors to binary class matrices.


# Create the neural network
if VERSION == 2:
    model = resnet_v2(input_shape=input_shape, depth=DEPTH)
else:
    model = resnet_v1(input_shape=input_shape, depth=DEPTH)

model.compile(loss='categorical_crossentropy',
              optimizer=Adam(lr=lr_schedule(0)),
              metrics=['accuracy'])
model.summary()





import time

start_time = time.time()

# Prepare callbacks for model saving and for learning rate adjustment.
lr_scheduler = LearningRateScheduler(lr_schedule)

lr_reducer = ReduceLROnPlateau(factor=np.sqrt(0.1),
                               cooldown=0,
                               patience=5,
                               min_lr=0.5e-6)

callbacks = [lr_reducer, lr_scheduler]

# Run training, with or without data augmentation.
if not USE_AUGMENTATION:
    print('Not using data augmentation.')
    model.fit(train_images, train_labelss,
              batch_size=BATCH_SIZE,
              epochs=EPOCHS,
              validation_data=(test_images, test_labelss),
              shuffle=True,
              callbacks=callbacks)
else:
    print('Using real-time data augmentation.')
    # This will do preprocessing and realtime data augmentation:
    datagen = ImageDataGenerator(
        # set input mean to 0 over the dataset
        featurewise_center=False,
        # set each sample mean to 0
        samplewise_center=False,
        # divide inputs by std of dataset
        featurewise_std_normalization=False,
        # divide each input by its std
        samplewise_std_normalization=False,
        # apply ZCA whitening
        zca_whitening=False,
        # epsilon for ZCA whitening
        zca_epsilon=1e-06,
        # randomly rotate images in the range (deg 0 to 180)
        rotation_range=0,
        # randomly shift images horizontally
        width_shift_range=0.1,
        # randomly shift images vertically
        height_shift_range=0.1,
        # set range for random shear
        shear_range=0.,
        # set range for random zoom
        zoom_range=0.,
        # set range for random channel shifts
        channel_shift_range=0.,
        # set mode for filling points outside the input boundaries
        fill_mode='nearest',
        # value used for fill_mode = "constant"
        cval=0.,
        # randomly flip images
        horizontal_flip=True,
        # randomly flip images
        vertical_flip=False,
        # set rescaling factor (applied before any other transformation)
        rescale=None,
        # set function that will be applied on each input
        preprocessing_function=None,
        # image data format, either "channels_first" or "channels_last"
        data_format=None,
        # fraction of images reserved for validation (strictly between 0 and 1)
        validation_split=0.0)

    # Compute quantities required for featurewise normalization
    # (std, mean, and principal components if ZCA whitening is applied).
    datagen.fit(train_images)

    # Fit the model on the batches generated by datagen.flow().
    model.fit_generator(datagen.flow(train_images, train_labelss, batch_size=BATCH_SIZE),
                        validation_data=(test_images, test_labelss),
                        epochs=EPOCHS, verbose=0, workers=1,
                        callbacks=callbacks, use_multiprocessing=False)
    
elapsed_time = time.time() - start_time
print("Elapsed time: {}".format(str(elapsed_time)))


scores = model.evaluate(test_images, test_labelss, verbose=1)
print('Test loss:', scores[0])
print('Test accuracy:', scores[1])

history_dict = model.history.history
loss_value = history_dict["loss"]
val_loss_values = history_dict["val_loss"]
acc= history_dict["acc"]
val_acc=history_dict["val_acc"]


epochs = range(1,len(loss_value)+1)

"""

plt.plot(epochs,loss_value,"bo",label="Train loss")
plt.plot(epochs, val_loss_values,"b" , label="acc loss")
plt.title("Train and acc loss")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.legend()
plt.show()

plt.plot(epochs,val_acc,"bo",label="Validation accuracy")
plt.plot(epochs, acc,"b" , label="actual accuracy")
plt.title("vali-model accuracy")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.legend()
plt.show()

#show 25 images and predict their labels..
for i in range(0,25):
    plt.imshow(testshow[i])
    plt.show()
    x=model.predict(test_images[[i]])
    if abs(x[0,0]-x[0,1])>0:
        a=np.argmax(max(x))
        if a==0:
            print("dead")
        elif a==1:
            print("alive")        
    else:
        print("not sure!")


#save and load model
model.save('canlitelef.h5')
from tensorflow.keras.models import load_model
model = load_model('canlitelef.h5') 
"""



