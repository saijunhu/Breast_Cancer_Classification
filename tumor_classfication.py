import tensorflow as tf
from tensorflow.python.keras import layers
from tensorflow.python.keras import models
from tensorflow.python.keras.applications.resnet50 import ResNet50
from tensorflow.python.keras.callbacks import EarlyStopping, ReduceLROnPlateau,CSVLogger
from tensorflow.python.keras.layers import (
    AveragePooling2D
)
from tensorflow.python.keras.preprocessing.image import ImageDataGenerator
from tensorflow.python.keras.regularizers import l2
from tensorflow.python.keras.optimizers import Adam


# get access to google drive
# from google.colab import drive
# drive.mount('/content/drive')

INPUT_SHAPE = (224, 224, 3)
IMG_WIDTH = 224
IMG_HEIGHT = 224
EPOCHS = 128
NB_CLASSES = 2  # 类别
BATCH_SIZE = 32
LEARN_RATE = 3e-3
train_path = r'/content/drive/My Drive/data/train/'
val_path = r'/content/drive/My Drive/data/test/'
base_path = r'/content/drive/My Drive/data/'
NUM_TRAIN = 1249
NUM_TEST = 745
SEED = 11


# 460 700

def create_model():
    dropout = 0.5
    reg = l2(0.01)
    opt = Adam(lr=LEARN_RATE)

    conv_base = ResNet50(weights='imagenet',
                         include_top=False,
                         input_shape=INPUT_SHAPE)
    conv_base.trainable = False
    model = models.Sequential()
    model.add(conv_base)
    model.add(AveragePooling2D(pool_size=(2, 2)))
    model.add(layers.Flatten())
    model.add(layers.Dropout(dropout))
    model.add(layers.Dense(512, activation='relu',
                           kernel_initializer='uniform',
                           kernel_regularizer=reg))
    model.add(layers.Dropout(dropout))
    model.add(layers.Dense(512, activation='relu',
                           kernel_initializer='uniform',
                           kernel_regularizer=reg))
    model.add(layers.Dropout(dropout))
    model.add(layers.Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy',
                  optimizer=opt,
                  metrics=['accuracy'])
    model.summary()
    # tf.keras.utils.plot_model(model, to_file='model_plot.png', show_shapes=True, show_layer_names=True)
    return model


def get_datagen():
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
        rotation_range=45,
        # randomly shift images horizontally
        width_shift_range=0.2,
        # randomly shift images vertically
        height_shift_range=0.2,
        # set range for random shear
        shear_range=0.1,
        # set range for random zoom
        zoom_range=0.2,
        # set range for random channel shifts
        channel_shift_range=0.,
        # set mode for filling points outside the input boundaries
        fill_mode='nearest',
        # value used for fill_mode = "constant"
        cval=0.,
        # randomly flip images
        horizontal_flip=True,
        # randomly flip images
        vertical_flip=True,
        # set rescaling factor (applied before any other transformation)
        rescale=1. / 255,
        # set function that will be applied on each input
        preprocessing_function=None,
        # image data format, either "channels_first" or "channels_last"
        data_format=None,
        # fraction of images reserved for validation (strictly between 0 and 1)
        validation_split=0.0)
    return datagen


def get_callbacks():
    earlystop = EarlyStopping(monitor='val_loss', min_delta=0.001, patience=10)
    lr_reduction = ReduceLROnPlateau(monitor='val_loss', patience=3, verbose=1, factor=0.5, min_lr=1e-5)
    csv_log = CSVLogger(base_path+'log.csv')
    # tensorboard = TensorBoard(log_dir='log(logs)')
    callbacks = [earlystop, lr_reduction, csv_log]
    return callbacks


if __name__ == "__main__":
    model = create_model()
    train_datagen = get_datagen()
    val_datagen = ImageDataGenerator(rescale=1. / 255)
    train_generator = train_datagen.flow_from_directory(train_path,
                                                        batch_size=BATCH_SIZE,
                                                        target_size=(IMG_WIDTH, IMG_HEIGHT),
                                                        seed=SEED,
                                                        class_mode='binary')
    val_generator = val_datagen.flow_from_directory(val_path,
                                                    batch_size=BATCH_SIZE,
                                                    target_size=(IMG_WIDTH, IMG_HEIGHT),
                                                    seed=SEED,
                                                    class_mode='binary')
    history = model.fit_generator(train_generator, epochs=EPOCHS,
                                  callbacks=get_callbacks(),
                                  steps_per_epoch=NUM_TRAIN // BATCH_SIZE,
                                  validation_data=val_generator,
                                  validation_steps=NUM_TEST // BATCH_SIZE
                                  )

    model.save_weights('demo.h5')
