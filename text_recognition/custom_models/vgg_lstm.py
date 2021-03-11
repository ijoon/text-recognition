import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))

import numpy as np
import tensorflow as tf
from tensorflow.keras import Input, Model
from tensorflow.keras.layers import (Conv2D, MaxPooling2D, Dense, LSTM,
                                     BatchNormalization, Activation, Reshape,
                                     add, concatenate, Lambda)

import train.data_utils as du
from config import cfg


class VGGLSTM:
    def __init__(self,
                 input_shape_hwc: '(img_w, img_h, channel)',
                 class_num: int,
                 max_text_len: int,
                 downsample_factor: int,
                 letters: str,
                 dataset_dir: str,
                 batch_size: int):

        self.input_shape_hwc = input_shape_hwc
        self.class_num = class_num
        self.max_text_len = max_text_len
        self.downsample_factor = downsample_factor
        self.letters = letters

        self.dataset_dir = dataset_dir
        self.batch_size = batch_size


    @staticmethod
    def ctc_func(args):
        y_pred, labels, input_length, label_length = args
        # the 2 is critical here since the first couple outputs of the RNN
        # tend to be garbage:
        y_pred = y_pred[:, 2:, :]
        return tf.keras.backend.ctc_batch_cost(
            labels, y_pred, input_length, label_length)

    def build_model(self, training: bool):

        # Input layer
        inputs = Input(name='inputs', shape=self.input_shape_hwc, 
                       dtype='float32') 

        # Convolution layer (VGG)
        y = Conv2D(16, (3, 3), padding='same', 
                   kernel_initializer='he_normal')(inputs) 
        y = BatchNormalization()(y)
        y = Activation('relu')(y)
        y = MaxPooling2D(pool_size=(2, 2))(y)

        y = Conv2D(32, (3, 3), padding='same', 
                   kernel_initializer='he_normal')(y)
        y = BatchNormalization()(y)
        y = Activation('relu')(y)
        y = MaxPooling2D(pool_size=(2, 2))(y)

        y = Conv2D(32, (3, 3), padding='same', 
                   kernel_initializer='he_normal')(y)
        y = BatchNormalization()(y)
        y = Activation('relu')(y)
        y = MaxPooling2D(pool_size=(1, 2))(y)

        y = Conv2D(64, (3, 3), padding='same', 
                   kernel_initializer='he_normal')(y)
        y = BatchNormalization()(y)
        y = Activation('relu')(y)
        y = MaxPooling2D(pool_size=(1, 2))(y)

        y = Conv2D(64, (3, 3), padding='same', 
                   kernel_initializer='he_normal')(y)
        y = BatchNormalization()(y)
        y = Activation('relu')(y)
        y = MaxPooling2D(pool_size=(1, 2))(y)

        y = Conv2D(128, (3, 3), padding='same', 
                   kernel_initializer='he_normal')(y)
        y = BatchNormalization()(y)
        y = Activation('relu')(y)
        y = MaxPooling2D(pool_size=(1, 2))(y)

        # CNN to RNN
        y = Reshape(target_shape=((32, -1)))(y)  
        y = Dense(128, activation='relu', kernel_initializer='he_normal')(y)  

        # RNN layer
        lstm_1 = LSTM(128, return_sequences=True, 
                      kernel_initializer='he_normal')(y)
        lstm_1b = LSTM(128, return_sequences=True, go_backwards=True, 
                       kernel_initializer='he_normal')(y)
        lstm1_merged = add([lstm_1, lstm_1b])

        lstm_2 = LSTM(128, return_sequences=True, 
                      kernel_initializer='he_normal')(lstm1_merged)
        lstm_2b = LSTM(128, return_sequences=True, go_backwards=True, 
                       kernel_initializer='he_normal')(lstm1_merged)
        lstm2_merged = concatenate([lstm_2, lstm_2b])

        # transforms RNN output to character activations:
        y = Dense(self.class_num, kernel_initializer='he_normal')(lstm2_merged)
        y_pred = Activation('softmax', name='softmax')(y)

        labels = Input(name='labels', shape=[self.max_text_len], dtype='int64')
        input_length = Input(name='input_length', shape=[1], dtype='int64')
        label_length = Input(name='label_length', shape=[1], dtype='int64')

        # Keras doesn't currently support loss funcs with extra parameters
        # so CTC loss is implemented in a lambda layer
        loss_out = Lambda(self.ctc_func, output_shape=(1,), name='ctc')(
            [y_pred, labels, input_length, label_length]) #(None, 1)

        if training:
            return Model(inputs=[inputs, labels, input_length, label_length], 
                         outputs=loss_out)
        else:
            return Model(inputs=[inputs],
                         outputs=y_pred)

    def train(self):
        model = self.build_model(training=True)
        model.compile(
            optimizer='Adam', 
            loss={'ctc': lambda y_true, y_pred: y_pred})

        train_ds, valid_ds = self.batch_generator()
        model.fit(
            x=train_ds,
            epochs=3,
            validation_data=valid_ds
        )
    
    def predict(self):
        pass

    def get_text_result(self, img) -> str:
        pass
    
    def batch_generator(self):
        image_paths, labels = du.get_image_paths_and_string_labels(
            directory=self.dataset_dir,
            allow_image_formats=('.jpeg', '.jpg'),
            letters=self.letters,
            label_length=self.max_text_len)

        total_ds = du.get_tf_dataset_for_images_and_string_labels(
            image_paths=image_paths, 
            labels=labels, 
            image_size_hw=[self.input_shape_hwc[0], self.input_shape_hwc[1]])

        train_ds, valid_ds = du.split_train_valid_for_tf_dataset(
            total_ds, 
            valid_ratio=0.2, 
            shuffle=True, 
            cache=True)
        
        train_ds = (
            train_ds
            .map(
                lambda image, label: du._map_preprocess_data(
                    image=image, 
                    label=label, 
                    augmentation=True, 
                    normalization=True),
                num_parallel_calls=tf.data.experimental.AUTOTUNE)
            .shuffle(train_ds.cardinality().numpy())
            .batch(self.batch_size, drop_remainder=True)
            .map(
                lambda images, labels: du._map_batch_for_ctc_loss(
                    images=images,
                    labels=labels,
                    image_w=self.input_shape_hwc[1],
                    downsample_factor=self.downsample_factor,
                    batch_size=self.batch_size,
                    text_length=self.max_text_len),
                num_parallel_calls=tf.data.experimental.AUTOTUNE)
            .prefetch(tf.data.experimental.AUTOTUNE)
        )

        valid_ds = (
            valid_ds
            .map(
                lambda image, label: du._map_preprocess_data(
                    image=image, 
                    label=label, 
                    augmentation=False, 
                    normalization=True),
                num_parallel_calls=tf.data.experimental.AUTOTUNE)
            .batch(self.batch_size, drop_remainder=True)
            .map(
                lambda images, labels: du._map_batch_for_ctc_loss(
                    images=images,
                    labels=labels,
                    image_w=self.input_shape_hwc[1],
                    downsample_factor=self.downsample_factor,
                    batch_size=self.batch_size,
                    text_length=self.max_text_len),
                num_parallel_calls=tf.data.experimental.AUTOTUNE)
            .prefetch(tf.data.experimental.AUTOTUNE)
        )

        return train_ds, valid_ds
                

if __name__ == '__main__':
    vgg_lstm = VGGLSTM(
        input_shape_hwc=(cfg['img_h'], cfg['img_w'], cfg['channel']),
        class_num=cfg['class_num'],
        max_text_len=cfg['max_text_len'],
        letters=cfg['char_vector'],
        dataset_dir=cfg['dataset_dir'],
        batch_size=cfg['train_batch_size'],
        downsample_factor=cfg['downsample_factor'])
        
    vgg_lstm.train()

