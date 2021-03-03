import tensorflow as tf
from tensorflow.keras import Input, Model
from tensorflow.keras.layers import (Conv2D, MaxPooling2D, Dense, LSTM,
                                     BatchNormalization, Activation, Reshape,
                                     Add, Concatenate, Lambda)

import parameter as params

class Recognizer:
    def __init__(self, training):
        self.input_shape = (params.IMG_W, params.IMG_H, 3)
        self.class_num = params.CLASS_NUM
        self.train_batch_size = params.TRAIN_BATCH_SIZE
        self.val_batch_size = params.VAL_BATCH_SIZE
        self.downsample_factor = params.DOWNSAMPLE_FACTOR
        self.max_text_len = params.MAX_TEXT_LEN

    @staticmethod
    def ctc_lambda_func(args):
        y_pred, labels, input_length, label_length = args
        # the 2 is critical here since the first couple outputs of the RNN
        # tend to be garbage:
        y_pred = y_pred[:, 2:, :]
        return tf.nn.ctc_loss(labels, y_pred, input_length, label_length)

    def build_model(self, training):
        # Input layer
        inputs = Input(name='input', shape=self.input_shape, dtype='float32') # (None, 128, 64, 1)

        # Convolution layer (VGG)
        inner = Conv2D(16, (3, 3), padding='same', kernel_initializer='he_normal')(inputs)  # (None, 128, 64, 64)
        inner = BatchNormalization()(inner)
        inner = Activation('relu')(inner)
        inner = MaxPooling2D(pool_size=(2, 2))(inner)  # (None,64, 32, 64)

        inner = Conv2D(32, (3, 3), padding='same', kernel_initializer='he_normal')(inner)
        inner = BatchNormalization()(inner)
        inner = Activation('relu')(inner)
        inner = MaxPooling2D(pool_size=(2, 2))(inner)  # (None, 32, 16, 128)

        inner = Conv2D(32, (3, 3), padding='same', kernel_initializer='he_normal')(inner)
        inner = BatchNormalization()(inner)
        inner = Activation('relu')(inner)
        inner = MaxPooling2D(pool_size=(1, 2))(inner)

        inner = Conv2D(64, (3, 3), padding='same', kernel_initializer='he_normal')(inner)
        inner = BatchNormalization()(inner)
        inner = Activation('relu')(inner)
        inner = MaxPooling2D(pool_size=(1, 2))(inner)

        inner = Conv2D(64, (3, 3), padding='same', kernel_initializer='he_normal')(inner)
        inner = BatchNormalization()(inner)
        inner = Activation('relu')(inner)
        inner = MaxPooling2D(pool_size=(1, 2))(inner)

        inner = Conv2D(128, (3, 3), padding='same', kernel_initializer='he_normal')(inner)
        inner = BatchNormalization()(inner)
        inner = Activation('relu')(inner)
        inner = MaxPooling2D(pool_size=(1, 2))(inner)

        # CNN to RNN
        inner = Reshape(target_shape=((32, -1)))(inner)  # (None, 32, 2048)
        inner = Dense(128, activation='relu', kernel_initializer='he_normal')(inner)  # (None, 32, 64)

        # RNN layer
        lstm_1 = LSTM(128, return_sequences=True, kernel_initializer='he_normal')(inner)  # (None, 32, 512)
        lstm_1b = LSTM(128, return_sequences=True, go_backwards=True, kernel_initializer='he_normal')(inner)
        lstm1_merged = Add([lstm_1, lstm_1b])  # (None, 32, 512)

        lstm_2 = LSTM(128, return_sequences=True, kernel_initializer='he_normal')(lstm1_merged)  # (None, 32, 512)
        lstm_2b = LSTM(128, return_sequences=True, go_backwards=True, kernel_initializer='he_normal')(lstm1_merged)
        lstm2_merged = Concatenate([lstm_2, lstm_2b])  # (None, 32, 512)

        # transforms RNN output to character activations:
        inner = Dense(self.class_num, kernel_initializer='he_normal')(lstm2_merged) #(None, 32, 63)
        y_pred = Activation('softmax', name='softmax')(inner)

        labels = Input(name='the_labels', shape=[self.max_text_len], dtype='float32') # (None ,8)
        input_length = Input(name='input_length', shape=[1], dtype='int64')     # (None, 1)
        label_length = Input(name='label_length', shape=[1], dtype='int64')     # (None, 1)

        # Keras doesn't currently support loss funcs with extra parameters
        # so CTC loss is implemented in a lambda layer
        loss_out = Lambda(self.ctc_lambda_func, output_shape=(1,), name='ctc')([y_pred, labels, input_length, label_length]) #(None, 1)

        if training:
            return Model(inputs=[inputs, labels, input_length, label_length], outputs=loss_out)
        else:
            return Model(inputs=[inputs], outputs=y_pred)
