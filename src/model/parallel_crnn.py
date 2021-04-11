from numpy.core.fromnumeric import shape, squeeze
from .abstract_model import AbstractModel

import os
import tensorflow as tf
import numpy as np
from tqdm import tqdm
from glob import glob
from datetime import datetime
import sklearn

from tensorflow.keras.models import Model
from tensorflow.keras.constraints import max_norm
from tensorflow.keras.optimizers import (
    Adam,
)
from tensorflow.keras.layers import (
    Dense,
    Activation,
    BatchNormalization,
    Input,
    Conv2D,
    MaxPool2D,
    Bidirectional,
    GRU,
    Flatten,
    Lambda,
    concatenate,
    Dropout,
)
from tensorflow.keras.callbacks import ModelCheckpoint, TensorBoard, ReduceLROnPlateau


class ParallelCRNN(AbstractModel):

    def __init__(self):
        self._model = self._create_parallel_cnn_birnn_model()
        self._batch_size = 64
        self._epoch_count = 70
        self._log_time = datetime.now().strftime('%d-%m-%Y-%H-%M-%S')
        if not os.path.exists('./logs'):
            os.makedirs('./logs')
        os.makedirs(f'./logs/ParallelCRNN_{self._log_time}')


    def train(self):
        x_train, y_train, x_valid, y_valid = [], [], [], []
        
        for np_name in tqdm(glob('/data/processed/gtzan/arr_train_*.npz'), ncols=100):
            npzfile = np.load(np_name)
            x_train.append(npzfile['arr_0'])
            y_train.append(npzfile['arr_1'])

        x_train = np.concatenate(x_train, axis=0)
        y_train = np.concatenate(y_train, axis=0)
        x_train, y_train = sklearn.utils.shuffle(x_train, y_train)

        for np_name in tqdm(glob('/data/processed/gtzan/arr_validate_*.npz'), ncols=100):
            npzfile = np.load(np_name)
            x_valid.append(npzfile['arr_0'])
            y_valid.append(npzfile['arr_1'])

        x_valid = np.concatenate(x_valid, axis=0)
        y_valid = np.concatenate(y_valid, axis=0)
        x_valid, y_valid = sklearn.utils.shuffle(x_valid, y_valid)

        tb_callback = TensorBoard(
            log_dir = f'./logs/{self._log_time}/tensorboard',
            histogram_freq = 1,
            write_graph = True,
            write_grads = False,
            write_images = False,
            embeddings_freq = 0,
            embeddings_layer_names = None,
            embeddings_metadata = None,
        )

        checkpoint_callback = ModelCheckpoint(
            f'./logs/{self._log_time}/weights.best.h5',
            monitor='val_accuracy',
            verbose=1,
            save_best_only=True,
            mode='max',
        )

        reducelr_callback = ReduceLROnPlateau(
            monitor='val_accuracy', factor=0.5, patience=10, min_delta=0.01, verbose=1
        )

        callbacks_list = [reducelr_callback, checkpoint_callback, tb_callback]
        print('Training...')
        self._model.fit(
            [ np.squeeze(arr, axis=1) for arr in np.split(np.expand_dims(x_train, axis=-1), 5 , axis=1) ],
            y_train,
            batch_size=self._batch_size,
            epochs=self._epoch_count,
            validation_data=( 
                [ np.squeeze(arr, axis=1) for arr in np.split(np.expand_dims(x_valid, axis=-1), 5 , axis=1) ],
                y_valid
            ),
            verbose=1,
            callbacks=callbacks_list,
        )

        os.mkdir(f'./logs/{self._log_time}/trained_model')
        self._model.save(f'./logs/{self._log_time}/trained_model')


    def evaluate(self):
        x_test, y_test = [], []

        for np_name in tqdm(glob('//data/processed/gtzan/arr_test_*.npz'), ncols=100):
            npzfile = np.load(np_name)
            x_test.append(npzfile['arr_0'])
            y_test.append(npzfile['arr_1'])

        x_test = np.concatenate(x_test, axis=0)
        y_test = np.concatenate(y_test, axis=0)
	    
        score = self._model.evaluate([ np.squeeze(arr, axis=1) for arr in np.split(np.expand_dims(x_test, axis=-1), 5 , axis=1) ], y_test)
        
        file = open(f'./logs/{self._log_time}/evaluation.log', 'w')
        file.write(f'Test loss: {score[0]} / Test accuracy: {score[1]}')




    def _create_cnn_block(self, Input_Layer):
        CNN_Block = Conv2D(
            filters=16,
            kernel_size=[3, 3],
            padding='same',
        )(Input_Layer)
        CNN_Block = BatchNormalization()(CNN_Block)
        CNN_Block = Activation('relu')(CNN_Block)
        CNN_Block = MaxPool2D(pool_size=(2, 2), strides=(2, 2))(CNN_Block)
        CNN_Block = Dropout(0.2)(CNN_Block)

        CNN_Block = Conv2D(
            filters=32,
            kernel_size=[3, 3],
            padding='same',
        )(CNN_Block)
        CNN_Block = BatchNormalization()(CNN_Block)
        CNN_Block = Activation('relu')(CNN_Block)
        CNN_Block = MaxPool2D(pool_size=(2, 2), strides=(2, 2))(CNN_Block)
        CNN_Block = Dropout(0.2)(CNN_Block)

        CNN_Block = Conv2D(
            filters=64,
            kernel_size=[3, 3],
            padding='same',
        )(CNN_Block)
        CNN_Block = BatchNormalization()(CNN_Block)
        CNN_Block = Activation('relu')(CNN_Block)
        CNN_Block = MaxPool2D(pool_size=(2, 2), strides=(2, 2))(CNN_Block)
        CNN_Block = Dropout(0.2)(CNN_Block)

        CNN_Block = Conv2D(
            filters=128,
            kernel_size=[3, 3],
            padding='same',
        )(CNN_Block)
        CNN_Block = BatchNormalization()(CNN_Block)
        CNN_Block = Activation('relu')(CNN_Block)
        CNN_Block = MaxPool2D(pool_size=(4, 4), strides=(4, 4))(CNN_Block)
        CNN_Block = Dropout(0.2)(CNN_Block)

        CNN_Block = Conv2D(
            filters=64,
            kernel_size=[3, 3],
            padding='same',
        )(CNN_Block)
        CNN_Block = (BatchNormalization())(CNN_Block)
        CNN_Block = (Activation('relu'))(CNN_Block)
        CNN_Block = MaxPool2D(pool_size=(4, 4), strides=(4, 4))(CNN_Block)
        CNN_Block = Dropout(0.2)(CNN_Block)

        CNN_Block = Flatten()(CNN_Block)

        return CNN_Block


    def _create_birnn_block(self, Input_Layer):
        BiRNN_Block = MaxPool2D(pool_size=(1, 4), strides=(1, 4))(Input_Layer)

        BiRNN_Block = Lambda(lambda x: tf.keras.backend.squeeze(x, axis=-1))(BiRNN_Block)
        BiRNN_Block = Bidirectional(GRU(128))(BiRNN_Block)
        BiRNN_Block = Dropout(0.5)(BiRNN_Block)

        return BiRNN_Block


    # Classification Block
    def _create_classification_block(self, CNN_Block, BiRNN_Block):
        Classification_Block = concatenate([CNN_Block, BiRNN_Block], axis=-1)
        Classification_Block = Dropout(0.5)(Classification_Block)

        Output_Layer = Dense(896, activation='relu')(
            Classification_Block
        )

        return Output_Layer


    def _create_parallel_cnn_birnn_model(self):
        print('Creating model...')
        Input_List = []
        Sub_Net_Outputs = []

        for _ in range(0,5):            
            Input_Layer = Input((128, 130, 1))
            Input_List.append(Input_Layer)
            Sub_Net_Outputs.append(self._create_cnn_block(Input_Layer)) 
            Sub_Net_Outputs.append(self._create_birnn_block(Input_Layer))
            #Sub_Net_Outputs.append(self._create_classification_block(self._create_cnn_block(Input_Layer), self._create_birnn_block(Input_Layer)))

        encoded_labels = np.load('/data/processed/gtzan/encoded_labels.npy')

        Final_Classification_Block = concatenate(Sub_Net_Outputs, axis=-1)
        Final_Classification_Block = Dropout(0.5)(Final_Classification_Block)
        Output_Layer = Dense(encoded_labels.shape[0], activation='softmax')(
            Final_Classification_Block
        )

        model = Model(Input_List, Output_Layer)

        opt = Adam(learning_rate=0.002)
        print('Compiling Model...')
        model.compile(
            loss='sparse_categorical_crossentropy', optimizer=opt, metrics=['accuracy']
        )

        model.summary()
        tf.keras.utils.plot_model(model, show_shapes=True, show_layer_names=True, rankdir='TB', expand_nested=True)

        return model
