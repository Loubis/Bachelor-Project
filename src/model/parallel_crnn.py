from numpy.core.fromnumeric import shape, squeeze
from .abstract_model import AbstractModel

import os
import json
import tensorflow as tf
import numpy as np
from tqdm import tqdm
from glob import glob
from datetime import datetime
import sklearn
from sklearn.metrics import classification_report


from tensorflow.keras.models import Model
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

    def __init__(self, base_path, dataset):
        self._batch_size = 32
        self._epoch_count = 100

        self._dataset_path = base_path + dataset
        self._metadata = json.load(open(f'{self._dataset_path}/metadata.json'))

        self._log_time = datetime.now().strftime('%d-%m-%Y-%H-%M-%S')
        self._log_path = f'./logs/ParallelCRNN/{dataset}/{self._log_time}'
        os.makedirs(self._log_path)

        self._model = self._create_parallel_cnn_birnn_model()


    def train(self):
        x_train, y_train, x_valid, y_valid = [], [], [], []
        
        for file in tqdm(glob(f'{self._dataset_path}/arr_train_*.npz'), ncols=100):
            npzfile = np.load(file)
            x_train.append(npzfile['arr_0'])
            y_train.append(npzfile['arr_1'])
        
        x_train = np.concatenate(x_train, axis=0)
        y_train = np.concatenate(y_train, axis=0)
        x_train, y_train = sklearn.utils.shuffle(x_train, y_train)

        for file in tqdm(glob(f'{self._dataset_path}/arr_validate_*.npz'), ncols=100):
            npzfile = np.load(file)
            x_valid.append(npzfile['arr_0'])
            y_valid.append(npzfile['arr_1'])

        x_valid = np.concatenate(x_valid, axis=0)
        y_valid = np.concatenate(y_valid, axis=0)
        x_valid, y_valid = sklearn.utils.shuffle(x_valid, y_valid)

        tb_callback = TensorBoard(
            log_dir = f'{self._log_path}/tensorboard',
            histogram_freq = 1,
            write_graph = True,
            write_grads = False,
            write_images = False,
            embeddings_freq = 0,
            embeddings_layer_names = None,
            embeddings_metadata = None,
        )

        checkpoint_callback = ModelCheckpoint(
            f'{self._log_path}/weights.best.h5',
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
            [ np.squeeze(arr, axis=1) for arr in np.split(np.expand_dims(x_train, axis=-1), self._metadata['split_count'], axis=1) ],
            y_train,
            batch_size=self._batch_size,
            epochs=self._epoch_count,
            validation_data=( 
                [ np.squeeze(arr, axis=1) for arr in np.split(np.expand_dims(x_valid, axis=-1), self._metadata['split_count'], axis=1) ],
                y_valid
            ),
            verbose=1,
            callbacks=callbacks_list,
        )

        os.mkdir(f'{self._log_path}/trained_model')
        self._model.save(f'{self._log_path}/trained_model')


    def evaluate(self):
        x_test, y_test = [], []

        for file in tqdm(glob(f'{self._dataset_path}/arr_test_*.npz'), ncols=100):
            npzfile = np.load(file)
            x_test.append(npzfile['arr_0'])
            y_test.append(npzfile['arr_1'])

        x_test = np.concatenate(x_test, axis=0)
        y_test = np.concatenate(y_test, axis=0)
	    
        score = self._model.evaluate([ np.squeeze(arr, axis=1) for arr in np.split(np.expand_dims(x_test, axis=-1), self._metadata['split_count'], axis=1) ], y_test)
        y_predictions = self._model.predict([ np.squeeze(arr, axis=1) for arr in np.split(np.expand_dims(x_test, axis=-1), self._metadata['split_count'], axis=1) ])
        y_predictions = np.argmax(y_predictions, axis=1)

        with open(f'{self._log_path}/evaluation.log', 'w') as file:
            file.write(f'Test loss: {score[0]} / Test accuracy: {score[1]}\n\n')
            file.write(classification_report(y_test, y_predictions, target_names=self._metadata['labels'].keys()))


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


    def _create_parallel_cnn_birnn_model(self):

        print('Creating model...')
        Input_List = []
        Sub_Net_Outputs = []

    
        for _ in range(0, self._metadata['split_count']):            
            Input_Layer = Input(
                (
                    self._metadata['data_shape'][0],
                    self._metadata['data_shape'][1],
                    1
                )
            )
            Input_List.append(Input_Layer)
            Sub_Net_Outputs.append(self._create_cnn_block(Input_Layer)) 
            Sub_Net_Outputs.append(self._create_birnn_block(Input_Layer))


        Final_Classification_Block = concatenate(Sub_Net_Outputs, axis=-1)
        Final_Classification_Block = Dropout(0.5)(Final_Classification_Block)
        Output_Layer = Dense(self._metadata['label_count'], activation='softmax')(
            Final_Classification_Block
        )

        model = Model(Input_List, Output_Layer)

        opt = Adam(learning_rate=0.002)
        print('Compiling Model...')
        model.compile(
            loss='sparse_categorical_crossentropy', optimizer=opt, metrics=['accuracy']
        )

        model.summary()
        tf.keras.utils.plot_model(model,to_file=f'{self._log_path}/model.png', show_shapes=True, show_layer_names=True, rankdir='TB', expand_nested=True)

        return model
