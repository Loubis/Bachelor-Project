from tensorflow.python.keras import activations
from tensorflow.python.keras.engine import input_layer
from tensorflow.python.keras.layers.core import Flatten
from .abstract_model import AbstractModel

import os
import tensorflow as tf
import numpy as np
from tqdm import tqdm
from glob import glob
from datetime import datetime
import sklearn

from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import (
    Adam,
)
from tensorflow.keras.layers import (
    Dense,
    concatenate,
    Dropout,
)
from tensorflow.keras.callbacks import ModelCheckpoint, TensorBoard, ReduceLROnPlateau
from tensorflow.keras.regularizers import l2

class ResNet50V2(AbstractModel):

    def __init__(self, dataset_path):
        self._dataset_path = dataset_path
        self._model = self.compile_model()
        self._batch_size = 64
        self._epoch_count = 70
        self._log_time = datetime.now().strftime('%d-%m-%Y-%H-%M-%S')
        if not os.path.exists('./logs'):
            os.makedirs('./logs')
        os.makedirs(f'./logs/ResNet50V2_{self._log_time}')


    def compile_model(self):
        print('Creating model...')
        encoded_labels = np.load(f'{self._dataset_path}encoded_labels.npy')
        Input_List = []
        Sub_Net_Outputs = []

        for _ in range(0,5):
            Input_layer = tf.keras.layers.Input((128, 130, 1))
            base_model = tf.keras.applications.ResNet50V2(include_top=False, weights=None, input_tensor=Input_layer)
            Input_List.append(base_model.input)
            Sub_Net_Outputs.append(Flatten()(base_model.outputs)) 

        print(Sub_Net_Outputs)
        Network = concatenate(Sub_Net_Outputs, axis=-1)
        Network = Dropout(0.2)(Network)
        Output_Layer = Dense(encoded_labels.shape[0], activation='softmax')(Network)

        model = Model(Input_List, Output_Layer)

        opt = Adam(learning_rate=0.002)
        print('Compiling Model...')
        model.compile(
            loss='sparse_categorical_crossentropy', optimizer=opt, metrics=['accuracy']
        )

        model.summary()
        tf.keras.utils.plot_model(model, show_shapes=True, show_layer_names=True, rankdir='TB', expand_nested=True)

        return model


    def train(self):
        x_train, y_train, x_valid, y_valid = [], [], [], []
        
        for np_name in tqdm(glob(f'{self._dataset_path}arr_train_*.npz'), ncols=100):
            npzfile = np.load(np_name)
            x_train.append(npzfile['arr_0'])
            y_train.append(npzfile['arr_1'])

        x_train = np.concatenate(x_train, axis=0)
        y_train = np.concatenate(y_train, axis=0)
        x_train, y_train = sklearn.utils.shuffle(x_train, y_train)

        for np_name in tqdm(glob(f'{self._dataset_path}arr_train_*.npz'), ncols=100):
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

        for np_name in tqdm(glob(f'{self._dataset_path}arr_test_*.npz'), ncols=100):
            npzfile = np.load(np_name)
            x_test.append(npzfile['arr_0'])
            y_test.append(npzfile['arr_1'])

        x_test = np.concatenate(x_test, axis=0)
        y_test = np.concatenate(y_test, axis=0)
	    
        score = self._model.evaluate([ np.squeeze(arr, axis=1) for arr in np.split(np.expand_dims(x_test, axis=-1), 5 , axis=1) ], y_test)
        
        file = open(f'./logs/{self._log_time}/evaluation.log', 'w')
        file.write(f'Test loss: {score[0]} / Test accuracy: {score[1]}')
