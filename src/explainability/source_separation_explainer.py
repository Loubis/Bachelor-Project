import os
import sys
import warnings
import librosa
import numpy as np
import itertools
from spleeter.separator import Separator
from spleeter.audio import STFTBackend
import tensorflow as tf
from sklearn.linear_model import LinearRegression

genres_dict = {
    'Hip-Hop': 0,
    'Pop': 1,
    'Rock': 2,
    'Folk': 3,
    'Experimental': 4,
    'Jazz': 5,
    'Electronic': 6,
    'International': 7,
    'Soul-RnB': 8,
    'Blues': 9,
    'Spoken': 10,
    'Country': 11,
    'Classical': 12,
    'Old-Time / Historic': 13,
    'Instrumental': 14,
    'Easy Listening': 15
}

channel_labels = ["vocals", "drums", "bass", "other"]

def load_file(file):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        audio, sr = librosa.load(file, mono=False, sr=44100, duration=30.0)

    # If loaded audio is only mono duplicate channel
    if audio.shape[0] != 2:
        audio = np.array([
            audio,
            audio
        ])

    if audio.shape[1] < 30*44100:
        audio = np.array([
            np.pad(audio[0], (0, 30*44100 - audio.shape[1]), mode='constant', constant_values=0),
            np.pad(audio[1], (0, 30*44100 - audio.shape[1]), mode='constant', constant_values=0)
        ])

    return audio


def slice_snippets(audio):
    # Slice audio in 10 second parts
    slice_length = 10 # seconds
    surplus = audio.shape[1] % (44100 * slice_length)
    number_of_snippets = audio.shape[1] // (44100 * slice_length)
    audio_snippets = np.array([
        np.array_split(audio[0][surplus:], number_of_snippets),
        np.array_split(audio[1][surplus:], number_of_snippets)
    ])

    # Prepare slices
    prepared_snippets = []
    for index in range(audio_snippets.shape[1]):
        prepared_snippets.append(
            np.array([
                audio_snippets[0][index],
                audio_snippets[1][index],
            ])
        )
    
    return prepared_snippets


def compute_source_separation(prepared_snippets):
    # Init Spleeter Separator
    seperator = Separator("spleeter:4stems", STFTBackend.TENSORFLOW, multiprocess=False)
    # Compute 
    source_seperation_slices = []
    for prepared_slice in prepared_snippets:
        prepared_slice = prepared_slice.reshape(prepared_slice.shape[1],prepared_slice.shape[0])
        source_seperation_slices.append(seperator.separate(prepared_slice, ""))
    
    return source_seperation_slices


def to_mono(waveform):
    if waveform.shape[0] != 2:
        waveform = waveform.reshape( 
            (waveform.shape[1], waveform.shape[0])
        )
    
    return librosa.to_mono(waveform)


def compute_spectrogramms(source_seperation_slices):
    spectrogram_slices = []
    for source_seperation_slice in source_seperation_slices:
        temp = {}
        for key, prediction in source_seperation_slice.items():
            temp[key] = librosa.power_to_db(
                librosa.feature.melspectrogram(
                    to_mono(prediction),
                    sr=44100,
                    n_fft=2048,
                    hop_length=1024
                ),
                ref=np.max
            )
        spectrogram_slices.append(temp)
    return spectrogram_slices


def load_tensorflow_model(model_path):
    return tf.keras.models.load_model(model_path, custom_objects={'tf': tf})


def explain_snippets(x_test, model):
    # Create Channel permutation to test and remove empty and full permutation
    l = [0,1]
    z_ = list(itertools.product(l, repeat=4))
    z_.remove((0,0,0,0))
    z_.remove((1,1,1,1))

    snippet_data = []
    for index, x in enumerate(x_test):
        print(f'Snippet number {index + 1}')
        Z_y = []
        # Get prediction for full input
        full_prediction = model.predict(np.array([x]))
        full_predictet_confidence = np.argmax(np.squeeze(full_prediction)) 
        full_predictet_label = list(genres_dict.keys())[list(genres_dict.values()).index(full_predictet_confidence)]

        # Create Sample and get label for prediction
        for permutation in list(z_):
            z = np.copy(x)
            for index, value in enumerate(permutation):
                if value == 0:
                    z[index] = np.full((z[index].shape[0], z[index].shape[1]), -80)
            z_prediction = model.predict(np.array([z]))
            
            # Get most confident value
            predictet_label = np.argmax(np.squeeze(z_prediction))
            Z_y.append(np.squeeze(z_prediction)[genres_dict[full_predictet_label]])
            print(f'Model predictet {list(genres_dict.keys())[list(genres_dict.values()).index(predictet_label)]} for permutation {permutation}; Target label with {np.squeeze(z_prediction)[genres_dict[full_predictet_label]]}')

        # Train explainable model for 
        reg = LinearRegression()
        reg.fit(z_, Z_y)

        snippet_data.append({
            "weights": reg.coef_,
            "label": full_predictet_label,
            "confidence": full_predictet_confidence
        })

    return snippet_data

def evalate_explanations(snippet_explanations):
    total_snippets = len(snippet_explanations)
    predictet_labels_count = {}
    for explanations in snippet_explanations:
        if explanations["label"] in predictet_labels_count:
            predictet_labels_count[explanations["label"]] = predictet_labels_count[explanations["label"]] + 1
        else:
            predictet_labels_count[explanations["label"]] = 0
    


def preprocess_instance_snippet(file_path, model_path):
    audio = load_file(file_path)
    snippets = slice_snippets(audio)
    source_separation_snippets = compute_source_separation(snippets)
    spectrogram_snippets = compute_spectrogramms(source_separation_snippets)
    model = load_tensorflow_model(model_path)
    snippet_explanations = explain_snippets(spectrogram_snippets, model)
    evalate_explanations(snippet_explanations)
    

