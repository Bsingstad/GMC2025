#!/usr/bin/env python

# Edit this script to add your team's code. Some functions are *required*, but you can edit most parts of the required functions,
# change or remove non-required functions, and add your own functions.

################################################################################
#
# Optional libraries, functions, and variables. You can change or remove them.
#
################################################################################

import joblib
import numpy as np
import os
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
import sys
from scipy import signal
import wfdb
import tensorflow as tf
from tqdm import tqdm

from helper_code import *

################################################################################
#
# Required functions. Edit these functions to add your code, but do not change the arguments for the functions.
#
################################################################################

# Train your models. This function is *required*. You should edit this function to add your code, but do *not* change the arguments
# of this function. If you do not train one of the models, then you can return None for the model.

# Train your model.
def train_model(data_folder, model_folder, verbose):
    NEW_FS = 100
    # Find the data files.
    if verbose:
        print('Finding the Challenge data...')

    records = find_records(data_folder)
    num_records = len(records)

    if num_records == 0:
        raise FileNotFoundError('No data were provided.')

    # Extract the features and labels from the data.
    if verbose:
        print('Extracting features and labels from the data...')

    features = np.zeros((num_records, 6), dtype=np.float64)
    X_data = np.zeros((num_records, 1000, 12), dtype=np.float64)
    labels = np.zeros(num_records, dtype=bool)

    # Iterate over the records.
    for i in tqdm(range(num_records)):
        if verbose:
            width = len(str(num_records))
            print(f'- {i+1:>{width}}/{num_records}: {records[i]}...')

        record = os.path.join(data_folder, records[i])
        features[i] = extract_features(record)
        ecg, text= load_signals(record)
        #text = load_text(record)
        fs = int(text["fs"])
        fs_ratio = NEW_FS/fs
        ecg_resamp = signal.resample(ecg,int(ecg.shape[0]*fs_ratio), axis=0)
        ecg_pad = tf.keras.utils.pad_sequences(
            np.moveaxis(ecg_resamp,0,-1),
            maxlen=1000,
            dtype='int32',
            padding='post',
            truncating='post',
            value=0.0
        )
        ecg_pad = np.moveaxis(ecg_pad,0,-1)
        X_data[i] = ecg_pad

        labels[i] = load_label(record)


    # Train the models.
    if verbose:
        print('Training the model on the data...')
    
    print("ECG array shape: ", X_data.shape)
    
    # This very simple model trains a random forest model with very simple features.

    # Define the parameters for the random forest classifier and regressor.
    #n_estimators = 12  # Number of trees in the forest.
    #max_leaf_nodes = 34  # Maximum number of leaf nodes in each tree.
    #random_state = 56  # Random state; set for reproducibility.

    # Fit the model.
    #model = RandomForestClassifier(
    #    n_estimators=n_estimators, max_leaf_nodes=max_leaf_nodes, random_state=random_state).fit(features, labels)
    
    model = build_model((1000,12), 1)
    history = model.fit(X_data, labels, epochs=1, batch_size=32, verbose=1)

    # Create a folder for the model if it does not already exist.
    os.makedirs(model_folder, exist_ok=True)

    # Save the model.
    save_model(model_folder, model)

    if verbose:
        print('Done.')
        print()

# Load your trained models. This function is *required*. You should edit this function to add your code, but do *not* change the
# arguments of this function. If you do not train one of the models, then you can return None for the model.
def load_model(model_folder, verbose):
    model_filename = os.path.join(model_folder, 'model.keras.h5')
    model = tf.keras.models.load_model(model_filename)
    return model

# Run your trained model. This function is *required*. You should edit this function to add your code, but do *not* change the
# arguments of this function.
def run_model(record, model, verbose):
    NEW_FS = 100
    # Load the model.
    
    # Extract the features.
    features = extract_features(record)
    features = features.reshape(1, -1)
    ecg, text= load_signals(record)
    #text = load_text(record)
    fs = int(text["fs"])
    fs_ratio = NEW_FS/fs
    ecg_resamp = signal.resample(ecg,int(ecg.shape[0]*fs_ratio), axis=0)
    ecg_pad = tf.keras.utils.pad_sequences(
        np.moveaxis(ecg_resamp,0,-1),
        maxlen=1000,
        dtype='int32',
        padding='post',
        truncating='post',
        value=0.0
    )
    ecg_pad = np.moveaxis(ecg_pad,0,-1)

    # Get the model outputs.
    probability_output = model.predict(np.expand_dims(ecg_pad,0))
    binary_output = (probability_output > 0.5).astype(int)
    

    return binary_output, probability_output

################################################################################
#
# Optional functions. You can change or remove these functions and/or add new functions.
#
################################################################################

# Extract your features.
def extract_features(record):
    header = load_header(record)
    age = get_age(header)
    sex = get_sex(header)
    
    one_hot_encoding_sex = np.zeros(3, dtype=bool)
    if sex == 'Female':
        one_hot_encoding_sex[0] = 1
    elif sex == 'Male':
        one_hot_encoding_sex[1] = 1
    else:
        one_hot_encoding_sex[2] = 1

    signal, fields = load_signals(record)

    # TO-DO: Update to compute per-lead features. Check lead order and update and use functions for reordering leads as needed.

    num_finite_samples = np.size(np.isfinite(signal))
    if num_finite_samples > 0:
        signal_mean = np.nanmean(signal)
    else:
        signal_mean = 0.0
    if num_finite_samples > 1:
        signal_std = np.nanstd(signal)
    else:
        signal_std = 0.0

    features = np.concatenate(([age], one_hot_encoding_sex, [signal_mean, signal_std]))

    return np.asarray(features, dtype=np.float32)

# Save your trained model.
def save_model(model_folder, model):
    #d = {'model': model}
    #filename = os.path.join(model_folder, 'model.sav')
    #joblib.dump(d, filename, protocol=0)
    filename = os.path.join(model_folder, 'model.keras.h5')
    model.save(filename)


def _inception_module(input_tensor, stride=1, activation='linear', use_bottleneck=True, kernel_size=40, bottleneck_size=36, nb_filters=36):

    if use_bottleneck and int(input_tensor.shape[-1]) > 1:
        input_inception = tf.keras.layers.Conv1D(filters=bottleneck_size, kernel_size=1, 
                                              padding='same', activation=activation, use_bias=False)(input_tensor)
    else:
        input_inception = input_tensor

    # kernel_size_s = [3, 5, 8, 11, 17]
    kernel_size_s = [kernel_size // (2 ** i) for i in range(3)]

    conv_list = []

    for i in range(len(kernel_size_s)):
        conv_list.append(tf.keras.layers.Conv1D(filters=nb_filters, kernel_size=kernel_size_s[i],  
                                              strides=stride, padding='same', activation=activation, use_bias=False)(
            input_inception))

    max_pool_1 = tf.keras.layers.MaxPool1D(pool_size=3, strides=stride, padding='same')(input_tensor)

    conv_6 = tf.keras.layers.Conv1D(filters=nb_filters, kernel_size=1, 
                                  padding='same', activation=activation, use_bias=False)(max_pool_1)

    conv_list.append(conv_6)

    x = tf.keras.layers.Concatenate(axis=2)(conv_list)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation(activation='relu')(x)
    return x

def _shortcut_layer(input_tensor, out_tensor):
    shortcut_y = tf.keras.layers.Conv1D(filters=int(out_tensor.shape[-1]), kernel_size=1, 
                                      padding='same', use_bias=False)(input_tensor)
    shortcut_y = tf.keras.layers.BatchNormalization()(shortcut_y)

    x = tf.keras.layers.Add()([shortcut_y, out_tensor])
    x = tf.keras.layers.Activation('relu')(x)
    return x

def build_model(input_shape, nb_classes, depth=6, use_residual=True, lr_init = 0.001, kernel_size=40, bottleneck_size=36, nb_filters=36, clf="binary", loss=tf.keras.losses.BinaryFocalCrossentropy()):
    input_layer = tf.keras.layers.Input(input_shape)

    x = input_layer
    input_res = input_layer

    for d in range(depth):

        x = _inception_module(x,kernel_size = kernel_size, bottleneck_size=bottleneck_size, nb_filters=nb_filters)

        if use_residual and d % 3 == 2:
            x = _shortcut_layer(input_res, x)
            input_res = x

    gap_layer = tf.keras.layers.GlobalAveragePooling1D()(x)

    output_layer = tf.keras.layers.Dense(units=nb_classes,activation='sigmoid')(gap_layer)  
    model = tf.keras.models.Model(inputs=input_layer, outputs=output_layer)
    model.compile(loss=loss, optimizer=tf.keras.optimizers.Adam(learning_rate=lr_init), 
                  metrics=[tf.keras.metrics.BinaryAccuracy(),
                           tf.keras.metrics.AUC(
                        num_thresholds=200,
                        curve='ROC',
                        summation_method='interpolation',
                        name="ROC",
                        multi_label=True,
                        ),
                       tf.keras.metrics.AUC(
                        num_thresholds=200,
                        curve='PR',
                        summation_method='interpolation',
                        name="PRC",
                        multi_label=True,
                        )
              ])
    return model
