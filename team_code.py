#!/usr/bin/env python

# Edit this script to add your team's code. Some functions are *required*, but you can edit most parts of the required functions,
# change or remove non-required functions, and add your own functions.

################################################################################
#
# Optional libraries, functions, and variables. You can change or remove them.
#
################################################################################

import time
import joblib
import numpy as np
import pandas as pd
import os
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
import sys
from scipy import signal
import wfdb
import tensorflow as tf
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from tensorflow.keras import layers, Model
from tensorflow.keras.layers import Activation
from tensorflow.keras.callbacks import ModelCheckpoint
import tensorflow as tf


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
    PRETRAIN = True
    FINETUNE = False
    NEW_FS = 250
    TIME = 7  # seconds
    AUXILIARY = False
    selected_leads = [0, 1, 2] + list(range(-6, 0))  # [-6, -5, -4, -3, -2, -1]
    SOURCE = "# Source:"
    os.makedirs(model_folder, exist_ok=True)
    # Find the data files.
    if verbose:
        print('Finding the Challenge data...')

    records = find_records(data_folder)
    num_records = len(records)
    records = sorted(records)

    pretrain_auxillary_labels = pd.read_csv(os.path.join("./", 'exams.csv'),dtype={'exam_id': str})


    if num_records == 0:
        raise FileNotFoundError('No data were provided.')

    # Extract the features and labels from the data.
    if verbose:
        print('Extracting features and labels from the data...')

    #features = np.zeros((num_records, 6), dtype=np.float64)
    #X_data = np.zeros((num_records, 1000, 12), dtype=np.float64)
    labels = np.zeros(num_records, dtype=bool)
    source_list = []
    record_list = [] 

    # Iterate over the records.
    for i in tqdm(range(num_records)):
        if verbose:
            width = len(str(num_records))
            print(f'- {i+1:>{width}}/{num_records}: {records[i]}...')
        
        record = os.path.join(data_folder, records[i])
        record_list.append(record)
        labels[i] = load_label(record)
        source_list.append(get_source(record))

    
    record_list = np.asarray(record_list)
    source_list = np.asarray(source_list)
    record_list_stripped = [os.path.basename(record) for record in record_list]
    #record_list_stripped = [int(s) for s in record_list_stripped]
    #print("record_list_stripped:", record_list_stripped)
    #time.sleep(50)
    #records = [int(os.path.basename(s)) for s in records]
    #
    #records = [str(s) for s in records]

    #indices_pretrain = np.where(source_list == 'CODE-15%')[0]
    indices_pretrain = np.where((source_list == 'CODE-15%')|(source_list == 'SaMi-Trop')|(source_list == 'Athlete')|(source_list == 'PTB-XL'))[0]
    indices_finetune = np.where((source_list == 'SaMi-Trop')|(source_list == 'PTB-XL')|(source_list == 'Athlete'))[0]
   
    # Train the models.
    if verbose:
        print('Training the model on the data...')
    
    if verbose:
        print('Pre-training the model...')

    temp_model = "./tempmodel/"
    temp_model_name = "temp_pretrain_model.weights.h5"
    os.makedirs(temp_model, exist_ok=True)

    model_checkp = tf.keras.callbacks.ModelCheckpoint(
        temp_model + temp_model_name,
        monitor="val_PRC",
        verbose=1,
        save_best_only=True,
        save_weights_only=True,
        mode="max",
        save_freq="epoch",
    )
    record_list_stripped = np.asarray(record_list_stripped)

    record_list_stripped_pretrain = record_list_stripped[indices_pretrain]
    record_list_stripped_pretrain = [int(s) for s in record_list_stripped_pretrain]
    
    if AUXILIARY == True:
        pretrain_auxillary_labels["exam_id"] = pretrain_auxillary_labels["exam_id"].astype(int)
        pretrain_auxillary_labels = pretrain_auxillary_labels[pretrain_auxillary_labels["exam_id"].isin(record_list_stripped_pretrain)]

        pretrain_auxillary_labels['exam_id'] = pd.Categorical(pretrain_auxillary_labels['exam_id'], categories=record_list_stripped_pretrain, ordered=True)
        pretrain_auxillary_labels = pretrain_auxillary_labels.sort_values('exam_id')

        if not (pretrain_auxillary_labels["exam_id"].values == record_list_stripped_pretrain).all():
            raise ValueError("Mismatch between pretrain_auxillary_labels and record_list_stripped_pretrain.")


        pretrain_auxillary_labels = pretrain_auxillary_labels[["1dAVb","RBBB","LBBB","SB","ST","AF"]].values
        pretrain_auxillary_labels = pretrain_auxillary_labels.astype(int)

    

    EPOCHS = 30
    BATCH_SIZE = 32
    record_list_pretrain = record_list[indices_pretrain]
    labels_pretrain = labels[indices_pretrain]

    print("labels_pretrain.shape:", labels_pretrain.shape)
    if AUXILIARY == True:
        labels_pretrain = np.hstack((pretrain_auxillary_labels, labels_pretrain.reshape(-1, 1)))
    else:
        labels_pretrain = np.expand_dims(labels_pretrain, axis=1)
    print(labels_pretrain.shape)


    X_train, X_val, y_train, y_val = train_test_split(record_list_pretrain,labels_pretrain, test_size=0.20, random_state=42)
    #model = build_model((1000,12), labels_pretrain.shape[1])
    model = build_inception_next(input_shape=(int(NEW_FS*TIME), len(selected_leads)), num_classes=labels_pretrain.shape[1])
    model.compile(loss=tf.keras.losses.BinaryCrossentropy(), optimizer=tf.keras.optimizers.AdamW(learning_rate=1e-3), 
              metrics=[#tf.keras.metrics.BinaryAccuracy(),
                       tf.keras.metrics.AUC(
                    num_thresholds=200,
                    curve='ROC',
                    summation_method='interpolation',
                    name="ROC",
                    multi_label=False,
                    ),
                   tf.keras.metrics.AUC(
                    num_thresholds=200,
                    curve='PR',
                    summation_method='interpolation',
                    name="PRC",
                    multi_label=False,
                    )
          ])
    if PRETRAIN == True:
        history = model.fit(balanced_batch_generator(BATCH_SIZE,generate_X(X_train, new_fs=NEW_FS, leads=selected_leads, time=TIME), generate_y(y_train), len(selected_leads), labels_pretrain.shape[1], y_train, time=TIME, new_fs=NEW_FS),
                            steps_per_epoch=(len(X_train)//BATCH_SIZE),
                            validation_data= batch_generator(BATCH_SIZE,generate_X(X_val, new_fs=NEW_FS, leads=selected_leads, time=TIME), generate_y(y_val), len(selected_leads), labels_pretrain.shape[1], time=TIME, new_fs=NEW_FS),
                            validation_steps=(len(X_val)//BATCH_SIZE), validation_freq=1,
                            epochs=EPOCHS, verbose=1,
                            callbacks=[model_checkp]
                            )
        model.load_weights(temp_model + temp_model_name)

    if verbose:
        print('Finetuning the model...')

    temp_model = "./tempmodel/"
    temp_model_name = "temp_finetune_model.weights.h5"
    os.makedirs(temp_model, exist_ok=True)

    model_checkp = tf.keras.callbacks.ModelCheckpoint(
        temp_model + temp_model_name,
        monitor="val_PRC",
        verbose=1,
        save_best_only=True,
        save_weights_only=True,
        mode="max",
        save_freq="epoch",
    )

    EPOCHS = 30
    BATCH_SIZE = 32
    record_list_finetune = record_list[indices_finetune]
    labels_finetune = labels[indices_finetune]
    X_train, X_val, y_train, y_val = train_test_split(record_list_finetune,labels_finetune,test_size=0.20, random_state=42)

    if FINETUNE == True:
        x = model.layers[-2].output  # Get output of second-last layer
        new_output = tf.keras.layers.Dense(1, activation='sigmoid')(x)

        # Create new model
        new_model = Model(inputs=model.input, outputs=new_output)
        
        for layer in new_model.layers[:-1]:  # Freeze all except last 
            layer.trainable = False

        for layer in new_model.layers[-1:]:  # Ensure last are trainable
            layer.trainable = True
        
        new_model.compile(loss=tf.keras.losses.BinaryCrossentropy(), optimizer=tf.keras.optimizers.AdamW(learning_rate=1e-3), 
                metrics=[tf.keras.metrics.BinaryAccuracy(),
                        tf.keras.metrics.AUC(
                        num_thresholds=200,
                        curve='ROC',
                        summation_method='interpolation',
                        name="ROC",
                        multi_label=False,
                        ),
                    tf.keras.metrics.AUC(
                        num_thresholds=200,
                        curve='PR',
                        summation_method='interpolation',
                        name="PRC",
                        multi_label=False,
                        )
            ])
        history = new_model.fit(balanced_batch_generator(BATCH_SIZE,generate_X(X_train, new_fs=NEW_FS, leads=selected_leads, time=TIME), generate_y(y_train), len(selected_leads), 1, y_train, time=TIME,new_fs=NEW_FS),
                            steps_per_epoch=(len(X_train)//BATCH_SIZE),
                            validation_data= batch_generator(BATCH_SIZE,generate_X(X_val, new_fs=NEW_FS, leads=selected_leads, time=TIME), generate_y(y_val), len(selected_leads), 1, time=TIME, new_fs=NEW_FS),
                            validation_steps=(len(X_val)//BATCH_SIZE), validation_freq=1,
                            epochs=EPOCHS, verbose=1,
                            callbacks=[model_checkp]
                            )
        model = new_model
    
    model.load_weights(temp_model + temp_model_name)


    os.makedirs(model_folder, exist_ok=True)

    # Save the model.
    save_model(model_folder, model)

    if verbose:
        print('Done.')
        print()

# Load your trained models. This function is *required*. You should edit this function to add your code, but do *not* change the
# arguments of this function. If you do not train one of the models, then you can return None for the model.
def load_model(model_folder, verbose):
    model_filename = os.path.join(model_folder, 'model.keras')
    model = tf.keras.models.load_model(model_filename)
    return model

# Run your trained model. This function is *required*. You should edit this function to add your code, but do *not* change the
# arguments of this function.
def run_model(record, model, verbose):
    NEW_FS = 250
    TIME = 7  # seconds
    selected_leads = [0, 1, 2] + list(range(-6, 0))  # [-6, -5, -4, -3, -2, -1]

    # Load the model.
    
    # Extract the features.
    ecg, text= load_signals(record)
    #text = load_text(record)
    fs = int(text["fs"])
    fs_ratio = NEW_FS/fs
    ecg_resamp = signal.resample(ecg,int(ecg.shape[0]*fs_ratio), axis=0)
    ecg_pad = tf.keras.utils.pad_sequences(
        np.moveaxis(ecg_resamp,0,-1),
        maxlen=(int(NEW_FS*TIME)),  # Assuming 7 seconds of data
        dtype='float32',
        padding='post',
        truncating='post',
        value=0.0
    )
    ecg_pad = np.moveaxis(ecg_pad,0,-1)
    # Select only the specified leads
    ecg_pad = ecg_pad[:, selected_leads]

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
    filename = os.path.join(model_folder, 'model.keras')
    model.save(filename)


def se_block(input_tensor, reduction=16):
    """Squeeze-and-Excitation block."""
    filters = input_tensor.shape[-1]
    se = layers.GlobalAveragePooling1D()(input_tensor)
    se = layers.Reshape((1, filters))(se)
    se = layers.Dense(filters // reduction, activation='gelu', use_bias=False)(se)
    se = layers.Dense(filters, activation='sigmoid', use_bias=False)(se)
    return layers.multiply([input_tensor, se])

def temporal_attention(x):
    q = layers.Conv1D(x.shape[-1], 1, padding='same')(x)
    k = layers.Conv1D(x.shape[-1], 1, padding='same')(x)
    q = layers.BatchNormalization()(q)
    k = layers.BatchNormalization()(k)
    attn = tf.keras.layers.Attention(use_scale=True)([q, k])  # Learnable scaling
    return layers.multiply([x, attn])

def inception_module(input_tensor, filters, kernel_size=40, stride=1):
    """
    Creates an inception-style module with multiple convolution paths and max pooling.
    
    Args:
        input_tensor: Input tensor to the module
        nb_filters: Number of filters for convolution layers
        kernel_size: Base kernel size for convolutions (default: 17)
        stride: Stride for convolutions and pooling (default: 1)
        activation: Activation function to use (default: 'relu')
        
    Returns:
        Concatenated output tensor from all paths
    """
    # Generate kernel sizes by dividing base kernel_size by powers of 2
    kernel_size_s = [kernel_size // (2 ** i) for i in range(4)]

    new_filters = filters // len(kernel_size_s)
    
    # Initialize list to store convolution outputs
    conv_list = []
    
    # Create parallel convolution paths with different kernel sizes
    for i in range(len(kernel_size_s)):
        conv = tf.keras.layers.Conv1D(
            filters=new_filters,
            kernel_size=kernel_size_s[i],
            strides=stride,
            padding='same',
            use_bias=False
        )(input_tensor)
        conv_list.append(conv)
    

    # Concatenate all paths along the channel axis
    return tf.keras.layers.Concatenate(axis=2)(conv_list)


def residual_block(x, filters, stride=1):
    """Basic residual block with two 3x1 conv layers and identity shortcut."""
    shortcut = x
    
    # Check if we need to adjust the shortcut connection
    if stride != 1 or x.shape[-1] != filters:
        shortcut = layers.Conv1D(filters, 1, strides=stride, padding='same')(shortcut)
        shortcut = layers.BatchNormalization()(shortcut)
    
    # First convolution
    x = layers.Conv1D(filters, 7, strides=stride, padding='same')(x)
    x = inception_module(x, filters=filters)
    x = layers.BatchNormalization()(x)
    x = Activation('gelu')(x)

    
    # Second convolution
    x = layers.Conv1D(filters, 3, strides=1, padding='same')(x)
    x = layers.BatchNormalization()(x)

    x = se_block(x)
    
    #x = temporal_attention(x)
    
    # Add shortcut and activate
    x = layers.Add()([x, shortcut])
    x = Activation('gelu')(x)
    #x = GELU()(x)
    
    return x


def build_inception_next(input_shape, num_classes):
    """Build xresnet101 model for 1D ECG signal classification."""
    inputs = layers.Input(shape=input_shape)
    
    # Initial convolution
    x = layers.Conv1D(64, 11, strides=2, padding='same')(inputs)
    x = layers.BatchNormalization()(x)
    x = Activation('gelu')(x)
    #x = GELU()(x)
    #x = layers.MaxPool1D(3, strides=2, padding='same')(x)

    # Residual stages
    # Stage 1 (3 blocks)
    for _ in range(3):
        x = residual_block(x, 64)
    
    # Stage 2 (4 blocks)
    x = residual_block(x, 128, stride=2)
    for _ in range(3):
        x = residual_block(x, 128)
    
    # Stage 3 (23 blocks)
    x = residual_block(x, 256, stride=2)
    for _ in range(22):
        x = residual_block(x, 256)
    
        
    # Stage 4 (3 blocks)
    x = residual_block(x, 512, stride=2)
    for _ in range(2):
        x = residual_block(x, 512)
    
    # Final layers
    x = layers.GlobalAveragePooling1D()(x)
    outputs = layers.Dense(num_classes, activation='sigmoid')(x)
    
    return Model(inputs, outputs)


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

#tf.keras.losses.BinaryFocalCrossentropy()
def build_model(input_shape, nb_classes, depth=6, use_residual=True, lr_init = 0.001, kernel_size=40, bottleneck_size=36, nb_filters=36, clf="binary", loss=tf.keras.losses.BinaryCrossentropy()):
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
    return model


def batch_generator(batch_size, gen_x, gen_y, num_leads, num_classes, time, new_fs): 
    batch_features = np.zeros((batch_size,int(new_fs*time), num_leads))
    batch_labels = np.zeros((batch_size,num_classes))
    while True:
        for i in range(batch_size):

            batch_features[i] = next(gen_x)
            batch_labels[i] = next(gen_y)
            
        yield batch_features, batch_labels



def generate_X(X_train_file, new_fs, leads, time):
    while True:
        for i in range(len(X_train_file)):
            ecg, text = load_signals(X_train_file[i])

            fs = int(text["fs"])
            fs_ratio = new_fs/fs
            ecg_resamp = signal.resample(ecg,int(ecg.shape[0]*fs_ratio), axis=0)
            ecg_pad = tf.keras.utils.pad_sequences(
                np.moveaxis(ecg_resamp,0,-1),
                maxlen=int(new_fs*time),  # Assuming 10 seconds of data
                dtype='int32',
                padding='post',
                truncating='post',
                value=0.0
            )
            ecg_pad = np.moveaxis(ecg_pad,0,-1)
            yield ecg_pad[:, leads]  # Select only the specified leads

def generate_y(y_train):
    while True:
        for i in range(len(y_train)):
            yield y_train[i]


def balanced_batch_generator(batch_size, gen_x, gen_y, num_leads, num_classes, y_data, time,new_fs):
    # Separate indices of positive and negative classes
    positive_indices = np.where(y_data == 1)[0]
    negative_indices = np.where(y_data == 0)[0]
    
    # Calculate the number of samples per class in each batch
    half_batch = batch_size // 2
    
    # Initialize arrays to hold the batch data
    batch_features = np.zeros((batch_size, int(new_fs*time), num_leads))
    batch_labels = np.zeros((batch_size, num_classes))
    
    while True:
        # Shuffle the indices to ensure randomness
        positive_indices = shuffle(positive_indices)
        negative_indices = shuffle(negative_indices)
        
        # Select half_batch samples from each class
        selected_positive_indices = positive_indices[:half_batch]
        selected_negative_indices = negative_indices[:half_batch]
        
        # Combine the selected indices
        selected_indices = np.concatenate([selected_positive_indices, selected_negative_indices])
        
        # Shuffle the combined indices to mix positive and negative samples
        selected_indices = shuffle(selected_indices)
        
        # Fill the batch with the selected samples
        for i, idx in enumerate(selected_indices):
            batch_features[i] = next(gen_x)
            batch_labels[i] = next(gen_y)
        
        yield batch_features, batch_labels

def get_source(record):
    # Load the record
    text = load_header(record)
    # Extract the source information from the text header
    source = get_variable(text, "# Source:")
    return source[0] 


