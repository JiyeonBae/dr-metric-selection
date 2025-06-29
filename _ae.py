import tensorflow as tf
from keras import backend as K
from keras.layers import Input, Dense
from keras.models import Model
from keras.callbacks import EarlyStopping
from sklearn.base import BaseEstimator, TransformerMixin
import os
import numpy as np
import random as rn
from enum import Enum

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# begin set seed
os.environ['PYTHONHASHSEED'] = '0'
np.random.seed(42)
rn.seed(42)
tf.random.set_seed(42)

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)
# end set seed

class ModelSize(Enum):
    SMALL = "SMALL"
    MEDIUM = "MEDIUM"
    LARGE = "LARGE"

class AutoencoderProjection(BaseEstimator, TransformerMixin):
    def __init__(self, model_size=ModelSize.SMALL):
        self.autoencoder = None
        self.encoder = None
        self.model_size = model_size
        self.stopper = EarlyStopping(monitor='val_loss', min_delta=0.001, patience=50, verbose=False, mode='min')

    def set_params(self, n_components, model_size):
        self.model_size = model_size

    def fit_transform(self, X, y=None):
        K.clear_session()

        if self.model_size == ModelSize.SMALL:
            ae_input = Input(shape=(X.shape[1],))
            encoded = Dense(2, activation='linear')(ae_input)
            decoded = Dense(X.shape[1], activation='sigmoid')(encoded)
        elif self.model_size == ModelSize.MEDIUM:
            ae_input = Input(shape=(X.shape[1],))
            encoded = Dense(16, activation='sigmoid')(ae_input)
            encoded = Dense(2, activation='linear')(encoded)
            decoded = Dense(16, activation='sigmoid')(encoded)
            decoded = Dense(X.shape[1], activation='sigmoid')(decoded)
        elif self.model_size == ModelSize.LARGE:
            ae_input = Input(shape=(X.shape[1],))
            encoded = Dense(128, activation='sigmoid')(ae_input)
            encoded = Dense(32, activation='sigmoid')(encoded)
            encoded = Dense(2, activation='linear')(encoded)
            decoded = Dense(32, activation='sigmoid')(encoded)
            decoded = Dense(128, activation='sigmoid')(decoded)
            decoded = Dense(X.shape[1], activation='sigmoid')(decoded)

        self.encoder = Model(inputs=ae_input, outputs=encoded)
        self.autoencoder = Model(ae_input, decoded)
        self.autoencoder.compile(optimizer='adam', loss='binary_crossentropy')

        self.autoencoder.fit(X, X, epochs=100, batch_size=32, shuffle=True,
                             validation_split=0.1, verbose=False, callbacks=[self.stopper])
        return self.encoder.predict(X)
