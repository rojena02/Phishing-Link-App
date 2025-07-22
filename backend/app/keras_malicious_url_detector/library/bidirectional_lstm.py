import numpy as np
from keras.models import Sequential
from keras.layers import (Embedding, SpatialDropout1D, LSTM, Bidirectional,
                          Dense, BatchNormalization, Dropout, Masking, LeakyReLU)
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint
from sklearn.model_selection import train_test_split

from sklearn.utils.class_weight import compute_class_weight


NB_LSTM_CELLS = 256
NB_DENSE_CELLS = 256
EMBEDDING_SIZE = 281

def make_bidirectional_lstm_model(num_input_tokens, embedding_dim=128, lstm_units=256, output_dim=2):
    model = Sequential([
        Masking(mask_value=0.0),  # Handles padded zeros if present

        Embedding(
            input_dim=num_input_tokens,
            output_dim=embedding_dim,
            embeddings_initializer='uniform',
            name='embedding'
        ),

        SpatialDropout1D(0.3),

        Bidirectional(LSTM(
            units=lstm_units,
            return_sequences=True,
            dropout=0.4,
            kernel_regularizer='l2')
        ),

        Bidirectional(LSTM(
            units=lstm_units // 2,
            return_sequences=False,
            dropout=0.4,
            kernel_regularizer='l2')
        ),

        Dense(512),
        LeakyReLU(alpha=0.1),
        BatchNormalization(),
        Dropout(0.5),

        Dense(256),
        LeakyReLU(alpha=0.1),
        BatchNormalization(),
        Dropout(0.4),

        Dense(128),
        LeakyReLU(alpha=0.1),
        Dropout(0.3),

        Dense(output_dim, activation='softmax', name='output')
    ])

    model.compile(
        optimizer=Adam(learning_rate=1e-3),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    return model

class BidirectionalLstmEmbedPredictor(object):
    model_name = 'bidirectional-lstm'

    def __init__(self):
        self.model = None
        self.num_input_tokens = None
        self.idx2char = None
        self.char2idx = None
        self.max_url_seq_length = None

    @staticmethod
    def get_config_file_path(model_dir_path):
        return model_dir_path + '/' + BidirectionalLstmEmbedPredictor.model_name + '-config.npy'

    @staticmethod
    def get_weight_file_path(model_dir_path):
        return model_dir_path + '/' + BidirectionalLstmEmbedPredictor.model_name + '.weights.h5'

    @staticmethod
    def get_architecture_file_path(model_dir_path):
        return model_dir_path + '/' + BidirectionalLstmEmbedPredictor.model_name + '-architecture.json'

    def load_model(self, model_dir_path):
        config = np.load(self.get_config_file_path(model_dir_path), allow_pickle=True).item()
        self.num_input_tokens = config['num_input_tokens']
        self.max_url_seq_length = config['max_url_seq_length']
        self.idx2char = config['idx2char']
        self.char2idx = config['char2idx']

        self.model = make_bidirectional_lstm_model(
            num_input_tokens=self.num_input_tokens,
            embedding_dim=EMBEDDING_SIZE,
            lstm_units=NB_LSTM_CELLS,
            output_dim=2
        )

        dummy_input = np.zeros((1, self.max_url_seq_length))
        self.model(dummy_input)
        self.model.load_weights(self.get_weight_file_path(model_dir_path))
        return self

    def predict(self, url):
        X = np.zeros(shape=(1, self.max_url_seq_length))
        for idx, c in enumerate(url[:self.max_url_seq_length]):
            if c in self.char2idx:
                X[0, idx] = self.char2idx[c]
        predicted = self.model.predict(X, verbose=0)[0]
        return np.argmax(predicted), predicted

    def extract_training_data(self, url_data):
        data_size = url_data.shape[0]
        X = np.zeros((data_size, self.max_url_seq_length))
        Y = np.zeros((data_size, 2))
        for i in range(data_size):
            url = url_data['text'][i]
            label = url_data['label'][i]
            for idx, c in enumerate(url[:self.max_url_seq_length]):
                if c in self.char2idx:
                    X[i, idx] = self.char2idx[c]
            Y[i, label] = 1
        return X, Y

    def fit(self, text_model, url_data, model_dir_path, batch_size=64, epochs=30,
            test_size=0.2, random_state=42):

        self.num_input_tokens = text_model['num_input_tokens']
        self.char2idx = text_model['char2idx']
        self.idx2char = text_model['idx2char']
        self.max_url_seq_length = text_model['max_url_seq_length']

        np.save(self.get_config_file_path(model_dir_path), text_model)

        checkpoint = ModelCheckpoint(
            filepath=self.get_weight_file_path(model_dir_path),
            save_weights_only=True,
            save_best_only=True,
            monitor='val_loss',
            mode='min',
            verbose=1
        )

        X, Y = self.extract_training_data(url_data)
        Xtrain, Xtest, Ytrain, Ytest = train_test_split(X, Y, test_size=test_size, random_state=random_state)

        self.model = make_bidirectional_lstm_model(
            num_input_tokens=self.num_input_tokens,
            embedding_dim=EMBEDDING_SIZE,
            lstm_units=NB_LSTM_CELLS,
            output_dim=2
        )

        with open(self.get_architecture_file_path(model_dir_path), 'wt') as f:
            f.write(self.model.to_json())

        class_weights = compute_class_weight(
            class_weight='balanced',
            classes=np.unique(Ytrain),
            y=Ytrain
        )

        history = self.model.fit(
            Xtrain, Ytrain,
            batch_size=batch_size,
            epochs=epochs,
            validation_data=(Xtest, Ytest),
            callbacks=[checkpoint],
            class_weights = class_weights,
            verbose=1
        )

        self.model.save_weights(self.get_weight_file_path(model_dir_path))
        np.save(model_dir_path + '/' + self.model_name + '-history.npy', history.history)

        return history
