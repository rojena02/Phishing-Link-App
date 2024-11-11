import numpy as np
from keras.layers import Embedding, SpatialDropout1D, LSTM, Bidirectional, Dense
from keras.optimizers import Adam
from sklearn.model_selection import train_test_split

from keras.models import Sequential
from keras.callbacks import ModelCheckpoint

NB_LSTM_CELLS = 256
NB_DENSE_CELLS = 256
EMBEDDING_SIZE = 281


def make_bidirectional_lstm_model(num_input_tokens, embedding_dim=EMBEDDING_SIZE, lstm_units=NB_LSTM_CELLS, output_dim=2):
    model = Sequential([
        # Specify input_length in Embedding layer
        Embedding(
            input_dim=num_input_tokens,
            output_dim=embedding_dim,
            input_length=None  # Allow variable length input
        ),
        SpatialDropout1D(0.2),
        Bidirectional(LSTM(
            units=lstm_units,
            dropout=0.2,
            recurrent_dropout=0.2
        )),
        Dense(output_dim, activation='softmax')
    ])
    
    # Compile model
    model.compile(
        optimizer=Adam(learning_rate=0.001),
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
        config_file_path = self.get_config_file_path(model_dir_path)
        weight_file_path = self.get_weight_file_path(model_dir_path)

        config = np.load(config_file_path, allow_pickle=True).item()
        self.num_input_tokens = config['num_input_tokens']
        self.max_url_seq_length = config['max_url_seq_length']
        self.idx2char = config['idx2char']
        self.char2idx = config['char2idx']
        embedding_size = config.get('embedding_size', 281)  # Using 281 as it matches your saved weights

        # Create model
        self.model = make_bidirectional_lstm_model(
            num_input_tokens=self.num_input_tokens,
            embedding_dim=EMBEDDING_SIZE,
            lstm_units=NB_LSTM_CELLS,
            output_dim=2
        )
        
        # Build the model by calling it once with dummy data
        dummy_input = np.zeros((1, self.max_url_seq_length))
        self.model(dummy_input)  # This builds the model
        # Now load weights
        self.model.load_weights(weight_file_path)
        return self

    def predict(self, url):
        data_size = 1
        X = np.zeros(shape=(data_size, self.max_url_seq_length))
        for idx, c in enumerate(url):
            if c in self.char2idx:
                X[0, idx] = self.char2idx[c]
        predicted = self.model.predict(X)[0]
        predicted_label = np.argmax(predicted)
        return predicted_label, predicted

    def extract_training_data(self, url_data):
        data_size = url_data.shape[0]
        X = np.zeros(shape=(data_size, self.max_url_seq_length))
        Y = np.zeros(shape=(data_size, 2))
        for i in range(data_size):
            url = url_data['text'][i]
            label = url_data['label'][i]
            for idx, c in enumerate(url):
                X[i, idx] = self.char2idx[c]
            Y[i, label] = 1

        return X, Y

    def fit(self, text_model, url_data, model_dir_path, batch_size=None, epochs=None,
            test_size=None, random_state=None):
        if batch_size is None:
            batch_size = 64
        if epochs is None:
            epochs = 30
        if test_size is None:
            test_size = 0.2
        if random_state is None:
            random_state = 42

        self.num_input_tokens = text_model['num_input_tokens']
        self.char2idx = text_model['char2idx']
        self.idx2char = text_model['idx2char']
        self.max_url_seq_length = text_model['max_url_seq_length']

        np.save(self.get_config_file_path(model_dir_path), text_model)

        weight_file_path = self.get_weight_file_path(model_dir_path)

        checkpoint = ModelCheckpoint(
            filepath = weight_file_path,
            save_weights_only=True,  # Saves only the weights, not the entire model
        )

        X, Y = self.extract_training_data(url_data)

        Xtrain, Xtest, Ytrain, Ytest = train_test_split(X, Y, test_size=test_size, random_state=random_state)

        self.model = make_bidirectional_lstm_model(self.num_input_tokens, self.max_url_seq_length)

        with open(self.get_architecture_file_path(model_dir_path), 'wt') as f:
            f.write(self.model.to_json())

        history = self.model.fit(Xtrain, Ytrain, batch_size=batch_size, epochs=epochs, verbose=1,
                                 validation_data=(Xtest, Ytest), callbacks=[checkpoint])

        self.model.save_weights(weight_file_path)

        np.save(model_dir_path + '/' + BidirectionalLstmEmbedPredictor.model_name + '-history.npy', history.history)

        return history
