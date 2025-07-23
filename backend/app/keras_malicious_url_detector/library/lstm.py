import numpy as np
from keras import Sequential
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.layers import LSTM, Dense, Dropout, Embedding, Bidirectional, BatchNormalization
from sklearn.model_selection import train_test_split
from keras.optimizers import Adam

from keras.regularizers import l2


early_stopping = EarlyStopping(
    monitor='val_loss',      # Metric to monitor
    patience=5,              # Number of epochs to wait for improvement
    restore_best_weights=True,  # Restore weights from best epoch
    mode='min',              # We want to minimize val_loss
    verbose=1
) 

def make_lstm_model(num_input_tokens, max_len=None, embedding_dim=64):
    model = Sequential()
    
    # Add embedding layer for proper input handling
    model.add(Embedding(
        input_dim=num_input_tokens,
        output_dim=embedding_dim,
        input_length=max_len,
        mask_zero=True,  # Handle variable URL lengths
        embeddings_initializer='glorot_uniform',
        name='embedding'
    ))
    
    # First LSTM layer - reduced complexity for URLs
    model.add(Bidirectional(
        LSTM(64,  # Reduced from 256
             return_sequences=True,  # Keep for stacking
             dropout=0.1,  # Reduced from 0.3
             recurrent_dropout=0.1,  # Reduced from 0.3
             kernel_regularizer=l2(1e-4)),
        name='bidirectional_lstm_1'
    ))
    
    # Second LSTM layer
    model.add(Bidirectional(
        LSTM(32,  # Reduced from 128
             return_sequences=False,
             dropout=0.1,  # Reduced from 0.3
             recurrent_dropout=0.1,  # Reduced from 0.3
             kernel_regularizer=l2(1e-4)),
        name='bidirectional_lstm_2'
    ))
    
    # Simplified dense layers for URL classification
    model.add(Dense(32, activation='relu', kernel_regularizer=l2(1e-4), name='dense_1'))  # Much smaller
    model.add(BatchNormalization())
    model.add(Dropout(0.2))  # Reduced from 0.5
    
    # Removed extra dense layers - URLs don't need deep processing
    
    # Output layer for binary classification
    model.add(Dense(1, activation='sigmoid', name='output'))  # Binary classification
    
    # Optimizer with gradient clipping for LSTM stability
    optimizer = Adam(
        learning_rate=0.001,
        beta_1=0.9,
        beta_2=0.999,
        epsilon=1e-07,
        clipnorm=1.0  # Added gradient clipping
    )
    
    model.compile(
        optimizer=optimizer,
        loss='binary_crossentropy',  # Binary classification for phishing detection
        metrics=['accuracy', 'precision', 'recall']
    )
    
    return model

class LstmPredictor(object):

    model_name = 'lstm'

    def __init__(self):
        self.model = None
        self.num_input_tokens = None
        self.idx2char = None
        self.char2idx = None
        self.max_url_seq_length = None

    @staticmethod
    def get_config_file_path(model_dir_path):
        return model_dir_path + '/' + LstmPredictor.model_name + '-config.npy'

    @staticmethod
    def get_weight_file_path(model_dir_path):
        return model_dir_path + '/' + LstmPredictor.model_name + '.weights.h5'

    @staticmethod
    def get_architecture_file_path(model_dir_path):
        return model_dir_path + '/' + LstmPredictor.model_name + '-architecture.json'

    def load_model(self, model_dir_path):
        config_file_path = self.get_config_file_path(model_dir_path)
        weight_file_path = self.get_weight_file_path(model_dir_path)

        config = np.load(config_file_path, allow_pickle=True).item()
        self.num_input_tokens = config['num_input_tokens']
        self.max_url_seq_length = config['max_url_seq_length']
        self.idx2char = config['idx2char']
        self.char2idx = config['char2idx']

        self.model = make_lstm_model(self.num_input_tokens)
        dummy_input = np.zeros((1, 256), dtype=np.int32)
        self.model(dummy_input)  # This builds the model
        self.model.load_weights(weight_file_path)

    
    def predict(self, url):
        data_size = 1
        X = np.zeros(shape=(data_size, self.max_url_seq_length))
        for idx, c in enumerate(url):
            if c in self.char2idx:
                X[0, idx] = self.char2idx[c]
        predicted = self.model.predict(X)[0][0]
        predicted_label = (predicted >= 0.5).astype(int)

        return predicted_label, predicted
    
    

    def extract_training_data(self, url_data):
        data_size = url_data.shape[0]
        X = np.zeros(shape=(data_size, self.max_url_seq_length), dtype=np.int32)
        Y = url_data['label'].values

        if not np.issubdtype(Y.dtype, np.number):
            Y = Y.astype(np.int32)

        for i in range(data_size):
            url = url_data['text'][i]
            for idx, c in enumerate(url):
                if idx < self.max_url_seq_length:
                    X[i, idx] = self.char2idx.get(c, self.char2idx.get('<UNK>', 0))
                else:
                    break
        return X, Y

    def fit(self, text_model, url_data, model_dir_path, batch_size=None, epochs=None,
            test_size=None, random_state=None):
        if batch_size is None:
            batch_size = 64
        if epochs is None:
            epochs = 50
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

        checkpoint = ModelCheckpoint(weight_file_path, save_weights_only=True)

        X, Y = self.extract_training_data(url_data)

        Xtrain, Xtest, Ytrain, Ytest = train_test_split(X, Y, test_size=test_size, random_state=random_state)
        self.model = make_lstm_model(self.num_input_tokens)

        with open(self.get_architecture_file_path(model_dir_path), 'wt') as f:
            f.write(self.model.to_json())

        history = self.model.fit(Xtrain, Ytrain, batch_size=batch_size, epochs=epochs, verbose=1,
                                 validation_data=(Xtest, Ytest), callbacks=[checkpoint, early_stopping])

        self.model.save_weights(weight_file_path)

        np.save(model_dir_path + '/' + LstmPredictor.model_name + '-history.npy', history.history)

        return history
