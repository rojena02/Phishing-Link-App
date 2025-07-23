import numpy as np
from keras import Sequential
from keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.keras.optimizers import Adam
from keras.regularizers import l2
from keras.layers import Embedding, SpatialDropout1D, Conv1D, MaxPooling1D, LSTM, Dense, BatchNormalization,Dropout, Bidirectional
from sklearn.model_selection import train_test_split

NB_LSTM_CELLS = 256
NB_DENSE_CELLS = 256
EMBEDDING_SIZE = 100

early_stopping = EarlyStopping(
    monitor='val_loss',      # Metric to monitor
    patience=5,              # Number of epochs to wait for improvement
    restore_best_weights=True,  # Restore weights from best epoch
    mode='min',              # We want to minimize val_loss
    verbose=1
) 

def make_cnn_lstm_model(num_input_tokens, max_len):
    model = Sequential()
    
    # Embedding layer optimized for URL characters
    model.add(Embedding(
        input_dim=num_input_tokens, 
        input_length=max_len, 
        output_dim=64,  # Smaller for character-level URL tokens
        mask_zero=True,  # Handle variable URL lengths
        embeddings_initializer='glorot_uniform'  # Better than uniform
    ))
    
    # Lighter spatial dropout for URL patterns
    model.add(SpatialDropout1D(0.1))  # Reduced from 0.3
    
    # Simplified CNN layers - URLs don't need very deep feature extraction
    # First conv block - capture local patterns like 'paypal', 'secure', etc.
    model.add(Conv1D(filters=64, kernel_size=3, padding='same', activation='relu'))  # Reduced filters
    model.add(BatchNormalization())
    model.add(MaxPooling1D(pool_size=2))
    model.add(Dropout(0.1))  # Reduced from 0.2
    
    # Second conv block with different kernel size
    model.add(Conv1D(filters=128, kernel_size=5, padding='same', activation='relu'))  # Reduced filters
    model.add(BatchNormalization())
    model.add(MaxPooling1D(pool_size=2))
    model.add(Dropout(0.1))  # Reduced from 0.3
    
    # Removed third conv block - too deep for URLs
    
    # Single Bidirectional LSTM - URLs don't need very deep sequence modeling
    model.add(Bidirectional(LSTM(
        64,  # Reduced from 256
        return_sequences=False,  # Changed to False since we removed second LSTM
        dropout=0.1,  # Added LSTM dropout
        recurrent_dropout=0.1,
        kernel_regularizer=l2(1e-4)
    )))
    model.add(Dropout(0.2))  # Reduced from 0.4
    
    # Simplified dense layers
    model.add(Dense(32, activation='relu', kernel_regularizer=l2(1e-4)))  # Much smaller
    model.add(BatchNormalization())
    model.add(Dropout(0.2))  # Reduced from 0.5
    
    # Output layer for binary classification
    model.add(Dense(units=1, activation='sigmoid'))  # Binary classification
    
    # Optimizer with gradient clipping for stability
    optimizer = Adam(
        learning_rate=0.001, 
        beta_1=0.9, 
        beta_2=0.999,
        clipnorm=1.0  # Added gradient clipping
    )
    
    model.compile(
        optimizer=optimizer, 
        loss='binary_crossentropy',  # Binary classification for phishing detection
        metrics=['accuracy', 'precision', 'recall']
    )
    
    return model

class CnnLstmPredictor(object):
    model_name = 'cnn-lstm'

    def __init__(self):
        self.model = None
        self.num_input_tokens = None
        self.idx2char = None
        self.char2idx = None
        self.max_url_seq_length = None

    @staticmethod
    def get_config_file_path(model_dir_path):
        return model_dir_path + '/' + CnnLstmPredictor.model_name + '-config.npy'

    @staticmethod
    def get_weight_file_path(model_dir_path):
        return model_dir_path + '/' + CnnLstmPredictor.model_name + '.weights.h5'

    @staticmethod
    def get_architecture_file_path(model_dir_path):
        return model_dir_path + '/' + CnnLstmPredictor.model_name + '-architecture.json'

    def load_model(self, model_dir_path):
        config_file_path = self.get_config_file_path(model_dir_path)
        weight_file_path = self.get_weight_file_path(model_dir_path)

        config = np.load(config_file_path, allow_pickle=True).item()
        self.num_input_tokens = config['num_input_tokens']
        self.max_url_seq_length = config['max_url_seq_length']
        self.idx2char = config['idx2char']
        self.char2idx = config['char2idx']

        self.model = make_cnn_lstm_model(self.num_input_tokens, self.max_url_seq_length)
        self.model.build(input_shape=(None, self.max_url_seq_length))

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

        self.model = make_cnn_lstm_model(self.num_input_tokens, self.max_url_seq_length)

        with open(self.get_architecture_file_path(model_dir_path), 'wt') as f:
            f.write(self.model.to_json())

        history = self.model.fit(Xtrain, Ytrain, batch_size=batch_size, epochs=epochs, verbose=1,
                                 validation_data=(Xtest, Ytest), callbacks=[checkpoint, early_stopping])

        self.model.save_weights(weight_file_path)

        np.save(model_dir_path + '/' + CnnLstmPredictor.model_name + '-history.npy', history.history)

        return history


