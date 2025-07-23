import numpy as np
from keras.layers import Embedding, SpatialDropout1D, LSTM, Bidirectional, Dense,BatchNormalization, Dropout
from keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from keras.regularizers import l2
from keras.models import Sequential
from keras.callbacks import ModelCheckpoint, EarlyStopping

NB_LSTM_CELLS = 256
NB_DENSE_CELLS = 256
EMBEDDING_SIZE = 281

early_stopping = EarlyStopping(
    monitor='val_loss',      # Metric to monitor
    patience=5,              # Number of epochs to wait for improvement
    restore_best_weights=True,  # Restore weights from best epoch
    mode='min',              # We want to minimize val_loss
    verbose=1
) 
def make_bidirectional_lstm_model(
        num_input_tokens, 
        max_url_seq_length=256,
        embedding_dim=64,  # Smaller for character-level URL tokens
        lstm_units=128,    # Keep your original size but could be smaller
        output_dim=1       # Binary classification: phishing or not
):
    model = Sequential([
        # Embedding layer optimized for URL characters
        Embedding(
            input_dim=num_input_tokens,
            output_dim=embedding_dim,
            input_length=max_url_seq_length,
            mask_zero=True,  # Important: handle variable URL lengths
            embeddings_initializer='glorot_uniform',  # Better than uniform
            name='embedding'
        ),
        
        # Lighter spatial dropout for URL patterns
        SpatialDropout1D(0.1),  # Reduced from 0.3
        
        # First bidirectional LSTM layer
        Bidirectional(LSTM(
            units=lstm_units,
            return_sequences=True,
            dropout=0.1,  # Reduced from 0.3
            recurrent_dropout=0.1,  # Reduced from 0.3
            kernel_regularizer=l2(1e-4)  # Lighter regularization
        ), name='bidirectional_lstm_1'),
        
        # Second bidirectional LSTM layer
        Bidirectional(LSTM(
            units=lstm_units // 2,
            return_sequences=False,
            dropout=0.1,  # Reduced from 0.3
            recurrent_dropout=0.1,  # Reduced from 0.3
            kernel_regularizer=l2(1e-4)  # Lighter regularization
        ), name='bidirectional_lstm_2'),
        
        # Simplified dense layers - URLs don't need deep processing
        Dense(64, activation='relu', name='dense_1'),  # Much smaller
        BatchNormalization(),
        Dropout(0.2),  # Reduced from 0.5
        
        Dense(32, activation='relu', name='dense_2'),  # Smaller
        Dropout(0.1),  # Reduced from 0.4
        
        # Output layer for binary classification
        Dense(output_dim, activation='sigmoid', name='output')  # sigmoid for binary
    ])
    
    # Updated optimizer and loss for binary classification
    model.compile(
        optimizer=Adam(
            learning_rate=0.001,
            beta_1=0.9,
            beta_2=0.999,
            epsilon=1e-07,
            clipnorm=1.0  # Added gradient clipping for LSTM stability
        ),
        loss='binary_crossentropy',  # Changed from categorical
        metrics=['accuracy', 'precision', 'recall']
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
        self.max_url_seq_length = int(config['max_url_seq_length'])
        self.idx2char = config['idx2char']
        self.char2idx = config['char2idx']

        # Create model
        self.model = make_bidirectional_lstm_model(
            num_input_tokens=self.num_input_tokens
        )
        # # Build the model by calling it once with dummy data
        dummy_input = np.zeros((1, 256), dtype=np.int32)
        print(f"DEBUG: dummy_input shape BEFORE model call: {dummy_input.shape}")

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

        self.model = make_bidirectional_lstm_model(self.num_input_tokens)

        with open(self.get_architecture_file_path(model_dir_path), 'wt') as f:
            f.write(self.model.to_json())

        history = self.model.fit(Xtrain, Ytrain, batch_size=batch_size, epochs=epochs, verbose=1,
                                 validation_data=(Xtest, Ytest), callbacks=[checkpoint, early_stopping])

        self.model.save_weights(weight_file_path)

        np.save(model_dir_path + '/' + BidirectionalLstmEmbedPredictor.model_name + '-history.npy', history.history)

        return history
