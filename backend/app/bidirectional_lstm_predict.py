from keras_malicious_url_detector.library.bidirectional_lstm import BidirectionalLstmEmbedPredictor
from keras_malicious_url_detector.library.utility.url_data_loader import load_url_data
import numpy as np

def main():
    data_dir_path = './data'
    model_dir_path = './models'

    predictor = BidirectionalLstmEmbedPredictor()
    predictor.load_model(model_dir_path)

    url_data = load_url_data(data_dir_path)
    count = 0
    
    unk_idx = predictor.char2idx.get('<UNK>', 0)

    for url, label in zip(url_data['text'], url_data['label']):
        if count >= 50: # Break after 50 for quick check
            break
        temp_X = np.zeros(shape=(1, predictor.max_url_seq_length), dtype=np.int32)
        tokenized_chars = []
        for idx, c in enumerate(url):
            if idx < predictor.max_url_seq_length:
                char_idx = predictor.char2idx.get(c, unk_idx) # Use the same UNK handling as in predict
                temp_X[0, idx] = char_idx
                tokenized_chars.append(char_idx)
            else:
                break
        
        predicted_label, predicted_probability = predictor.predict(url) # Using your fixed predict method
        print(f'  Predicted: {predicted_label} (Prob: {predicted_probability:.4f}) Actual: {label}')
        
        count += 1

if __name__ == '__main__':
    main()