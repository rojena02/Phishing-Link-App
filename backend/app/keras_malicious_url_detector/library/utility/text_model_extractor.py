def extract_text_model(urls):
    char2idx = dict()
    HARD_MAX_URL_SEQ_LENGTH = 256 

    for url in urls:
        for c in url:
            if c not in char2idx:
                char2idx[c] = len(char2idx)
    
    num_input_tokens = len(char2idx)
    idx2char = dict([(idx, c) for c, idx in char2idx.items()])

    config = dict()
    config['num_input_tokens'] = num_input_tokens
    config['char2idx'] = char2idx
    config['idx2char'] = idx2char
    
    config['max_url_seq_length'] = HARD_MAX_URL_SEQ_LENGTH

    return config