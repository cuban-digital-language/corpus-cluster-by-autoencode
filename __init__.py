try:
    from . import data_mining as dm
    from . import tokenizer_digital_language as tokenizer
except (ModuleNotFoundError, ImportError):
    import data_mining as dm
    import tokenizer_digital_language as tokenizer

import os
from keras.utils import pad_sequences
from keras import Input, Model, optimizers
from keras.layers import Bidirectional, LSTM, Embedding, RepeatVector, Dense
import numpy as np
from keras.preprocessing.text import Tokenizer
from sklearn.cluster import KMeans, DBSCAN
import pickle
from keras.callbacks import ModelCheckpoint
from tensorflow import train
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer, LancasterStemmer
from nltk import SnowballStemmer


def remove_stopwords(words):
    """Remove stop words from list of tokenized words"""
    new_words = []
    for word in words:
        if word not in stopwords.words('spanish'):
            new_words.append(word)
    return new_words


def stem_words(words):
    """Stem words in list of tokenized words"""
    stemmer = SnowballStemmer('spanish')
    stems = []
    for word in words:
        stem = stemmer.stem(word)
        stems.append(stem)
    return stems


def save_tokens(list_tokens):
    text = ''
    for tokens in list_tokens:
        for token in tokens:
            text += token + ' '
        text += '\n'
    with open(os.path.join(os.getcwd(), 'tokens'), 'w') as f:
        f.write(text)


def red_tokens():
    with open(os.path.join(os.getcwd(), 'tokens'), 'r') as f:
        text = f.read()
    list_tokens = []
    list_texts = text.split('\n')
    for tokens in list_texts:
        t = []
        for a in tokens.split(' '):
            if not a == '':
                t.append(a)
        list_tokens.append(t)
    return list_tokens


def get_tokens():
    try:
        tokens = red_tokens()
        # index_text = loadData('index_text', 'b')
        index_text = {}
        return tokens, index_text
    except:
        pass
    texts = dm.get_all_text('data_mining/')

    _tokenizer = tokenizer.SpacyCustomTokenizer()

    text_index = {}
    list_tokens = []
    i = 0
    print('start tokenizer')
    for text in texts:
        tokens = set(t.lemma.lower() for t in _tokenizer(text[0]) if t.lemma != None and
                     not t.is_url() and not t.is_stop and not t.is_user_tag() and not t.is_hashtag() and not t.space() and t.pos != 'PRON')
        tokens = remove_stopwords(tokens)
        # tokens = stem_words(tokens)
        # tokens = lemmatize_verbs(tokens)
        if len(tokens) > 0:
            text_index[i] = text
            list_tokens.append(tokens)
            i += 1
            # print(tokens)
    print('end tokenizer')
    save_tokens(list_tokens)
    return list_tokens, text_index


def get_words(sents: [[]]):
    words = set()
    for sent in sents:
        words = words.union(set(sent))

    return words


def MappDocToVector(model_encoder, pad_seq):
    doc_vector = {}
    vector_doc = {}

    pads_vector = model_encoder.predict(pad_seq)
    for i, v in enumerate(pads_vector):
        n_v = tuple(x for x in v)
        doc_vector[i] = n_v
        vector_doc[n_v] = i

    return doc_vector, vector_doc


def saveData(data: str, path: str, type: str = ''):
    with open(os.path.join(os.getcwd(), path), f'w{type}') as f:
        f.write(data)


def loadData(path: str, type: str = ''):
    with open(os.path.join(os.getcwd(), path), f'r{type}') as f:
        text = f.read()
        f.close()
    return text


def loadData2(path: str, type: str = ''):
    with open(os.path.join(os.getcwd(), path), f'r{type}') as f:
        return f


def model():
    list_tokens, index_text = get_tokens()  # la lista de los textos tokenizados
    print(len(list_tokens), 'number of sentences')
    sents = list_tokens
    maxlen = max([len(s) for s in sents])
    print(maxlen)
    vocab = get_words(sents)  # todas las palabras que aparecen en los textos
    num_words = len(vocab)
    print(num_words, 'num words')
    # num_words = 10000
    embed_dim = 16
    batch_size = 64
    # maxlen = 100

    tok = Tokenizer(num_words=num_words, split=' ')
    tok.fit_on_texts(sents)
    seqs = tok.texts_to_sequences(sents)
    pad_seqs = pad_sequences(seqs, maxlen)
    print(pad_seqs)
    encoder_inputs = Input(shape=(maxlen,), name='Encoder-Input')
    emb_layer = Embedding(num_words, embed_dim, input_length=maxlen, name='Body-Word-Embedding', mask_zero=False)
    x = emb_layer(encoder_inputs)
    state_h = Bidirectional(LSTM(embed_dim, activation='relu', name='Encoder-Last-LSTM'))(x)
    encoder_model = Model(inputs=encoder_inputs, outputs=state_h, name='Encoder-Model')
    seq2seq_encoder_out = encoder_model(encoder_inputs)

    decoded = RepeatVector(maxlen)(seq2seq_encoder_out)
    decoder_lstm = Bidirectional(LSTM(embed_dim, name='Decoder-LSTM-before'))
    decoder_lstm_output = decoder_lstm(decoded)
    decoder_dense = Dense(maxlen, activation='softmax', name='Final-Output-Dense-before')
    decoder_outputs = decoder_dense(decoder_lstm_output)
    # decoder_dense = Dense(maxlen, activation='softmax', name='Final-Output-Dense-before-2')
    # decoder_outputs = decoder_dense(decoder_outputs)

    seq2seq_Model = Model(encoder_inputs, decoder_outputs)
    seq2seq_Model.compile(optimizer=optimizers.Nadam(learning_rate=0.001), loss='mae')

    seq2seq_Model.summary()

    checkpoint_dir = './training'
    checkpoint_prefix = os.path.join(checkpoint_dir, 'ckpt_{epoch}')

    checkpoint_callback = ModelCheckpoint(filepath=checkpoint_prefix, save_weights_only=True)
    latest = train.latest_checkpoint(checkpoint_dir)

    if latest:
        seq2seq_Model.load_weights(latest)

    history = seq2seq_Model.fit(pad_seqs, pad_seqs,
                                callbacks=[checkpoint_callback],
                                batch_size=batch_size,
                                epochs=10)

    doc_vector, vector_doc = MappDocToVector(encoder_model, pad_seqs)

    kmeans = KMeans(n_clusters=10, random_state=0)
    X = [x for x in doc_vector.values()]
    kmeans.fit(X)
    predict = kmeans.predict(X)

    ret = []
    for i, cluster in enumerate(predict):
        ret.append((index_text[i], cluster))

    serialize_cluster = pickle.dumps(kmeans)
    serialize_encoder_model = pickle.dumps(encoder_model)
    serialize_index_text = pickle.dumps(encoder_model)

    saveData(serialize_cluster, 'cluster', 'b')
    saveData(serialize_encoder_model, 'encoder_model', 'b')
    saveData(str(maxlen), 'maxlen')
    saveData(serialize_index_text, 'index_text', 'b')
    # guardar esto en un txt
    # guardar tambien el maxlen

    return encoder_model, seq2seq_Model, ret


def Predictor(text: str):
    maxlen = int(loadData('maxlen'))
    _tokenizer = tokenizer.SpacyCustomTokenizer()
    nlp = _tokenizer.nlp

    tokens = [t.text for t in nlp(text)]
    num_words = len(set(tokens))
    sents = [tokens]

    tok = Tokenizer(num_words=num_words, split=' ')
    tok.fit_on_texts(sents)
    seqs = tok.texts_to_sequences(sents)
    pad_seqs = pad_sequences(seqs, maxlen)

    encoder_model = pickle.loads(loadData('encoder_model', 'b'))
    cluster = pickle.loads(loadData('cluster', 'b'))

    encoder_predict = encoder_model.predict(pad_seqs)

    n_v = tuple(x for x in encoder_predict[0])
    return cluster.predict([n_v])


model()

a = Predictor('yo soy una estrella')

print(a)
