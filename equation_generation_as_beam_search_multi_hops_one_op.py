"""Equation Generation using Memory Network Encoder and LSTM Decoder with beam search."""
from __future__ import print_function
import argparse
import numpy as np
from functools import reduce
import re
from math import log
from pickle import dump
from keras.models import Input, Model
from keras.layers.embeddings import Embedding
from keras.layers import Dense, Permute, Dropout, add, dot, concatenate, Activation, Lambda
from keras.layers import LSTM
from keras.preprocessing.sequence import pad_sequences
import keras.backend as K
from pickle import load
from itertools import chain
np.random.seed(1337)  # for reproducibility


def load_object_from_pickle(pickle_file_path):
    with open(pickle_file_path, 'rb') as file_load:
        return load(file_load)


def create_item_to_index_dictionary(list_items):
    return {item: index for index, item in enumerate(list_items)}


def create_index_to_item_dictionary(dict_items):
    return {v: k for k, v in dict_items.items()}


def tokenize(sent):
    '''Return the tokens of a sentence including punctuation.

    >>> tokenize('Bob dropped the apple. Where is the apple?')
    ['Bob', 'dropped', 'the', 'apple', '.', 'Where', 'is', 'the', 'apple', '?']
    '''
    return [x.strip() for x in re.split('(\W+)', sent) if x.strip()]


def parse_stories(lines, only_supporting=False):
    '''Parse stories provided in the bAbi tasks format

    If only_supporting is true, only the sentences that support the answer are kept.
    '''
    data = []
    story = []
    for line in lines:
        nid, line = line.lower().split(' ', 1)
        nid = int(nid)
        if nid == 1:
            story = []
        if '\t' in line:
            q, a, supporting = line.split('\t')
            q = tokenize(q)
            substory = None
            if only_supporting:
                # Only select the related substory
                supporting = map(int, supporting.split())
                substory = [story[i - 1] for i in supporting]
            else:
                # Provide all the substories
                substory = [x for x in story if x]
            data.append((substory, q, a))
            story.append('')
        else:
            sent = tokenize(line)
            story.append(sent)
    return data


def get_stories(f, only_supporting=False, max_length=None):
    '''Given a file name, read the file, retrieve the stories, and then convert the sentences into a single story.

    If max_length is supplied, any stories longer than max_length tokens will be discarded.
    '''
    data = parse_stories(f.readlines(), only_supporting=only_supporting)
    flatten = lambda data: reduce(lambda x, y: x + y, data, [])
    data = [(flatten(story), q, answer) for story, q,
            answer in data if not max_length or len(flatten(story)) < max_length]
    return data


def vectorize_stories(data, word_idx, word_idx_answer, story_maxlen, query_maxlen, word_embeddings):
    X, XEmbedding, Xq, XqEmbedding, Y, Y_1, Y_2, Y_3, Y_4 = list(), list(), list(), list(), list(), list(), list(), list(), list()
    # print("length of data")
    for story, query, answer in data:
        x = [word_idx[w] for w in story]
        xEmbedding = [word_embeddings[word_idx[w]].tolist() for w in story]
        xq = [word_idx[w] for w in query]
        xqEmbedding = [word_embeddings[word_idx[w]].tolist() for w in query]
        # let's not forget that index 0 is reserved
        y = np.zeros((4, len(word_idx_answer)))
        y_1 = np.zeros((1, len(word_idx_answer),))
        y_2 = np.zeros((1, len(word_idx_answer),))
        y_3 = np.zeros((1, len(word_idx_answer),))
        y_4 = np.zeros((1, len(word_idx_answer),))
        # y_inputs = np.zeros((4, len(word_idx_answer)))
        for index, item in enumerate(answer.split()):
            y[index][word_idx_answer[item]] = 1
            if index == 0:
                y_2[0][word_idx_answer[item]] = 1
            if index == 1:
                y_3[0][word_idx_answer[item]] = 1
            if index == 2:
                y_4[0][word_idx_answer[item]] = 1
        y_1[0][0] = 1
        y[-1][-1] = 1
        X.append(x)
        Xq.append(xq)
        XEmbedding.append(xEmbedding)
        XqEmbedding.append(xqEmbedding)
        Y.append(y)
        Y_1.append(y_1)
        Y_2.append(y_2)
        Y_3.append(y_3)
        Y_4.append(y_4)
    return pad_sequences(X, maxlen=story_maxlen, padding='post'), pad_sequences(XEmbedding, maxlen=story_maxlen, padding='post'), pad_sequences(Xq, maxlen=query_maxlen, padding='post'), pad_sequences(XqEmbedding, maxlen=query_maxlen, padding='post'), np.array(Y), np.array(Y_1), np.array(Y_2), np.array(Y_3), np.array(Y_4)


def predict_sequence(infdec, all_hidden_states, hidden_state, cell_state, cardinality, vocab_answer_dict, n_steps=3):
    # encode
    target_seq = np.array([0.0 for _ in range(cardinality)]).reshape((-1, 1, cardinality))
    target_seq[0, 0, 0] = 1
    # collect predictions
    hidden_state = hidden_state.reshape((1, 64))
    cell_state = cell_state.reshape((1, 64))
    all_hidden_states = all_hidden_states.reshape((-1, 34, 64))
    output = list()
    for t in range(n_steps):
        # predict next char
        yhat, h, c = infdec.predict([target_seq, hidden_state, cell_state, all_hidden_states], batch_size=1)
        # store prediction
        # print(yhat.shape)
        token_index = np.argmax(yhat[0, -1, :])
        output.append(vocab_answer_dict[token_index])
        # update state
        hidden_state, cell_state = h, c
        # update target sequence
        # target_seq = np.zeros((1, 1, 8))
        target_seq = np.zeros((1, 1, 9))
        target_seq[0, 0, token_index] = 1.
    return ' '.join(output)


def one_hot_decode(encoded_seq):
    return [np.argmax(vector) for vector in encoded_seq]


def predict_sequence_with_beam_search(infdec, all_hidden_states, hidden_state, cell_state, cardinality, vocab_answer_dict, embedding_size, max_length, n_steps=10, k=3):
    # encode
    # target_seq = np.array([0.0 for _ in range(cardinality)]).reshape((-1, 1, cardinality))
    # target_seq[0, 0, 0] = 1
    # collect predictions
    hidden_state = hidden_state.reshape((1, embedding_size))
    cell_state = cell_state.reshape((1, embedding_size))
    all_hidden_states = all_hidden_states.reshape((-1, max_length, embedding_size))
    output = list()
    sequences = list()
    # list of length 0
    sequences.append([list(), 1.])
    states = [[[hidden_state, cell_state]]]
    for t in range(n_steps):
        # predict next char
        allCandidates = list()
        currSequence = sequences[-1] if t > 0 else sequences
        currStates = states[-1]
        tempStates = list()
        # print('t=', t, len(currSequence))
        for i in range(len(currSequence)):
            token_index = currSequence[i][0][-1] if t > 0 else 0
            # print(currSequence[i])
            # print('token', token_index)
            # print('t', token_index)
            # update target sequence
            target_seq = np.zeros((1, 1, cardinality))
            target_seq[0, 0, token_index] = 1.
            yhat, h, c = infdec.predict([target_seq, currStates[i][0], currStates[i][1], all_hidden_states], batch_size=1)
            tempStates.append([h, c])
            # store prediction
            results = yhat[0, -1, :]
            indexes = results.shape[0]
            for index in range(indexes):
                candidate = (currSequence[i][0] + [index], currSequence[i][1] * -log(results[index]))
                allCandidates.append(candidate)
            # update state
        if t == 0:
            for i in range(k - 1):
                tempStates.append(tempStates[-1])
        states.append(tempStates)
        ordered = sorted(allCandidates, key=lambda x: x[1])
        sequences.append(ordered[: k])
        # print(sequences[-1], states[-1])
    for sequence in sequences[-1]:
        output += [' '.join((map(lambda x: vocab_answer_dict[x], sequence[0])))]
    return '\t'.join(output)


def train_model(train_file, test_file, out_file, word_idx, word_embeddings, epochs):
    # question="Jane had 4 apples. She gave 1 to Umesh. How many apples does jane hav now?"
    EMBED_HIDDEN_SIZE = 50
    SENT_HIDDEN_SIZE = 100
    QUERY_HIDDEN_SIZE = 100
    BATCH_SIZE = 32
    EPOCHS = epochs
    LSTM_OUTPUT_DIM = 64
    print('Embed / Sent / Query = {}, {}, {}'.format(EMBED_HIDDEN_SIZE, SENT_HIDDEN_SIZE, QUERY_HIDDEN_SIZE))
    train = get_stories(
        open(train_file, 'r', encoding='utf-8'))
    answer_vocab = set(
        chain(*(map(lambda x: x.split(), list(zip(*train))[2]))))
    vocab_answer_dict = {key + 1: val for key, val in dict(enumerate(answer_vocab)).items()}
    vocab_answer_dict[0] = 'START'
    vocab_answer_dict[len(vocab_answer_dict) + 1] = 'END'
    word_idx_answer = {val: key for key, val in vocab_answer_dict.items()}
    test = get_stories(open(test_file, 'r', encoding='utf-8'))
    vocab = sorted(reduce(lambda x, y: x | y,
                          (set(story + q + [answer]) for story, q, answer in train + test)))
    vocab_size = len(vocab) + 1
    story_maxlen = max(map(len, (x for x, _, _ in train + test)))
    query_maxlen = max(map(len, (x for _, x, _ in train + test)))
    VECTOR_LENGTH = 300
    X, XEmbedding, Xq, XqEmbedding, Y, Y_1, Y_2, Y_3, Y_4 = vectorize_stories(
        train, word_idx, word_idx_answer, story_maxlen, query_maxlen, word_embeddings)
    tX, tXEmbedding, tXq, tXqEmbedding, tY, tY_1, tY_2, tY_3, tY_4 = vectorize_stories(
        test, word_idx, word_idx_answer, story_maxlen, query_maxlen, word_embeddings)
    num_decoder_tokens = len(word_idx_answer)
    print('Build model...')
    print(vocab_size, vocab_answer_dict)
    input_story_embedding = Input(shape=(story_maxlen, VECTOR_LENGTH), name='story_word_embedding')
    input_query_embedding = Input(shape=(query_maxlen, VECTOR_LENGTH), name='query_word_embedding')
    no_of_hops = 2
    embeddings_A, embeddings_C = [], []
    concatenated = None
    for hop in range(no_of_hops):
        if hop == 0:
            input_m = Input(shape=(story_maxlen,))
            input_embedding_m = Embedding(
                input_dim=vocab_size, output_dim=64, input_length=story_maxlen)(input_m)
            dropout_input_m = Dropout(0.2)(input_embedding_m)
            embeddings_A.append((input_m, input_embedding_m, dropout_input_m))
            input_c = Input(shape=(story_maxlen,))
            input_embedding_c = Embedding(input_dim=vocab_size, output_dim=query_maxlen, input_length=story_maxlen)(input_c)
            dropout_input_c = Dropout(0.2)(input_embedding_c)
            embeddings_C.append((input_c, input_embedding_c, dropout_input_c))
            question = Input(shape=(query_maxlen,))
            question_embedding = Embedding(
                input_dim=vocab_size, output_dim=64, input_length=query_maxlen)(question)
            dropout_question = Dropout(
                0.2)(question_embedding)
            question_embedding_final = concatenate([input_query_embedding, dropout_question])
        else:
            input_m = embeddings_A[-1][0]
            input_embedding_m = embeddings_A[-1][1]
            dropout_input_m = embeddings_A[-1][2]
            input_c = embeddings_C[-1][0]
            input_embedding_c = embeddings_C[-1][1]
            dropout_input_c = embeddings_C[-1][2]
            question_embedding_final = concatenated
        print(dropout_input_m.name, dropout_input_m.shape)
        story_embeddings_m_final = concatenate([input_story_embedding, dropout_input_m])
        match = dot([story_embeddings_m_final, question_embedding_final], axes=(2, 2))
        activation_softmax = Activation('softmax')
        probabilities_for_query = activation_softmax(match)
        dropout_input_c = Dropout(0.2)(input_embedding_c)
        response = add([probabilities_for_query, dropout_input_c])
        permuted_response = Permute((2, 1))(response)
        output_dim = VECTOR_LENGTH + LSTM_OUTPUT_DIM
        dense_for_response = Dense(output_dim)(permuted_response)
        concatenated = add([dense_for_response, question_embedding_final])
    encoder = LSTM(LSTM_OUTPUT_DIM, return_sequences=True, return_state=True, name='Encoder_LSTM')
    encoder_states, h, c = encoder(concatenated)
    encoder_model = Model(inputs=[input_story_embedding, input_query_embedding, input_m, question, input_c], outputs=[encoder_states, h, c])
    print('encoder states', type(encoder_states))
    max_decoder_seq_length = 4
    # Set up the decoder, which will only process one timestep at a time.
    decoder_inputs_initial = Input(shape=(1, num_decoder_tokens))
    decoder_inputs_1 = Input(shape=(1, num_decoder_tokens))
    decoder_inputs_2 = Input(shape=(1, num_decoder_tokens,))
    decoder_inputs_3 = Input(shape=(1, num_decoder_tokens,))
    decoder_lstm = LSTM(LSTM_OUTPUT_DIM, return_sequences=True, return_state=True)
    decoder_dense = Dense(num_decoder_tokens, activation='softmax')
    # compute attention
    all_outputs = list()
    states = [h, c]
    inputs = decoder_inputs_initial
    for _ in range(max_decoder_seq_length):
        # Run the decoder on one timestep
        if _ == 1:
            inputs = decoder_inputs_1
        elif _ == 2:
            inputs = decoder_inputs_2
        else:
            inputs = decoder_inputs_3
        print('index', _, inputs.shape)
        # inputs = Reshape((1, num_decoder_tokens))(inputs)
        print('inpts', inputs.shape)
        outputs, state_h, state_c = decoder_lstm(inputs, initial_state=states)
        dot_product = dot([outputs, encoder_states], axes=[2, 2])
        attention_weights = Activation('softmax')(dot_product)
        attention_weights_permuted = Permute((2, 1))(attention_weights)
        print('att', attention_weights_permuted.shape, encoder_states.shape)
        weighted_encoder_states = dot([attention_weights_permuted, encoder_states], axes=[1, 1])
        print('wt', weighted_encoder_states.shape)
        outputs = decoder_dense(weighted_encoder_states)
        all_outputs.append(outputs)
        states = [state_h, state_c]
    # Concatenate all predictions
    decoder_outputs = Lambda(lambda x: K.concatenate(x, axis=1))(all_outputs)
    decoder_inputs = Input(shape=(1, num_decoder_tokens))
    decoder_state_input_h = Input(shape=(LSTM_OUTPUT_DIM,))
    decoder_state_input_c = Input(shape=(LSTM_OUTPUT_DIM,))
    encoder_states_to_decoder = Input(shape=(None, LSTM_OUTPUT_DIM,))
    decoder_states_inputs = [decoder_state_input_h, decoder_state_input_c]
    print('decoder_inputs', decoder_inputs.shape)
    outputs_lstm, state_h_dec, state_c_dec = decoder_lstm(decoder_inputs, initial_state=decoder_states_inputs)
    dot_product_dec = dot([outputs_lstm, encoder_states_to_decoder], axes=[2, 2])
    attention_weights_dec = Activation('softmax')(dot_product_dec)
    attention_weights_permuted_dec = Permute((2, 1))(attention_weights_dec)
    weighted_encoder_states_dec = dot([attention_weights_permuted_dec, encoder_states_to_decoder], axes=[1, 1])
    outputs_dec = decoder_dense(weighted_encoder_states_dec)
    decoder_model = Model([decoder_inputs, decoder_state_input_h, decoder_state_input_c, encoder_states_to_decoder], [outputs_dec, state_h_dec, state_c_dec])

    # Define and compile model as previously
    model = Model([input_story_embedding, input_query_embedding, input_m, question, input_c, decoder_inputs_initial, decoder_inputs_1, decoder_inputs_2, decoder_inputs_3], decoder_outputs)
    print(model.summary())
    model.compile(
        optimizer='adam', loss='categorical_crossentropy', metrics=['acc'])
    model.fit([XEmbedding, XqEmbedding, X, Xq, X, Y_1, Y_2, Y_3, Y_4], Y, batch_size=BATCH_SIZE, epochs=EPOCHS, validation_split=0.05)
    predicted_test_all, hidden_state_all, cell_state_all = encoder_model.predict([tXEmbedding, tXqEmbedding, tX, tXq, tX], batch_size=BATCH_SIZE)
    all_generated_equations = list()
    print(predicted_test_all.shape)
    samples = predicted_test_all.shape[0]
    print('#samples', samples)
    out_desc = open(out_file, 'w')
    all_generated_equations = list()
    for i in range(samples):
        hidden_state = hidden_state_all[i]
        cell_state = cell_state_all[i]
        all_hidden_states = predicted_test_all[i]
        equation_generated = predict_sequence_with_beam_search(decoder_model, all_hidden_states, hidden_state, cell_state, len(word_idx_answer), vocab_answer_dict, LSTM_OUTPUT_DIM, query_maxlen, 4, 3)
        all_generated_equations.append(equation_generated)
    out_desc.write('\n'.join(all_generated_equations) + '\n')


def dump_object_to_pickle_file(data_object, pickle_file_path):
    with open(pickle_file_path, 'wb') as dump_pickle:
        dump(data_object, dump_pickle)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train', dest='tr', help='Enter the training file')
    parser.add_argument('--test', dest='te', help='Enter the test file')
    parser.add_argument('--mem', dest='mem', help='Enter the word embeddings')
    parser.add_argument('--pkl', dest='pkl', help='Enter the word to index mapping')
    parser.add_argument('--epoch', dest='epoch', help='Enter the no of epochs', type=int)
    parser.add_argument('--out', dest='out', help='Enter the output file path where the predictions and gold output will be saved')
    args = parser.parse_args()
    word_idx = load_object_from_pickle(args.pkl)
    dimensions = 300
    word_embeddings = np.memmap(args.mem, dtype='float32', mode='r', shape=(len(word_idx), dimensions))
    print('word-embeddings-shape', word_embeddings.shape)
    train_model(args.tr, args.te, args.out, word_idx, word_embeddings, args.epoch)


if __name__ == '__main__':
    main()
