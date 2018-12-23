from keras.models import Model
from keras.layers import TimeDistributed,Conv1D,Dense,Embedding,Input,Dropout,LSTM,Bidirectional,MaxPooling1D,Flatten,concatenate,Reshape,Convolution1D
from keras.initializers import RandomUniform
from keras import backend as K


def gen_CNN_RNN_model(wordEmbeddings,char2Idx,label2Idx,case2Idx):
    words_input = Input(shape=(None,), dtype='int32', name='words_input')
    words = Embedding(input_dim=wordEmbeddings.shape[0], output_dim=wordEmbeddings.shape[1], weights=[wordEmbeddings],
                      trainable=False)(words_input)
    casing_input= Input(shape=(None,1,), dtype='int32', name='casing_input')
    embed_casing_out = TimeDistributed(
        Embedding(len(case2Idx), 1, embeddings_initializer=RandomUniform(minval=-0.5, maxval=0.5)),
        name='case_embedding')(casing_input)
    dropout = Dropout(0.5)(embed_casing_out)
    conv1d_out = TimeDistributed(Conv1D(kernel_size=1, filters=1, padding='same', activation='relu', strides=1))(
        dropout)
    maxpool_out = TimeDistributed(MaxPooling1D(1))(conv1d_out)
    casing = TimeDistributed(Flatten())(maxpool_out)
    casing = Dropout(0.5)(casing)

    character_input = Input(shape=(None, 52,), name='char_input')
    embed_char_out = TimeDistributed(
        Embedding(len(char2Idx), 30, embeddings_initializer=RandomUniform(minval=-0.5, maxval=0.5)),
        name='char_embedding')(character_input)
    dropout = Dropout(0.5)(embed_char_out)
    conv1d_out = TimeDistributed(Conv1D(kernel_size=3, filters=30, padding='same', activation='tanh', strides=1))(
        dropout)
    maxpool_out = TimeDistributed(MaxPooling1D(52))(conv1d_out)
    char = TimeDistributed(Flatten())(maxpool_out)
    char = Dropout(0.5)(char)
    output = concatenate([words, casing, char])
    output = Bidirectional(LSTM(400, return_sequences=True, dropout=0.50, recurrent_dropout=0.25))(output)
    output = Bidirectional(LSTM(400, return_sequences=True, dropout=0.50, recurrent_dropout=0.25))(output)
    output = TimeDistributed(Dense(len(label2Idx), activation='softmax'))(output)
    model = Model(inputs=[words_input, casing_input, character_input], outputs=[output])
    model.compile(loss='sparse_categorical_crossentropy', optimizer='nadam')
    model.summary()

    return model


def gen_RNN_RNN_model(wordEmbeddings,caseEmbeddings,label2Idx):
    words_input = Input(shape=(None,), dtype='int32', name='words_input')
    words = Embedding(input_dim=wordEmbeddings.shape[0], output_dim=wordEmbeddings.shape[1], weights=[wordEmbeddings],
                      trainable=False)(words_input)
    casing_input = Input(shape=(None,), dtype='int32', name='casing_input')
    casing = Embedding(output_dim=caseEmbeddings.shape[1], input_dim=caseEmbeddings.shape[0], weights=[caseEmbeddings],
                       trainable=False)(casing_input)
    character_input = Input(shape=(None, 52,), name='char_input')

    rnn1 = Bidirectional(LSTM(200, return_sequences=True, dropout=0.50, recurrent_dropout=0.25))(character_input)
    rnn1 = Dropout(0.5)(rnn1)
    rnn1_out = TimeDistributed(Dense(len(label2Idx), activation='softmax'))(rnn1)

    char = TimeDistributed(Flatten())(rnn1_out)
    char = Dropout(0.5)(char)
    output = concatenate([words, casing, char])
    output = Bidirectional(LSTM(200, return_sequences=True, dropout=0.50, recurrent_dropout=0.25))(output)
    output = Bidirectional(LSTM(200, return_sequences=True, dropout=0.50, recurrent_dropout=0.25))(output)
    output = Dropout(0.5)(output)
    output = TimeDistributed(Dense(len(label2Idx), activation='softmax'))(output)
    model = Model(inputs=[words_input, casing_input, character_input], outputs=[output])
    model.compile(loss='sparse_categorical_crossentropy', optimizer='nadam')
    model.summary()

    return model
