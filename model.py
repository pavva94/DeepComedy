import tensorflow as tf
from keras.layers import Layer
from keras.layers import Add, Dense, LSTM
from keras.models import Model
import numpy as np


class BasicDanteRNN(Model):
    def __init__(self, latent_dim, n_tokens, tokenizer, generative=False):
        super(BasicDanteRNN, self).__init__()
        self.n_tokens = n_tokens
        self.latent_dim = latent_dim
        self.generative = generative
        self.tokenizer = tokenizer
        self.lstm = LSTM(latent_dim, return_state=True, return_sequences=True, name='lstm')
        self.tl1 = BasicTrainingLine(self.lstm, self.latent_dim, self.n_tokens)
        self.tl2 = BasicTrainingLine(self.lstm, self.latent_dim, self.n_tokens)
        self.tl3 = BasicTrainingLine(self.lstm, self.latent_dim, self.n_tokens)

    def call(self, inputs, training=None):
        # print(inputs) # (<tf.Tensor 'IteratorGetNext:0' shape=(None, 46, 41) dtype=float32>, <tf.Tensor 'ExpandDims:0' shape=(None, 1) dtype=int64>, <tf.Tensor 'IteratorGetNext:2' shape=(None, 46, 41) dtype=float32>, <tf.Tensor 'ExpandDims_1:0' shape=(None, 1) dtype=int64>, <tf.Tensor 'IteratorGetNext:4' shape=(None, 46, 41) dtype=float32>, <tf.Tensor 'ExpandDims_2:0' shape=(None, 1) dtype=int64>)
        outputs = []

        if self.generative:
            # syl1, syl2, syl3 = (11, 11, 11)
            # syl1 = tf.expand_dims((syl1), 0)
            # syl2 = tf.expand_dims((syl2), 0)
            # syl3 = tf.expand_dims((syl3), 0)

            # TODO: there is a better method for create a tensor with shape (None, 1) or at least with 2 dim
            input_eval, syl = inputs
            syl = tf.expand_dims(tf.expand_dims(syl, 0), 1)

            if input_eval is None:
                # using random start
                first_char = chr(int(np.random.randint(ord('a'), ord('z') + 1)))
                # print(tokenizer.texts_to_sequences(first_char))
                # Converting start string to numbers (vectorizing)
                input_eval = self.tokenizer.texts_to_sequences(first_char)[0]
                input_eval = tf.expand_dims(input_eval, 0)

            # print((input_eval, syl))

            x1 = self.tl1((input_eval, syl), training=False, previous_line=None)
            outputs.append(x1)

            # print(x1)

            x2 = self.tl2((x1, syl), training=False, previous_line=self.tl1)
            outputs.append(x2)
            # print(x2)

            x3 = self.tl3((x2, syl), training=False, previous_line=self.tl2)
            outputs.append(x3)
        else:

            char1, syl1, char2, syl2, char3, syl3 = inputs
            outputs.append(self.tl1((char1, syl1), training=training, previous_line=None))
            outputs.append(self.tl2((char2, syl2), training=training, previous_line=self.tl1))
            outputs.append(self.tl3((char3, syl3), training=training, previous_line=self.tl2))

        # input_layer = [self.inp1, self.inp2, self.inp3]

        # print(outputs)

        return outputs


class BasicTrainingLine(Layer):
    def __init__(self, lstm, latent_dim, n_tokens):
        super(BasicTrainingLine, self).__init__()
        self.lstm = lstm
        self.n_tokens = n_tokens
        self.dense_in = Dense(latent_dim, activation='relu')
        self.dense_out = Dense(self.n_tokens, activation='softmax')
        self.lstm_h = None
        self.lstm_c = None

    def call(self, inputs, training=None, previous_line=None, **kwargs):
        # print("BasicTrainingLine Start")
        # print(inputs)
        # x = self.syllable_input(inputs) NON SI FA PERCHé é UN INPUT LAYER
        # INPUTS: ListWrapper([<tf.Tensor 'input_151:0' shape=(None, None, 41) dtype=float32>, UN CARATTERE TOKENIZZATO
        #       <tf.Tensor 'input_152:0' shape=(None, 1) dtype=float32>]) NUMERO SILLABE

        chars, syllable = inputs
        # print(chars)
        # print(syllable)

        x = self.dense_in(syllable, training=training)
        # print(self.n_tokens)

        # print(x)

        if previous_line:
            # WHAT ARE THESE ADD? Simply layer that do an addition maybe
            initial_state = [
                Add()([
                    previous_line.lstm_h,
                    x
                ]),
                Add()([
                    previous_line.lstm_c,
                    x
                ])
            ]

            # print(previous_line.lstm_c)
        else:
            initial_state = [x, x]

        # print(initial_state)

        lstm_out, self.lstm_h, self.lstm_c = self.lstm(chars, initial_state=initial_state, training=training)
        outputs = self.dense_out(lstm_out, training=training)
        # print(lstm_out)
        # print(outputs)

        # print("BasicTrainingLine End")

        return outputs