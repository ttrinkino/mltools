import tensorflow as tf
from tensorflow.contrib.layers import fully_connected, dropout, batch_norm, variance_scaling_initializer
from tensorflow.contrib.rnn import GRUCell, LSTMCell, LayerNormBasicLSTMCell, MultiRNNCell, DropoutWrapper
from tensorflow.contrib.distributions import Bernoulli, Normal
import numpy as np
from datetime import datetime as dt
from sklearn.utils import shuffle
import matplotlib.pyplot as plt
import os


def main():
    data = dict()
    data['X_train_static_mins'] = np.load('D:/data_files/Imbal_nntrain_long_X_train_static_mins.npy')
    data['X_train_time_0'] = np.load('D:/data_files/Imbal_nntrain_long_X_train_time_0.npy')

    vae = VAE(
        encoder_layer_sizes=[100, 50, 2],
        rnn_encoder_layer_sizes=[40, 20],
        encoder_dropout=0.8,
        rnn_encoder_dropout=0.8
    )

    vae.fit(
        data=data,
        epochs=1000,
        max_seconds=3000,
        batch_norm_decay=0.9,
        learning_rate=1e-5
    )

    X_sample, X_test_sample = vae.posterior_predictive_sample(data['X_train_static_mins'], data['X_train_time_0'])
    print(X_sample.shape, X_test_sample.shape)


# variational autoencoder with static and timeseries features accepted
class VAE:
    def __init__(self, encoder_layer_sizes, decoder_layer_sizes=None,
                 encoder_dropout=1.0, decoder_dropout=None,
                 rnn_encoder_layer_sizes=None, rnn_decoder_layer_sizes=None,
                 rnn_encoder_dropout=1.0, rnn_decoder_dropout=None,
                 save_file=None, tensorboard=None):

        self.encoder_layer_sizes = encoder_layer_sizes
        # copy encoder layer sizes to decoder
        if decoder_layer_sizes is None:
            self.decoder_layer_sizes = list(reversed(encoder_layer_sizes[:-1]))
        else:
            self.decoder_layer_sizes = decoder_layer_sizes
        if type(encoder_dropout) == list:
            self.encoder_dropout = encoder_dropout
        else:
            self.encoder_dropout = [encoder_dropout for _ in self.encoder_layer_sizes]
        if type(decoder_dropout) == list:
            self.decoder_dropout = decoder_dropout
        elif decoder_dropout is None:
            self.decoder_dropout = list(reversed(self.encoder_dropout))
        else:
            self.decoder_dropout = [decoder_dropout for _ in range(len(self.decoder_layer_sizes) + 1)]

        self.rnn_encoder_layer_sizes = rnn_encoder_layer_sizes
        # copy rnn encoder layer sizes to rnn decoder
        if rnn_encoder_layer_sizes is not None:
            if rnn_decoder_layer_sizes is None:
                self.rnn_decoder_layer_sizes = list(reversed(rnn_encoder_layer_sizes[:-1]))
            else:
                self.rnn_decoder_layer_sizes = rnn_decoder_layer_sizes
            self.rnn_encoder_dropout = rnn_encoder_dropout
            if rnn_decoder_dropout is None:
                self.rnn_decoder_dropout = rnn_encoder_dropout
            else:
                self.rnn_decoder_dropout = rnn_decoder_dropout

        self.save_file = save_file
        self.tensorboard = tensorboard
        if self.tensorboard is not None:
            saved = os.path.exits(self.tensorboard)
            assert saved == False

    def fit(self, data, epochs=1000, max_seconds=600, activation=tf.nn.elu,
            batch_norm_decay=0.9, learning_rate=1e-5, batch_sz=1024,
            adapt_lr=False, print_progress=True, show_fig=True):

        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

        # static features
        X = data['X_train_static_mins']
        N, D = X.shape
        self.X = tf.placeholder(tf.float32, shape=(None, D), name='X')

        # timeseries features
        X_time = data['X_train_time_0']
        T1, N1, D1 = X_time.shape
        assert N == N1
        self.X_time = tf.placeholder(tf.float32, shape=(T1, None, D1), name='X_time')
        self.train = tf.placeholder(tf.bool, shape=(), name='train')
        self.rnn_keep_p_encode = tf.placeholder(tf.float32, shape=(), name='rnn_keep_p_encode')
        self.rnn_keep_p_decode = tf.placeholder(tf.float32, shape=(), name='rnn_keep_p_decode')
        adp_learning_rate = tf.placeholder(tf.float32, shape=(), name='adp_learning_rate')

        he_init = variance_scaling_initializer()
        bn_params = {
            'is_training': self.train,
            'decay': batch_norm_decay,
            'updates_collections': None
        }
        latent_size = self.encoder_layer_sizes[-1]

        inputs = self.X
        with tf.variable_scope('static_encoder'):
            for layer_size, keep_p in zip(self.encoder_layer_sizes[:-1], self.encoder_dropout[:-1]):
                inputs = dropout(inputs, keep_p, is_training=self.train)
                inputs = fully_connected(
                    inputs, layer_size, weights_initializer=he_init, activation_fn=activation,
                    normalizer_fn=batch_norm, normalizer_params=bn_params
                )

        if self.rnn_encoder_layer_sizes:
            with tf.variable_scope('rnn_encoder'):
                rnn_cell = MultiRNNCell([
                    LayerNormBasicLSTMCell(s, activation=tf.tanh, dropout_keep_prob=self.rnn_encoder_dropout)
                    for s in self.rnn_encoder_layer_sizes]
                )
                time_inputs, states = tf.nn.dynamic_rnn(
                    rnn_cell, self.X_time, swap_memory=True,
                    time_major=True, dtype=tf.float32
                )
                time_inputs = tf.transpose(time_inputs, perm=(1, 0, 2))
                time_inputs = tf.reshape(time_inputs, shape=(-1, self.rnn_encoder_layer_sizes[-1] * T1))

            inputs = tf.concat([inputs, time_inputs], axis=1)

        with tf.variable_scope('latent_space'):
            inputs = dropout(inputs, self.encoder_dropout[-1], is_training=self.train)
            loc = fully_connected(
                inputs, latent_size, weights_initializer=he_init, activation_fn=None,
                normalizer_fn=batch_norm, normalizer_params=bn_params
            )
            scale = fully_connected(
                inputs, latent_size, weights_initializer=he_init, activation_fn=tf.nn.softplus,
                normalizer_fn=batch_norm, normalizer_params=bn_params
            )

            standard_normal = Normal(
                loc=np.zeros(latent_size, dtype=np.float32),
                scale=np.ones(latent_size, dtype=np.float32)
            )
            e = standard_normal.sample(tf.shape(loc)[0])
            outputs = e * scale + loc

            static_output_size = self.decoder_layer_sizes[0]
            if self.rnn_decoder_layer_sizes:
                time_output_size = self.rnn_decoder_layer_sizes[0] * T1
                output_size = static_output_size + time_output_size
            else:
                output_size = static_output_size
            outputs = fully_connected(
                outputs, output_size, weights_initializer=he_init, activation_fn=activation,
                normalizer_fn=batch_norm, normalizer_params=bn_params
            )
            if self.rnn_decoder_layer_sizes:
                outputs, time_outputs = tf.split(outputs, [static_output_size, time_output_size], axis=1)

        with tf.variable_scope('static_decoder'):
            for layer_size, keep_p in zip(self.decoder_layer_sizes, self.decoder_dropout[:-1]):
                outputs = dropout(outputs, keep_p, is_training=self.train)
                outputs = fully_connected(
                    outputs, layer_size, weights_initializer=he_init, activation_fn=activation,
                    normalizer_fn=batch_norm, normalizer_params=bn_params
                )
            outputs = dropout(outputs, self.decoder_dropout[-1], is_training=self.train)
            outputs = fully_connected(
                outputs, D, weights_initializer=he_init, activation_fn=None,
                normalizer_fn=batch_norm, normalizer_params=bn_params
            )

            X_hat = Bernoulli(logits=outputs)
            self.posterior_predictive = X_hat.sample()
            self.posterior_predictive_probs = tf.nn.sigmoid(outputs)

        if self.rnn_decoder_layer_sizes:
            with tf.variable_scope('rnn_decoder'):
                self.rnn_decoder_layer_sizes.append(D1)
                time_output_size = self.rnn_decoder_layer_sizes[0]
                time_outputs = tf.reshape(time_outputs, shape=(-1, T1, time_output_size))
                time_outputs = tf.transpose(time_outputs, perm=(1, 0, 2))
                rnn_cell = MultiRNNCell([
                    LayerNormBasicLSTMCell(s, activation=tf.tanh, dropout_keep_prob=self.rnn_decoder_dropout)
                    for s in self.rnn_decoder_layer_sizes]
                )
                time_outputs, states = tf.nn.dynamic_rnn(
                    rnn_cell, time_outputs, swap_memory=True,
                    time_major=True, dtype=tf.float32
                )
                time_outputs = tf.transpose(time_outputs, perm=(1, 0, 2))
                time_outputs = tf.reshape(time_outputs, shape=(-1, T1 * D1))
                X_hat_time = Bernoulli(logits=time_outputs)
                posterior_predictive_time = X_hat_time.sample()
                posterior_predictive_time = tf.reshape(posterior_predictive_time, shape=(-1, T1, D1))
                self.posterior_predictive_time = tf.transpose(posterior_predictive_time, perm=(1, 0, 2))
                self.posterior_predictive_probs_time = tf.nn.sigmoid(time_outputs)

        kl_div = -tf.log(scale) + 0.5 * (scale ** 2 + loc ** 2) - 0.5
        kl_div = tf.reduce_sum(kl_div, axis=1)

        expected_log_likelihood = tf.reduce_sum(
            X_hat.log_prob(self.X),
            axis=1
        )
        X_time_trans = tf.transpose(self.X_time, perm=(1, 0, 2))
        X_time_reshape = tf.reshape(X_time_trans, shape=(-1, T1 * D1))
        if self.rnn_encoder_layer_sizes:
            expected_log_likelihood_time = tf.reduce_sum(
                X_hat_time.log_prob(X_time_reshape),
                axis=1
            )
            elbo = -tf.reduce_sum(expected_log_likelihood + expected_log_likelihood_time - kl_div)
        else:
            elbo = -tf.reduce_sum(expected_log_likelihood - kl_div)
        train_op = tf.train.AdamOptimizer(learning_rate=adp_learning_rate).minimize(elbo)

        tf.summary.scalar('elbo', elbo)
        if self.save_file:
            saver = tf.train.Saver()

        if self.tensorboard:
            for v in tf.trainable_variables():
                tf.summary.histogram(v.name, v)
            train_merge = tf.summary.merge_all()
            writer = tf.summary.FileWriter(self.tensorboard)

        self.init_op = tf.global_variables_initializer()
        n = 0
        n_batches = N // batch_sz
        costs = list()
        min_cost = np.inf

        t0 = dt.now()
        with tf.Session() as sess:
            sess.run(self.init_op)
            for epoch in range(epochs):
                idxs = shuffle(range(N))
                X_train = X[idxs]
                X_train_time = X_time[:, idxs]

                for batch in range(n_batches):
                    n += 1
                    X_batch = X_train[batch * batch_sz:(batch + 1) * batch_sz]
                    X_batch_time = X_train_time[:, batch * batch_sz:(batch + 1) * batch_sz]

                    sess.run(
                        train_op,
                        feed_dict={
                            self.X: X_batch,
                            self.X_time: X_batch_time,
                            self.rnn_keep_p_encode: self.rnn_encoder_dropout,
                            self.rnn_keep_p_decode: self.rnn_decoder_dropout,
                            self.train: True,
                            adp_learning_rate: learning_rate
                        }
                    )
                    if n % 100 == 0 and print_progress:
                        cost = sess.run(
                            elbo,
                            feed_dict={
                                self.X: X,
                                self.X_time: X_time,
                                self.rnn_keep_p_encode: 1.0,
                                self.rnn_keep_p_decode: 1.0,
                                self.train: False
                            }
                        )
                        cost /= N
                        costs.append(cost)

                        if adapt_lr and epoch > 0:
                            if cost < min_cost:
                                min_cost = cost
                            elif cost > min_cost * 1.01:
                                learning_rate *= 0.75
                                if print_progress:
                                    print('Updating Learning Rate', learning_rate)

                        print('Epoch:', epoch, 'Batch:', batch, 'Cost:', cost)

                        if self.tensorboard:
                            train_sum = sess.run(
                                train_merge,
                                feed_dict={
                                    self.X: X,
                                    self.X_time: X_time,
                                    self.rnn_keep_p_encode: 1.0,
                                    self.rnn_keep_p_decode: 1.0,
                                    self.train: False
                                }
                            )
                            writer.add_summary(train_sum, n)

                seconds = (dt.now() - t0).seconds
                if seconds > max_seconds:
                    if print_progress:
                        print('Breaking after', seconds, 'seconds')
                    break

            if self.save_file:
                saver.save(sess, self.save_file)

            if self.tensorboard:
                writer.add_graph(sess.graph)

        if show_fig:
            plt.plot(costs)
            plt.title('Costs and Scores')
            plt.show()

    def posterior_predictive_sample(self, X, X_time):
        with tf.Session() as sess:
            sess.run(self.init_op)
            X_sample, X_time_sample = sess.run(
                (self.posterior_predictive, self.posterior_predictive_time),
                feed_dict={
                    self.X: X,
                    self.X_time: X_time,
                    self.rnn_keep_p_encode: 1.0,
                    self.rnn_keep_p_decode: 1.0,
                    self.train: False
                }
            )
        return X_sample, X_time_sample


if __name__ == '__main__':
    main()
