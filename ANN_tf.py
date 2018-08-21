from utils import getData, cnnTransform, accuracy_score
from tensorflow.contrib.rnn import GRUCell, LSTMCell, LayerNormBasicLSTMCell, MultiRNNCell, DropoutWrapper
from tensorflow.contrib.layers import fully_connected, dropout, batch_norm, variance_scaling_initializer, conv2d
import tensorflow as tf
import numpy as np
from sklearn.utils import shuffle
from sklearn.metrics import precision_score, recall_score, f1_score, log_loss
import matplotlib.pyplot as plt
from datetime import datetime as dt
from datetime import timedelta
import os


def main():
    data = getData(
        'D:/data_files/Imbal_nntrain_long.feather', load_file=True
    )

    save_file = False  # './tmp/ann'

    ann = ANN(
        rnn0_layer_sizes=[120, 90, 60, 30],
        rnn1_layer_sizes=[120, 90, 60, 30],
        rnn2_layer_sizes=[120, 90, 60, 30],
        rnn3_layer_sizes=[120, 90, 60, 30],
        cnn0_filter_sizes=[20, 20, 20, 20],
        cnn1_filter_sizes=[20, 20, 20, 20],
        cnn2_filter_sizes=[20, 20, 20, 20],
        cnn3_filter_sizes=[20, 20, 20, 20],
        fc_layer_sizes=[200, 150, 100, 50],
        final_layer_sizes=[400, 300, 200, 100],
        rnn0_keep_prob=0.8,
        rnn1_keep_prob=0.8,
        rnn2_keep_prob=0.8,
        rnn3_keep_prob=0.8,
        fc_keep_probs=0.8,
        final_keep_probs=0.8,
        save_file=save_file
    )

    ann.fit(
        data,
        rnn_cell_type=LayerNormBasicLSTMCell,
        learning_rate=1e-5,
        pos_weight=1.0,
        batch_norm_decay=0.9,
        max_seconds=6000,
        epochs=10000,
        batch_sz=1024,
        show_fig=True
    )

    # y_test = data['y_test']
    # X_test_static = data['X_test_static_mins']
    # X_test_time0 = data['X_test_time_0']
    # X_test_time1 = data['X_test_time_1']
    # X_test_time2 = data['X_test_time_2']
    # X_test_time3 = data['X_test_time_3']
    # X_test_rptime0 = data['X_test_rptime_0']
    # X_test_rptime1 = data['X_test_rptime_1']
    # X_test_rptime2 = data['X_test_rptime_2']
    # X_test_rptime3 = data['X_test_rptime_3']
    #
    # y_pred = load_predict(
    #     save_file, X_test_static, X_test_time0, X_test_time1, X_test_time2, X_test_time3,
    #     X_test_rptime0, X_test_rptime1, X_test_rptime2, X_test_rptime3
    # )

    # probs = 1 / (1 + np.exp(-y_pred))
    # probs = probs[:, 1] / np.sum(probs, axis=1)
    # y_pred = np.argmax(y_pred, axis=1)

    # np.save('./tmp/y_pred.npy', y_pred)
    # print(f1_score(y_test, y_pred), precision_score(y_test, y_pred), accuracy_score(y_test, y_pred))


class ANN:
    def __init__(self, rnn0_layer_sizes, rnn1_layer_sizes, rnn2_layer_sizes,
                 rnn3_layer_sizes, cnn0_filter_sizes, cnn1_filter_sizes,
                 cnn2_filter_sizes, cnn3_filter_sizes, fc_layer_sizes, final_layer_sizes=None,
                 rnn0_keep_prob=1.0, rnn1_keep_prob=1.0, rnn2_keep_prob=1.0,
                 rnn3_keep_prob=1.0, fc_keep_probs=None, final_keep_probs=None,
                 regression=False, save_file=False, tensorboard=False):

        self.rnn0_layer_sizes = rnn0_layer_sizes
        self.rnn1_layer_sizes = rnn1_layer_sizes
        self.rnn2_layer_sizes = rnn2_layer_sizes
        self.rnn3_layer_sizes = rnn3_layer_sizes
        self.cnn0_filter_sizes = cnn0_filter_sizes
        self.cnn1_filter_sizes = cnn1_filter_sizes
        self.cnn2_filter_sizes = cnn2_filter_sizes
        self.cnn3_filter_sizes = cnn3_filter_sizes
        self.fc_layer_sizes = fc_layer_sizes
        if final_layer_sizes:
            self.final_layer_sizes = final_layer_sizes
        else:
            self.final_layer_sizes = list()
        self.rnn0_keep_prob = rnn0_keep_prob
        self.rnn1_keep_prob = rnn1_keep_prob
        self.rnn2_keep_prob = rnn2_keep_prob
        self.rnn3_keep_prob = rnn3_keep_prob
        if type(fc_keep_probs) == float:
            self.fc_keep_probs = [fc_keep_probs for _ in fc_layer_sizes]
        else:
            self.fc_keep_probs = fc_keep_probs
        if type(final_keep_probs) == float:
            self.final_keep_probs = [final_keep_probs for _ in range(len(final_layer_sizes) + 1)]
        else:
            self.final_keep_probs = final_keep_probs
        self.regression = regression
        self.save_file = save_file
        self.tensorboard = tensorboard
        if self.tensorboard:
            saved = os.path.exists(self.tensorboard)
            assert(saved == False)

    def fit(self, data, learning_rate=1e-5, beta1=0.9, beta2=0.999, epsilon=1e-8, adapt_lr=False, pos_weight=1.0,
            cost_func=tf.train.AdamOptimizer, rnn_cell_type=LSTMCell, activation=tf.nn.elu, rnn_activation=tf.tanh,
            batch_norm_decay=0.9, epochs=100, max_seconds=600, use_peepholes=False, batch_sz=1000, show_fig=False,
            return_pred=False, return_scores=False, print_progress=True, reset_graph=False):

        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

        y_train, y_test = data['y_train'], data['y_test']
        N_test = y_test.shape[0]
        K = len(set(y_train))
        if K == 2:
            y_temp = np.zeros((len(y_train), K))
            y_temp[np.arange(len(y_train)), y_train] = 1
            y_train = y_temp
            y_temp = np.zeros((len(y_test), K))
            y_temp[np.arange(len(y_test)), y_test] = 1
            y_test = y_temp

        X_train_time0, X_test_time0 = data['X_train_time_0'], data['X_test_time_0']
        T, N, D_time0 = X_train_time0.shape
        X_time0 = tf.placeholder(tf.float32, shape=(T, None, D_time0), name='X_time0')

        X_train_time1, X_test_time1 = data['X_train_time_1'], data['X_test_time_1']
        T1, N1, D_time1 = X_train_time1.shape
        assert N1 == N
        X_time1 = tf.placeholder(tf.float32, shape=(T1, None, D_time1), name='X_time1')

        X_train_time2, X_test_time2 = data['X_train_time_2'], data['X_test_time_2']
        T2, N2, D_time2 = X_train_time2.shape
        assert N2 == N
        X_time2 = tf.placeholder(tf.float32, shape=(T2, None, D_time2), name='X_time2')

        X_train_time3, X_test_time3 = data['X_train_time_3'], data['X_test_time_3']
        T3, N3, D_time3 = X_train_time3.shape
        assert N3 == N
        X_time3 = tf.placeholder(tf.float32, shape=(T3, None, D_time3), name='X_time3')

        X_rptime0 = tf.placeholder(tf.float32, shape=(None, T, T, D_time0), name='X_rptime0')
        X_rptime1 = tf.placeholder(tf.float32, shape=(None, T1, T1, D_time1), name='X_rptime1')
        X_rptime2 = tf.placeholder(tf.float32, shape=(None, T2, T2, D_time2), name='X_rptime2')
        X_rptime3 = tf.placeholder(tf.float32, shape=(None, T3, T3, D_time3), name='X_rptime3')
        X_test_rptime0 = data['X_test_rptime_0']
        X_test_rptime1 = data['X_test_rptime_1']
        X_test_rptime2 = data['X_test_rptime_2']
        X_test_rptime3 = data['X_test_rptime_3']

        X_train_static, X_test_static = data['X_train_static_mins'], data['X_test_static_mins']
        N_static, D_static = X_train_static.shape
        assert N_static == N
        X_static = tf.placeholder(tf.float32, shape=(None, D_static), name='X_static')

        if K == 2:
            y = tf.placeholder(tf.float32, shape=(None, K), name='y')
        else:
            y = tf.placeholder(tf.int64, shape=(None,), name='y')
        train = tf.placeholder(tf.bool, shape=(), name='train')
        adaptive_lr = tf.placeholder(tf.float32, shape=(), name='adaptive_learning_rate')
        rnn0_keep_p = tf.placeholder(tf.float32, shape=(), name='rnn0_keep_p')
        rnn1_keep_p = tf.placeholder(tf.float32, shape=(), name='rnn1_keep_p')
        rnn2_keep_p = tf.placeholder(tf.float32, shape=(), name='rnn2_keep_p')
        rnn3_keep_p = tf.placeholder(tf.float32, shape=(), name='rnn3_keep_p')

        he_init = variance_scaling_initializer()
        bn_params = {
            'is_training': train,
            'decay': batch_norm_decay,
            'updates_collections': None
        }

        static_inputs = X_static
        with tf.name_scope('fc_layers'):
            if self.fc_keep_probs:
                for layer_size, keep_p in zip(self.fc_layer_sizes, self.fc_keep_probs):
                    static_inputs = dropout(static_inputs, keep_p, is_training=train)
                    static_inputs = fully_connected(
                        static_inputs, layer_size, weights_initializer=he_init, activation_fn=activation,
                        normalizer_fn=batch_norm, normalizer_params=bn_params
                    )
            else:
                for layer_size in self.fc_layer_sizes:
                    static_inputs = fully_connected(
                        static_inputs, layer_size, weights_initializer=he_init, activation_fn=activation,
                        normalizer_fn=batch_norm, normalizer_params=bn_params
                    )

        with tf.variable_scope('rnn0_vars', initializer=he_init):
            with tf.name_scope('rnn0_layers'):
                if rnn_cell_type == LayerNormBasicLSTMCell:
                    rnn_cell0 = MultiRNNCell(
                        [LayerNormBasicLSTMCell(s, dropout_keep_prob=rnn0_keep_p, activation=rnn_activation)
                         for s in self.rnn0_layer_sizes]
                    )
                elif rnn_cell_type == LSTMCell:
                    rnn_cell0 = MultiRNNCell([
                        DropoutWrapper(
                            LSTMCell(s, activation=rnn_activation, use_peepholes=use_peepholes),
                            input_keep_prob=rnn0_keep_p) for s in self.rnn0_layer_sizes]
                    )
                else:
                    rnn_cell0 = MultiRNNCell([
                        DropoutWrapper(
                            rnn_cell_type(s, activation=rnn_activation),
                            input_keep_prob=rnn0_keep_p) for s in self.rnn0_layer_sizes]
                    )
                outputs0, states0 = tf.nn.dynamic_rnn(
                    rnn_cell0, X_time0, swap_memory=True,
                    time_major=True, dtype=tf.float32
                )
            time_inputs0 = outputs0[-1]

        with tf.variable_scope('rnn1_vars', initializer=he_init):
            with tf.name_scope('rnn1_layers'):
                if rnn_cell_type == LayerNormBasicLSTMCell:
                    rnn_cell1 = MultiRNNCell(
                        [LayerNormBasicLSTMCell(s, dropout_keep_prob=rnn1_keep_p, activation=rnn_activation)
                         for s in self.rnn1_layer_sizes]
                    )
                elif rnn_cell_type == LSTMCell:
                    rnn_cell1 = MultiRNNCell([
                        DropoutWrapper(
                            LSTMCell(s, activation=rnn_activation, use_peepholes=use_peepholes),
                            input_keep_prob=rnn1_keep_p) for s in self.rnn1_layer_sizes]
                    )
                else:
                    rnn_cell1 = MultiRNNCell([
                        DropoutWrapper(
                            rnn_cell_type(s, activation=rnn_activation),
                            input_keep_prob=rnn1_keep_p) for s in self.rnn1_layer_sizes]
                    )
                outputs1, states1 = tf.nn.dynamic_rnn(
                    rnn_cell1, X_time1, swap_memory=True,
                    time_major=True, dtype=tf.float32
                )
            time_inputs1 = outputs1[-1]

        with tf.variable_scope('rnn2_vars', initializer=he_init):
            with tf.name_scope('rnn2_layers'):
                if rnn_cell_type == LayerNormBasicLSTMCell:
                    rnn_cell2 = MultiRNNCell(
                        [LayerNormBasicLSTMCell(s, dropout_keep_prob=rnn2_keep_p, activation=rnn_activation)
                         for s in self.rnn2_layer_sizes]
                    )
                elif rnn_cell_type == LSTMCell:
                    rnn_cell2 = MultiRNNCell([
                        DropoutWrapper(
                            LSTMCell(s, activation=rnn_activation, use_peepholes=use_peepholes),
                            input_keep_prob=rnn2_keep_p) for s in self.rnn2_layer_sizes]
                    )
                else:
                    rnn_cell2 = MultiRNNCell([
                        DropoutWrapper(
                            rnn_cell_type(s, activation=rnn_activation),
                            input_keep_prob=rnn2_keep_p) for s in self.rnn2_layer_sizes]
                    )
                outputs2, states2 = tf.nn.dynamic_rnn(
                    rnn_cell2, X_time2, swap_memory=True,
                    time_major=True, dtype=tf.float32
                )
            time_inputs2 = outputs2[-1]

        with tf.variable_scope('rnn3_vars', initializer=he_init):
            with tf.name_scope('rnn3_layers'):
                if rnn_cell_type == LayerNormBasicLSTMCell:
                    rnn_cell3 = MultiRNNCell(
                        [LayerNormBasicLSTMCell(s, dropout_keep_prob=rnn3_keep_p, activation=rnn_activation)
                         for s in self.rnn3_layer_sizes]
                    )
                elif rnn_cell_type == LSTMCell:
                    rnn_cell3 = MultiRNNCell([
                        DropoutWrapper(
                            LSTMCell(s, activation=rnn_activation, use_peepholes=use_peepholes),
                            input_keep_prob=rnn3_keep_p) for s in self.rnn3_layer_sizes]
                    )
                else:
                    rnn_cell3 = MultiRNNCell([
                        DropoutWrapper(
                            rnn_cell_type(s, activation=rnn_activation),
                            input_keep_prob=rnn3_keep_p) for s in self.rnn3_layer_sizes]
                    )
                outputs3, states3 = tf.nn.dynamic_rnn(
                    rnn_cell3, X_time3, swap_memory=True,
                    time_major=True, dtype=tf.float32
                )
            time_inputs3 = outputs3[-1]

        rptime_inputs0 = X_rptime0
        with tf.name_scope('cnn0_layers'):
            for layer_size in self.cnn0_filter_sizes:
                rptime_inputs0 = conv2d(
                    inputs=rptime_inputs0, num_outputs=layer_size,
                    kernel_size=3, stride=1, rate=1,
                    activation_fn=tf.nn.relu, padding='SAME',
                    normalizer_fn=batch_norm, normalizer_params=bn_params,
                    weights_initializer=he_init, biases_initializer=he_init
                )
                rptime_inputs0 = tf.nn.max_pool(
                    rptime_inputs0, ksize=[1, 2, 2, 1],
                    strides=[1, 2, 2, 1], padding='SAME'
                )
            dim = np.prod(rptime_inputs0.get_shape().as_list()[1:])
            rptime_inputs0 = tf.reshape(
                rptime_inputs0, [-1, dim]
            )

        rptime_inputs1 = X_rptime1
        with tf.name_scope('cnn1_layers'):
            for layer_size in self.cnn1_filter_sizes:
                rptime_inputs1 = conv2d(
                    inputs=rptime_inputs1, num_outputs=layer_size,
                    kernel_size=3, stride=1, rate=1,
                    activation_fn=tf.nn.relu, padding='SAME',
                    normalizer_fn=batch_norm, normalizer_params=bn_params,
                    weights_initializer=he_init, biases_initializer=he_init
                )
                rptime_inputs1 = tf.nn.max_pool(
                    rptime_inputs1, ksize=[1, 2, 2, 1],
                    strides=[1, 2, 2, 1], padding='SAME'
                )
            dim = np.prod(rptime_inputs1.get_shape().as_list()[1:])
            rptime_inputs1 = tf.reshape(
                rptime_inputs1, [-1, dim]
            )

        rptime_inputs2 = X_rptime2
        with tf.name_scope('cnn2_layers'):
            for layer_size in self.cnn2_filter_sizes:
                rptime_inputs2 = conv2d(
                    inputs=rptime_inputs2, num_outputs=layer_size,
                    kernel_size=3, stride=1, rate=1,
                    activation_fn=tf.nn.relu, padding='SAME',
                    normalizer_fn=batch_norm, normalizer_params=bn_params,
                    weights_initializer=he_init, biases_initializer=he_init
                )
                rptime_inputs2 = tf.nn.max_pool(
                    rptime_inputs2, ksize=[1, 2, 2, 1],
                    strides=[1, 2, 2, 1], padding='SAME'
                )
            dim = np.prod(rptime_inputs2.get_shape().as_list()[1:])
            rptime_inputs2 = tf.reshape(
                rptime_inputs2, [-1, dim]
            )

        rptime_inputs3 = X_rptime3
        with tf.name_scope('cnn3_layers'):
            for layer_size in self.cnn3_filter_sizes:
                rptime_inputs3 = conv2d(
                    inputs=rptime_inputs3, num_outputs=layer_size,
                    kernel_size=3, stride=1, rate=1,
                    activation_fn=tf.nn.relu, padding='SAME',
                    normalizer_fn=batch_norm, normalizer_params=bn_params,
                    weights_initializer=he_init, biases_initializer=he_init
                )
                rptime_inputs3 = tf.nn.max_pool(
                    rptime_inputs3, ksize=[1, 2, 2, 1],
                    strides=[1, 2, 2, 1], padding='SAME'
                )
            dim = np.prod(rptime_inputs3.get_shape().as_list()[1:])
            rptime_inputs3 = tf.reshape(
                rptime_inputs3, [-1, dim]
            )

        inputs = tf.concat(
            [static_inputs, time_inputs0, time_inputs1, time_inputs2, time_inputs3,
             rptime_inputs0, rptime_inputs1, rptime_inputs2, rptime_inputs3], axis=1
        )
        print('Final Input Shape', inputs.shape)

        with tf.name_scope('final_layers'):
            if self.final_keep_probs:
                for fc_size, keep_p in zip(self.final_layer_sizes, self.final_keep_probs[:-1]):
                    inputs = dropout(inputs, keep_p, is_training=train)
                    inputs = fully_connected(
                        inputs, fc_size, weights_initializer=he_init, activation_fn=activation,
                        normalizer_fn=batch_norm, normalizer_params=bn_params
                    )
                inputs = dropout(inputs, self.final_keep_probs[-1], is_training=train)
                logits = fully_connected(
                    inputs, K, weights_initializer=he_init, activation_fn=None,
                    normalizer_fn=batch_norm, normalizer_params=bn_params
                )
            else:
                for fc_size in self.final_layer_sizes:
                    inputs = fully_connected(
                        inputs, fc_size, weights_initializer=he_init, activation_fn=activation,
                        normalizer_fn=batch_norm, normalizer_params=bn_params
                    )
                logits = fully_connected(
                    inputs, K, weights_initializer=he_init, activation_fn=None,
                    normalizer_fn=batch_norm, normalizer_params=bn_params
                )
        with tf.name_scope('cost'):
            if K == 2:
                train_cost = tf.reduce_mean(
                    tf.nn.weighted_cross_entropy_with_logits(
                        logits=logits,
                        targets=y,
                        pos_weight=pos_weight
                    )
                )
                test_cost = tf.reduce_mean(
                    tf.nn.weighted_cross_entropy_with_logits(
                        logits=logits,
                        targets=y,
                        pos_weight=pos_weight
                    )
                )
            else:
                train_cost = tf.reduce_mean(
                    tf.nn.sparse_softmax_cross_entropy_with_logits(
                        logits=logits,
                        labels=y
                    )
                )
                test_cost = tf.reduce_mean(
                    tf.nn.sparse_softmax_cross_entropy_with_logits(
                        logits=logits,
                        labels=y
                    )
                )
        tf.summary.scalar('train_cost', train_cost)
        tf.summary.scalar('test_cost', test_cost, collections=['test_summaries'])

        with tf.name_scope('train_op'):
            if cost_func == tf.train.AdamOptimizer:
                opt = tf.train.AdamOptimizer(
                    learning_rate=adaptive_lr,
                    beta1=beta1,
                    beta2=beta2,
                    epsilon=epsilon,
                    name='adam'
                )
            elif cost_func == tf.train.RMSPropOptimizer:
                opt = tf.train.RMSPropOptimizer(
                    learning_rate=adaptive_lr,
                    decay=beta1,
                    momentum=beta2,
                    epsilon=epsilon,
                    name='rmsprop'
                )
            elif cost_func == tf.train.AdagradOptimizer:
                opt = tf.train.AdagradOptimizer(
                    learning_rate=adaptive_lr,
                    name='adagrad'
                )
            elif cost_func == tf.train.GradientDescentOptimizer:
                opt = tf.train.GradientDescentOptimizer(
                    learning_rate=adaptive_lr,
                    name='graddecent'
                )
            else:
                opt = cost_func(
                    learning_rate=adaptive_lr,
                    name='cost'
                )
            gradients = opt.compute_gradients(train_cost)
            gradients = map(
                lambda grad: grad if grad[0] is None else
                [tf.clip_by_value(grad[0], -10., 10.), grad[1]],
                gradients
            )
            train_op = opt.apply_gradients(gradients)

        with tf.name_scope('accuracy_op'):
            probs = tf.cast(logits, tf.float32, name='probs')
            prediction = tf.argmax(logits, 1, name='predict')
            one_hot = tf.one_hot(prediction, K)
            if K == 2:
                train_accuracy = tf.reduce_mean(
                    tf.cast(tf.equal(
                        one_hot,
                        y
                    ), tf.float32
                    )
                )
                accuracy = tf.reduce_mean(
                    tf.cast(tf.equal(
                        one_hot,
                        y
                    ), tf.float32
                    )
                )
            else:
                train_accuracy = tf.reduce_mean(
                    tf.cast(tf.equal(
                        prediction,
                        y
                    ), tf.float32
                    )
                )
                accuracy = tf.reduce_mean(
                    tf.cast(tf.equal(
                        prediction,
                        y
                    ), tf.float32
                    )
                )
        tf.summary.scalar('train_accuracy', train_accuracy)
        tf.summary.scalar('accuracy', accuracy, collections=['test_summaries'])

        if self.tensorboard:
            for v in tf.trainable_variables():
                tf.summary.histogram(v.name, v)
            train_merge = tf.summary.merge_all()
            test_merge = tf.summary.merge_all(key='test_summaries')
            writer = tf.summary.FileWriter(self.tensorboard)

        init_sess = tf.global_variables_initializer()

        if self.save_file:
            saver = tf.train.Saver()

        n = 0
        max_score = 0
        min_cost = np.inf
        train_costs = list()
        test_costs = list()
        accuracies = list()
        train_accuracies = list()
        precisions = list()
        recalls = list()
        f1s = list()
        test_batch_sz = 1024
        n_batches = N // batch_sz
        n_test_batches = N_test // test_batch_sz + 1
        calc_acc = dt.now()

        t0 = dt.now()
        with tf.Session() as sess:
            sess.run(init_sess)
            for epoch in range(epochs):
                idxs = shuffle(range(N))
                y_train = y_train[idxs]
                X_train_static = X_train_static[idxs]
                X_train_time0 = X_train_time0[:, idxs]
                X_train_time1 = X_train_time1[:, idxs]
                X_train_time2 = X_train_time2[:, idxs]
                X_train_time3 = X_train_time3[:, idxs]
                # X_train_rptime0 = X_train_rptime0[:, idxs]
                # X_train_rptime1 = X_train_rptime1[:, idxs]
                # X_train_rptime2 = X_train_rptime2[:, idxs]
                # X_train_rptime3 = X_train_rptime3[:, idxs]

                for batch in range(n_batches):
                    n += 1
                    y_batch = y_train[batch * batch_sz: (batch + 1) * batch_sz]
                    X_batch_static = X_train_static[batch * batch_sz:(batch + 1) * batch_sz]
                    X_batch_time0 = X_train_time0[:, batch * batch_sz:(batch + 1) * batch_sz]
                    X_batch_time1 = X_train_time1[:, batch * batch_sz:(batch + 1) * batch_sz]
                    X_batch_time2 = X_train_time2[:, batch * batch_sz:(batch + 1) * batch_sz]
                    X_batch_time3 = X_train_time3[:, batch * batch_sz:(batch + 1) * batch_sz]
                    X_batch_rptime0 = cnnTransform(X_batch_time0)
                    X_batch_rptime1 = cnnTransform(X_batch_time1)
                    X_batch_rptime2 = cnnTransform(X_batch_time2)
                    X_batch_rptime3 = cnnTransform(X_batch_time3)

                    sess.run(
                        train_op,
                        feed_dict={
                            X_static: X_batch_static, X_time0: X_batch_time0,
                            X_time1: X_batch_time1, X_time2: X_batch_time2,
                            X_time3: X_batch_time3, X_rptime0: X_batch_rptime0,
                            X_rptime1: X_batch_rptime1, X_rptime2: X_batch_rptime2,
                            X_rptime3: X_batch_rptime3, y: y_batch,
                            rnn0_keep_p: self.rnn0_keep_prob, rnn1_keep_p: self.rnn1_keep_prob,
                            rnn2_keep_p: self.rnn2_keep_prob, rnn3_keep_p: self.rnn3_keep_prob,
                            train: True, adaptive_lr: learning_rate
                        }
                    )

                    if dt.now() > calc_acc or (epoch == epochs - 1 and batch == n_batches - 1):
                        calc_acc = dt.now() + timedelta(0, 300)
                        trainc, train_acc = sess.run(
                            [train_cost, train_accuracy],
                            feed_dict={
                                X_static: X_batch_static, X_time0: X_batch_time0,
                                X_time1: X_batch_time1, X_time2: X_batch_time2,
                                X_time3: X_batch_time3, X_rptime0: X_batch_rptime0,
                                X_rptime1: X_batch_rptime1, X_rptime2: X_batch_rptime2,
                                X_rptime3: X_batch_rptime3, y: y_batch,
                                rnn0_keep_p: 1.0, rnn1_keep_p: 1.0,
                                rnn2_keep_p: 1.0, rnn3_keep_p: 1.0,
                                train: False
                            }
                        )
                            # feed_dict = {
                            #     X_static: X_train_static, X_time0: X_train_time0,
                            #     X_time1: X_train_time1, X_time2: X_train_time2,
                            #     X_time3: X_train_time3, X_rptime0: X_train_rptime0,
                            #     X_rptime1: X_train_rptime1, X_rptime2: X_train_rptime2,
                            #     X_rptime3: X_train_rptime3, y: y_train,
                            #     rnn0_keep_p: 1.0, rnn1_keep_p: 1.0,
                            #     rnn2_keep_p: 1.0, rnn3_keep_p: 3.0,
                            #     train: False
                            # }
                        testc = list()
                        acc = list()
                        test_prec = list()
                        recall = list()
                        f1 = list()
                        y_prob = list()
                        for test_batch in range(n_test_batches):
                            idx0 = test_batch * test_batch_sz
                            idx1 = (test_batch + 1) * test_batch_sz
                            X_test_batch_static = X_test_static[idx0:idx1]
                            X_test_batch_time0 = X_test_time0[:, idx0:idx1]
                            X_test_batch_time1 = X_test_time1[:, idx0:idx1]
                            X_test_batch_time2 = X_test_time2[:, idx0:idx1]
                            X_test_batch_time3 = X_test_time3[:, idx0:idx1]
                            X_test_batch_rptime0 = X_test_rptime0[idx0:idx1]
                            X_test_batch_rptime1 = X_test_rptime1[idx0:idx1]
                            X_test_batch_rptime2 = X_test_rptime2[idx0:idx1]
                            X_test_batch_rptime3 = X_test_rptime3[idx0:idx1]
                            y_test_batch = y_test[idx0:idx1]
                            test_len = y_test_batch.shape[0]

                            testc_, acc_, test_preds, y_probs = sess.run(
                                [test_cost, accuracy, prediction, probs],
                                feed_dict={
                                    X_static: X_test_batch_static, X_time0: X_test_batch_time0,
                                    X_time1: X_test_batch_time1, X_time2: X_test_batch_time2,
                                    X_time3: X_test_batch_time3, X_rptime0: X_test_batch_rptime0,
                                    X_rptime1: X_test_batch_rptime1, X_rptime2: X_test_batch_rptime2,
                                    X_rptime3: X_test_batch_rptime3, y: y_test_batch,
                                    rnn0_keep_p: 1.0, rnn1_keep_p: 1.0,
                                    rnn2_keep_p: 1.0, rnn3_keep_p: 1.0,
                                    train: False
                                }
                            )
                            if K == 2:
                                y_test_batch = np.argmax(y_test_batch, 1)
                                if np.sum(test_preds) > 0 and np.sum(y_test_batch) > 0:
                                    test_prec_ = precision_score(y_test_batch, test_preds)
                                    recall_ = recall_score(y_test_batch, test_preds)
                                    f1_ = f1_score(y_test_batch, test_preds)
                                else:
                                    test_prec_ = 0.
                                    recall_ = 0.
                                    f1_ = 0.
                            else:
                                max_cval = K - 1
                                for cval in range(1, max_cval):
                                    y_test_batch[y_test_batch == cval] = 0
                                    test_preds[test_preds == cval] = 0
                                y_test_batch[y_test_batch == max_cval] = 1
                                test_preds[test_preds == max_cval] = 1
                                if np.sum(test_preds) > 0 and np.sum(y_test_batch) > 0:
                                    test_prec_ = precision_score(y_test_batch, test_preds)
                                    recall_ = recall_score(y_test_batch, test_preds)
                                    f1_ = f1_score(y_test_batch, test_preds)
                                else:
                                    test_prec_ = 0.
                                    recall_ = 0.
                                    f1_ = 0.
                            testc_ = testc_ * (test_len / N_test)
                            acc_ = acc_ * (test_len / N_test)
                            test_prec_ = test_prec_ * (test_len / N_test)
                            recall_ = recall_ * (test_len / N_test)
                            f1_ = f1_ * (test_len / N_test)
                            testc.append(testc_)
                            acc.append(acc_)
                            test_prec.append(test_prec_)
                            recall.append(recall_)
                            f1.append(f1_)
                            y_prob.append(y_probs)

                        testc = np.sum(testc)
                        acc = np.sum(acc)
                        test_prec = np.sum(test_prec)
                        recall = np.sum(recall)
                        f1 = np.sum(f1)
                        train_costs.append(trainc)
                        train_accuracies.append(train_acc)
                        test_costs.append(testc)
                        accuracies.append(acc)
                        precisions.append(test_prec)
                        recalls.append(recall)
                        f1s.append(f1)
                        y_prob = np.concatenate(y_prob)

                        if adapt_lr and epoch > 0:
                            if testc < min_cost:
                                min_cost = testc
                            elif testc > min_cost * 1.01:
                                min_cost = testc
                                learning_rate *= 0.5
                                print('Updating Learning Rate', learning_rate)

                        if self.save_file and epoch > 2 and test_prec > max_score:
                            max_score = test_prec
                            np.save(self.save_file + '_probs' + str(epoch) + '.npy', y_prob)
                            saver.save(sess, self.save_file + '_best' + str(epoch))

                        if print_progress:
                            print('Epoch:', epoch, 'Batch:', batch, 'Train Costs:', round(trainc, 4),
                                  'Test Costs:', round(testc, 4), 'Train Accuracy:', round(train_acc, 4),
                                  'Accuracy:', round(acc, 4), 'Precision:', round(test_prec, 4),
                                  'Recall:', round(recall, 4), 'F1:', round(f1, 4))

                        if self.tensorboard:
                            train_sum = sess.run(
                                train_merge,
                                feed_dict={
                                    X_static: X_train_static, X_time0: X_train_time0,
                                    X_time1: X_train_time1, X_time2: X_train_time2,
                                    X_time3: X_train_time3, X_rptime0: X_train_rptime0,
                                    X_rptime1: X_train_rptime1, X_rptime2: X_train_rptime2,
                                    X_rptime3: X_train_rptime3, y: y_train,
                                    rnn0_keep_p: 1.0, rnn1_keep_p: 1.0,
                                    rnn2_keep_p: 1.0, rnn3_keep_p: 1.0,
                                    train: False
                                }
                            )
                            test_sum = sess.run(
                                test_merge,
                                feed_dict={
                                    X_static: X_test_static, X_time0: X_test_time0,
                                    X_time1: X_test_time1, X_time2: X_test_time2,
                                    X_time3: X_test_time3, X_rptime0: X_test_rptime0,
                                    X_rptime1: X_test_rptime1, X_rptime2: X_test_rptime2,
                                    X_rptime3: X_test_rptime3, y: y_test,
                                    rnn0_keep_p: 1.0, rnn1_keep_p: 1.0,
                                    rnn2_keep_p: 1.0, rnn3_keep_p: 1.0,
                                    train: False
                                }
                            )
                            writer.add_summary(train_sum, n)
                            writer.add_summary(test_sum, n)

                seconds = (dt.now() - t0).seconds
                if seconds > max_seconds:
                    if print_progress:
                        print('Breaking after', seconds, 'seconds')
                    break

            if self.save_file:
                np.save(self.save_file + '_probs.npy', y_prob)
                saver.save(sess, self.save_file)

            if self.tensorboard:
                writer.add_graph(sess.graph)

            if return_pred:
                y_pred = list()
                for test_batch in range(n_test_batches):
                    idx0 = test_batch * test_batch_sz
                    idx1 = (test_batch + 1) * test_batch_sz
                    X_test_batch_static = X_test_static[idx0:idx1]
                    X_test_batch_time0 = X_test_time0[:, idx0:idx1]
                    X_test_batch_time1 = X_test_time1[:, idx0:idx1]
                    X_test_batch_time2 = X_test_time2[:, idx0:idx1]
                    X_test_batch_time3 = X_test_time3[:, idx0:idx1]
                    X_test_batch_rptime0 = X_test_rptime0[idx0:idx1]
                    X_test_batch_rptime1 = X_test_rptime1[idx0:idx1]
                    X_test_batch_rptime2 = X_test_rptime2[idx0:idx1]
                    X_test_batch_rptime3 = X_test_rptime3[idx0:idx1]

                    y_pred_batch = sess.run(
                        probs,
                        feed_dict={
                            X_static: X_test_batch_static,
                            X_time0: X_test_batch_time0,
                            X_time1: X_test_batch_time1,
                            X_time2: X_test_batch_time2,
                            X_time3: X_test_batch_time3,
                            X_rptime0: X_test_batch_rptime0,
                            X_rptime1: X_test_batch_rptime1,
                            X_rptime2: X_test_batch_rptime2,
                            X_rptime3: X_test_batch_rptime3,
                            rnn0_keep_p: 1.0,
                            rnn1_keep_p: 1.0,
                            rnn2_keep_p: 1.0,
                            rnn3_keep_p: 1.0,
                            train: False
                        }
                    )
                    y_pred.append(y_pred_batch)
                y_pred = np.concatenate(y_pred, axis=0)

        if show_fig:
            plt.plot(train_costs)
            plt.plot(test_costs)
            plt.plot(accuracies)
            plt.plot(train_accuracies)
            plt.plot(precisions)
            plt.plot(recalls)
            plt.plot(f1s)
            plt.title('Costs and Scores')
            plt.show()

        if reset_graph:
            tf.reset_default_graph()

        if return_scores and return_pred:
            return train_costs, test_costs, train_accuracies, accuracies, y_pred
        elif return_scores:
            return train_costs, test_costs, train_accuracies, accuracies
        elif return_pred:
            return y_pred


def load_predict(filename, X_test_static, X_test_time0, X_test_time1, X_test_time2, X_test_time3,
                       X_test_rptime0, X_test_rptime1, X_test_rptime2, X_test_rptime3, probs=False):
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

    N_test = X_test_static.shape[0]
    test_batch_sz = 1024
    n_test_batches = N_test // test_batch_sz + 1
    saver = tf.train.import_meta_graph(filename + '.meta')
    with tf.Session() as sess:
        saver.restore(sess, tf.train.latest_checkpoint('./tmp/'))
        graph = tf.get_default_graph()
        X_static = graph.get_tensor_by_name('X_static:0')
        X_time0 = graph.get_tensor_by_name('X_time0:0')
        X_time1 = graph.get_tensor_by_name('X_time1:0')
        X_time2 = graph.get_tensor_by_name('X_time2:0')
        X_time3 = graph.get_tensor_by_name('X_time3:0')
        X_rptime0 = graph.get_tensor_by_name('X_rptime0:0')
        X_rptime1 = graph.get_tensor_by_name('X_rptime1:0')
        X_rptime2 = graph.get_tensor_by_name('X_rptime2:0')
        X_rptime3 = graph.get_tensor_by_name('X_rptime3:0')
        rnn0_keep_p = graph.get_tensor_by_name('rnn0_keep_p:0')
        rnn1_keep_p = graph.get_tensor_by_name('rnn1_keep_p:0')
        rnn2_keep_p = graph.get_tensor_by_name('rnn2_keep_p:0')
        rnn3_keep_p = graph.get_tensor_by_name('rnn3_keep_p:0')
        train = graph.get_tensor_by_name('train:0')
        if probs:
            prediction = graph.get_tensor_by_name('accuracy_op/probs:0')
        else:
            prediction = graph.get_tensor_by_name('accuracy_op/predict:0')

        y_pred = list()
        for test_batch in range(n_test_batches):
            idx0 = test_batch * test_batch_sz
            idx1 = (test_batch + 1) * test_batch_sz
            X_test_batch_static = X_test_static[idx0:idx1]
            X_test_batch_time0 = X_test_time0[:, idx0:idx1]
            X_test_batch_time1 = X_test_time1[:, idx0:idx1]
            X_test_batch_time2 = X_test_time2[:, idx0:idx1]
            X_test_batch_time3 = X_test_time3[:, idx0:idx1]
            X_test_batch_rptime0 = X_test_rptime0[idx0:idx1]
            X_test_batch_rptime1 = X_test_rptime1[idx0:idx1]
            X_test_batch_rptime2 = X_test_rptime2[idx0:idx1]
            X_test_batch_rptime3 = X_test_rptime3[idx0:idx1]

            y_pred_batch = sess.run(
                prediction,
                feed_dict={
                    X_static: X_test_batch_static,
                    X_time0: X_test_batch_time0,
                    X_time1: X_test_batch_time1,
                    X_time2: X_test_batch_time2,
                    X_time3: X_test_batch_time3,
                    X_rptime0: X_test_batch_rptime0,
                    X_rptime1: X_test_batch_rptime1,
                    X_rptime2: X_test_batch_rptime2,
                    X_rptime3: X_test_batch_rptime3,
                    rnn0_keep_p: 1.0,
                    rnn1_keep_p: 1.0,
                    rnn2_keep_p: 1.0,
                    rnn3_keep_p: 1.0,
                    train: False
                }
            )
            y_pred.append(y_pred_batch)
        y_pred = np.concatenate(y_pred, axis=0)

        return y_pred


if __name__ == '__main__':
    main()