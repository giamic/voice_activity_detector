import tensorflow as tf
from tensorflow.contrib import rnn


def model_rnn(rnn_input, n_embeddings, *args, **kwargs):
    lstm_cell = rnn.BasicLSTMCell(n_embeddings, forget_bias=1.0)
    embeddings, state = rnn.static_rnn(lstm_cell, rnn_input, dtype=tf.float32)
    logits = []
    W = tf.Variable(tf.random_normal([n_embeddings, 1], stddev=0.1), name="dense_weights")
    b = tf.Variable(tf.zeros([1]), name="dense_biases")
    for e in embeddings:
        l = tf.add(tf.matmul(e, W), b)
        logits.append(l)
    return logits


def model_double_rnn(rnn_input, n_embeddings, n_hidden, *args, **kwargs):
    hidden_cell = rnn.BasicLSTMCell(n_hidden, forget_bias=1.0, name='first_layer')
    output_cell = rnn.BasicLSTMCell(n_embeddings, forget_bias=1.0, name='second_layer')
    hidden, state_hidden = rnn.static_rnn(hidden_cell, rnn_input, dtype=tf.float32)
    embeddings, state_output = rnn.static_rnn(output_cell, hidden, dtype=tf.float32)
    logits = []
    W = tf.Variable(tf.random_normal([n_embeddings, 1], stddev=0.1), name="dense_weights")
    b = tf.Variable(tf.zeros([1]), name="dense_biases")
    for e in embeddings:
        l = tf.add(tf.matmul(e, W), b)
        logits.append(l)
    return logits
