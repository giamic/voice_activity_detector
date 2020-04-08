import numpy as np
from datetime import datetime

import tensorflow as tf
from tensorflow.python.profiler import option_builder
from tensorflow.python.profiler.model_analyzer import Profiler
from tensorflow.python.summary.writer.writer import FileWriter
from tensorflow.python.training.saver import Saver

from config import *
from dataset_load import create_tfrecords_iterator
from models import model_rnn, model_double_rnn

model = model_rnn
# model = model_double_rnn
model_folder = os.path.join(MODEL_BASE_FOLDER, model.__name__ + '_' + datetime.now().strftime('%Y-%m-%d_%H-%M-%S'))

trn_itr = create_tfrecords_iterator(INPUT_TRN, BATCH_SIZE, SHUFFLE_BUFFER)
vld_itr = create_tfrecords_iterator(INPUT_VLD, BATCH_SIZE, SHUFFLE_BUFFER)

handle = tf.placeholder(tf.string, shape=[])
x, y = tf.data.Iterator.from_string_handle(handle, trn_itr.output_types, trn_itr.output_shapes).get_next()

rnn_input = tf.unstack(tf.reshape(x, [BATCH_SIZE, NUM_MFCCS-1, MAX_SEQUENCE_LENGTH]), axis=-1)
labels = tf.unstack(y, axis=-1)

logits = model(rnn_input, N_EMBEDDINGS, N_HIDDEN_LAYER)

loss = tf.losses.sigmoid_cross_entropy(labels, logits)
train_step = tf.train.AdamOptimizer().minimize(loss, global_step=tf.train.create_global_step())
tf.summary.scalar('loss', loss)

accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.greater(logits, 0), tf.cast(labels, tf.bool)), tf.float32))
tf.summary.scalar('accuracy', accuracy)

with tf.Session() as sess:
    tf.global_variables_initializer().run()
    th = sess.run(trn_itr.string_handle())
    vh = sess.run(vld_itr.string_handle())

    merged = tf.summary.merge_all()
    trn_writer = FileWriter(os.path.join(model_folder, 'train'), sess.graph)
    vld_writer = FileWriter(os.path.join(model_folder, 'validation'))
    saver = Saver()
    profiler = Profiler(sess.graph)
    opts = (option_builder.ProfileOptionBuilder(option_builder.ProfileOptionBuilder.trainable_variables_parameter())
            .with_file_output(os.path.join(model_folder, 'profile_model.txt')).build())
    profiler.profile_name_scope(options=opts)

    value_lv = None
    lv = tf.Summary()
    lv.value.add(tag='loss', simple_value=value_lv)
    value_av = None
    av = tf.Summary()
    av.value.add(tag='accuracy', simple_value=value_av)

    for n in range(N_STEPS):
        print("step {} out of {}".format(n, N_STEPS))
        global_step = sess.run(tf.train.get_global_step())
        if n % TRN_STEPS_PER_EPOCH == 0 or n == N_STEPS - 1:
            print("test time")
            # the following code is just to calculate accuracy and loss on the entire validation set
            acc_vld, lss_vld = 0, 0
            for i in range(VLD_STEPS_PER_EPOCH):
                summary, acc, lss, = sess.run([merged, accuracy, loss], feed_dict={handle: vh})
                acc_vld += acc
                lss_vld += lss
            acc_vld /= VLD_STEPS_PER_EPOCH
            lss_vld /= VLD_STEPS_PER_EPOCH

            av.value[0].simple_value = acc_vld
            lv.value[0].simple_value = lss_vld
            vld_writer.add_summary(av, global_step=global_step)
            vld_writer.add_summary(lv, global_step=global_step)

            # summary = sess.run(merged, feed_dict={handle: vh})
            # vld_writer.add_summary(summary, global_step=global_step)
            saver.save(sess, os.path.join(model_folder, "model.ckpt"))
        else:
            summary, _ = sess.run([merged, train_step], feed_dict={handle: th})
            if np.random.random() > 0.9:
                trn_writer.add_summary(summary, global_step=global_step)
