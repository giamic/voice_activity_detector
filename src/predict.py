import json
import sys

import librosa
import numpy as np
import scipy.signal
import tensorflow as tf

from config import *
from dataset_creation import create_audio_input
from models import model_rnn, model_double_rnn


def recreate_output(prediction, offset, output_file):
    output = {'speech_segments': []}
    p, count, start, end = 0, 0, 0, 0

    it = iter(prediction)
    while True:
        try:
            n = next(it)
        except StopIteration:
            break
        if n != p:
            if end >= start:
                start = count
            else:
                end = count
                d = {"start_time": round(start * FRAME_DELTA - offset / SR, 2),
                     "end_time": round(end * FRAME_DELTA - offset / SR, 2)}
                output['speech_segments'].append(d)
        count += 1
        p = n
    with open(output_file, 'w') as f:
        json.dump(output, f)
    return


def create_beeping_audio(audio_file, prediction, offset):
    audio, sr = librosa.load(audio_file, sr=SR)
    beep = np.array([])
    sound_frequency = 10. / HOP_LENGTH
    max_volume = 0.25
    for p in prediction:
        beep = np.append(beep, max_volume * p * np.sin(2 * math.pi * sound_frequency * np.arange(HOP_LENGTH)))
    if offset >= 0:
        beep = beep[offset:offset + len(audio)]
    else:
        empty = np.zeros(len(audio))
        for i in range(len(beep)):
            empty[offset + 1] = beep[i]
        beep = empty
        # beep = np.append(np.zeros(-offset), beep)
    output = beep + audio
    output_file = audio_file[:-4] + '_beep_do.wav'
    librosa.output.write_wav(output_file, output, sr)
    return


def desparsify_output(prediction):
    prev_count, curr_count, prev_value, curr_value = 0, 0, 0, 1
    for i in range(len(prediction)):
        if prediction[i] == curr_value:
            curr_count += 1
        else:
            if curr_count >= MIN_BEEP_LENGTH:
                prev_count, curr_count = curr_count, 1
            elif 0 < curr_count < MIN_BEEP_LENGTH:
                for j in range(i - curr_count, i):
                    prediction[j] = prediction[i]
                prev_count, curr_count = prev_count + curr_count + 1, prev_count + curr_count + 1
            else:  # only possible case: curr_count == 0
                curr_count = 1
            prev_value, curr_value = curr_value, prev_value
    return prediction


def detect_voice(audio_files, model_folder):
    if not isinstance(audio_files, list):
        audio_files = [audio_files]
    x = tf.placeholder(tf.float32)
    rnn_input = tf.unstack(tf.reshape(x, [1, NUM_MFCCS - 1, MAX_SEQUENCE_LENGTH]), axis=-1)
    # model = model_rnn
    model = model_double_rnn

    logits = model(rnn_input, N_EMBEDDINGS, N_HIDDEN_LAYER)
    y = tf.cast(tf.greater(logits, 0), tf.int32)

    with tf.Session() as sess:
        saver = tf.train.Saver()
        saver.restore(sess, os.path.join(model_folder, "model.ckpt"))

        for af in audio_files:
            data, offset = create_audio_input(af, smoothing=SMOOTHING)
            pred = sess.run(y, feed_dict={x: data})
            pred = desparsify_output(np.squeeze(pred))

            output_file = af[:-3] + 'json'
            recreate_output(pred, offset, output_file)
            create_beeping_audio(af, pred, offset)
    return


if __name__ == '__main__':
    if len(sys.argv) == 1:
        model_folder = os.path.join('..', 'models', 'model_double_rnn_2018-10-10_00-48-29')
        audio_files = os.listdir(TST_FOLDER)
        audio_files = [os.path.join(TST_FOLDER, af) for af in audio_files if
                       len(af.split('_')) == 1 and af[-4:] != 'json']
    else:
        audio_files, model_folder = sys.argv[1], sys.argv[2]
    detect_voice(audio_files, model_folder)
