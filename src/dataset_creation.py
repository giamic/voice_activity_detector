import json
import shutil

import librosa
import numpy as np
import scipy.signal
import tensorflow as tf

from config import *


def train_validation_split(data_folder, ratio=0.9, audio_format='wav', labels=True):
    file_names = sorted(os.listdir(data_folder))
    audio_files = [f for f in file_names if f[-len(audio_format):] == audio_format]
    label_files = [f[:-4] + '.json' for f in audio_files]

    if len(audio_files) == 0:
        print("No data available for splitting, I'm leaving.")
        return

    try:
        os.makedirs(os.path.join(data_folder, 'train'))
        os.makedirs(os.path.join(data_folder, 'validation'))
    except IOError:
        pass
    if labels:
        for a, l in zip(audio_files, label_files):
            if np.random.random() > ratio:
                os.rename(os.path.join(data_folder, a), os.path.join(data_folder, 'validation', a))
                os.rename(os.path.join(data_folder, l), os.path.join(data_folder, 'validation', l))
            else:
                os.rename(os.path.join(data_folder, a), os.path.join(data_folder, 'train', a))
                os.rename(os.path.join(data_folder, l), os.path.join(data_folder, 'train', l))
    else:
        for a in audio_files:
            if np.random.random() > ratio:
                os.rename(os.path.join(data_folder, a), os.path.join(data_folder, 'validation', a))
            else:
                os.rename(os.path.join(data_folder, a), os.path.join(data_folder, 'train', a))
    return


def smooth(x, window_len, window='hann'):
    if x.ndim != 1:
        raise ValueError("smooth only accepts 1 dimension arrays.")

    if x.size < window_len:
        raise ValueError("Input vector needs to be bigger than window size.")

    if window_len < 3:
        return x

    w = eval('scipy.signal.windows.' + window + '(window_len)')
    s = np.r_[x[window_len - 1:0:-1], x, x[-2:-window_len - 1:-1]]
    y = scipy.signal.fftconvolve(s, w / sum(w), mode='same')
    return y


def create_label_input(label_file, offset):
    with open(label_file) as f:
        d = json.load(f)
    labels_long = np.zeros(MAX_AUDIO_LENGTH, dtype=np.int32)
    for s in d['speech_segments']:
        start = max(math.floor(s['start_time'] * SR) + offset, 0)
        end = max(math.ceil(s['end_time'] * SR) + offset, 0)
        labels_long[start:end] = 1
    labels = np.zeros(MAX_SEQUENCE_LENGTH, dtype=np.int32)
    for i in range(MAX_SEQUENCE_LENGTH):
        labels[i] = int(np.round(np.mean(labels_long[i * HOP_LENGTH:(i + 1) * HOP_LENGTH])))
    return labels


def create_audio_input(audio_file, music_folder=None, noise_folder=None, impulse_response_folder=None, augment=False,
                       smoothing=False):
    """

    :param audio_file:
    :return: input, t_total, n_frames
    """
    audio, _ = librosa.load(audio_file, sr=SR)
    if smoothing:
        audio = smooth(audio, SMOOTHING_WINDOW_SIZE)
    audio, offset = _extend_audio(audio)
    x = []
    mfccs = librosa.feature.mfcc(audio, sr=SR, n_mfcc=NUM_MFCCS, n_fft=N_FFT, hop_length=HOP_LENGTH)
    x.append(mfccs[1:])
    if augment:
        if noise_folder is not None:
            noise = _prepare_confounding_sound(noise_folder, '.wav')
            scaling_factor = 0.33 * (max(audio) - min(audio)) / (max(noise) - min(noise))
            out = audio + scaling_factor * noise
            mfccs = librosa.feature.mfcc(out, sr=SR, n_mfcc=NUM_MFCCS, n_fft=N_FFT, hop_length=HOP_LENGTH)
            x.append(mfccs[1:])
        else:
            noise = np.random.normal(0, 0.01, MAX_AUDIO_LENGTH)
            mfccs = librosa.feature.mfcc(audio + noise, sr=SR, n_mfcc=NUM_MFCCS, n_fft=N_FFT, hop_length=HOP_LENGTH)
            x.append(mfccs[1:])
        if music_folder is not None:
            music = _prepare_confounding_sound(music_folder, '.mp3')
            scaling_factor = 0.33 * (max(audio) - min(audio)) / (max(music) - min(music))
            out = audio + scaling_factor * music
            mfccs = librosa.feature.mfcc(out, sr=SR, n_mfcc=NUM_MFCCS, n_fft=N_FFT, hop_length=HOP_LENGTH)
            x.append(mfccs[1:])
        if impulse_response_folder is not None:
            resp = _load_one_audio_file(impulse_response_folder, '.wav')
            out = scipy.signal.fftconvolve(audio, resp, mode='same')
            out /= max(np.abs(out))
            mfccs = librosa.feature.mfcc(out, sr=SR, n_mfcc=NUM_MFCCS, n_fft=N_FFT, hop_length=HOP_LENGTH)
            x.append(mfccs[1:])
    return x, offset


def _prepare_confounding_sound(folder, suffix):
    sound = _load_one_audio_file(folder, suffix)
    sound = _extend_by_repeating(sound)
    return sound


def _load_one_audio_file(folder, suffix):
    audio_files = os.listdir(folder)
    audio_files = [f for f in audio_files if f[-len(suffix):] == suffix]
    audio_file = audio_files[np.random.randint(len(audio_files))]
    audio, _ = librosa.load(os.path.join(folder, audio_file), sr=SR)
    return audio


def _extend_by_repeating(music):
    extended_music = np.zeros(MAX_AUDIO_LENGTH)
    window = scipy.signal.windows.boxcar(len(music))
    music *= window
    for i in range(len(extended_music)):
        extended_music[i] = music[i % len(music)]
    return extended_music


def _extend_audio(audio):
    extended_audio = np.zeros(MAX_AUDIO_LENGTH)
    if len(audio) == 0:  # the audio file is empty
        offset = 0
    elif MAX_AUDIO_LENGTH >= len(audio):  # the audio file is too short
        offset = np.random.randint(0, MAX_AUDIO_LENGTH - len(audio) + 1)
        extended_audio[offset:offset + len(audio)] = audio  # put the audio file somewhere in the middle
    else:  # the audio file is too long
        offset = np.random.randint(0, len(audio) - MAX_AUDIO_LENGTH + 1)
        extended_audio = audio[offset:offset + MAX_AUDIO_LENGTH]  # take an excerpt of the audio file
        offset = -offset
    return extended_audio, offset


def create_input(audio_file, label_file=None, music_folder=None, noise_folder=None, impulse_response_folder=None,
                 augment=False, smoothing=False):
    x, offset = create_audio_input(audio_file, music_folder, noise_folder, impulse_response_folder, augment, smoothing)
    if label_file is not None:
        y = create_label_input(label_file, offset)
    else:
        y = np.zeros(MAX_SEQUENCE_LENGTH, dtype=np.int32)
    return x, y


def transform_into_tfrecord(data_folder, output_path, music_folder=None, noise_folder=None,
                            impulse_response_folder=None):
    if os.path.isfile(output_path):
        raise PermissionError("The output file already exists. Exiting to prevent data loss.")
    audio_files = [os.path.join(data_folder, f) for f in (os.listdir(data_folder)) if f[-5:] != '.json']
    label_files = [f[:-4] + '.json' for f in audio_files]
    music_files = [os.path.join(music_folder, f) for f in (os.listdir(music_folder)) if f[-5:] != '.json']

    with tf.python_io.TFRecordWriter(output_path) as writer:
        n = 0
        for mf in music_files:
            if (n % 10) == 0:
                print("music file {} out of {}".format(n, len(music_files)))
            x, y = create_input(mf, noise_folder=noise_folder, augment=True, smoothing=SMOOTHING)
            for i in x:
                i = i.flatten()
                example = tf.train.Example()
                example.features.feature["x"].float_list.value.extend(i)
                example.features.feature["label"].int64_list.value.extend(y)
                writer.write(example.SerializeToString())
            n += 1
        n = 0
        for af, lf in zip(audio_files, label_files):
            if (n % 10) == 0:
                print("original example {} out of {}".format(n, len(audio_files)))
            x, y = create_input(af, lf, music_folder, noise_folder, impulse_response_folder, augment=True,
                                smoothing=SMOOTHING)
            for i in x:
                i = i.flatten()
                example = tf.train.Example()
                example.features.feature["x"].float_list.value.extend(i)
                example.features.feature["label"].int64_list.value.extend(y)
                writer.write(example.SerializeToString())
            n += 1

    return


def take_additional_songs(additional_data_path, output_path, n_songs_per_author=20):
    authors = os.listdir(additional_data_path)
    authors = [a for a in authors if os.path.isdir(a)]
    labels = {'speech_segments': []}
    for a in authors:
        print("working on author")
        songs = os.listdir(os.path.join(additional_data_path, a))
        l = len(songs)
        for _ in range(n_songs_per_author):
            i = np.random.randint(l)
            shutil.move(os.path.join(additional_data_path, a, songs[i]), os.path.join(output_path, songs[i]))
            with open(os.path.join(output_path, songs[i][:-3] + 'json'), 'w') as f:
                json.dump(labels, f)
    return


if __name__ == '__main__':
    # train_validation_split(NOISE_FOLDER, labels=False)
    # train_validation_split(IMPULSE_RESPONSE_FOLDER, labels=False)

    mf_trn, mf_vld = os.path.join(MUSIC_FOLDER, 'train'), os.path.join(MUSIC_FOLDER, 'validation')
    nf_trn, nf_vld = os.path.join(NOISE_FOLDER, 'train'), os.path.join(NOISE_FOLDER, 'validation')
    irf_trn, irf_vld = os.path.join(IMPULSE_RESPONSE_FOLDER, 'train'), os.path.join(IMPULSE_RESPONSE_FOLDER,
                                                                                    'validation')

    transform_into_tfrecord(TRN_FOLDER, INPUT_TRN, mf_trn, nf_trn, irf_trn)
    transform_into_tfrecord(VLD_FOLDER, INPUT_VLD, mf_vld, nf_vld, irf_vld)
    # take_additional_songs(ADDITIONAL_DATA_FOLDER, os.path.join(DATA_FOLDER, 'additional'))
    # train_validation_split(os.path.join(DATA_FOLDER, 'additional'), audio_format='mp3')
