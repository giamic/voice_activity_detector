import tensorflow as tf
import math
import os

# PATHS
from tensorflow.python.framework.errors_impl import NotFoundError

ADDITIONAL_DATA_FOLDER = os.path.join(os.path.sep, 'media', 'giamic', 'Local Disk', 'music_analysis', 'data', 'spotify_previews', 'recordings')
DATA_FOLDER = os.path.join('..', 'data')
MODEL_BASE_FOLDER = os.path.join('..', 'models')
TRN_FOLDER = os.path.join(DATA_FOLDER, 'original', 'train')
VLD_FOLDER = os.path.join(DATA_FOLDER, 'original', 'validation')
TST_FOLDER = os.path.join(DATA_FOLDER, 'test')
MUSIC_FOLDER = os.path.join(DATA_FOLDER, 'music')
NOISE_FOLDER = os.path.join(DATA_FOLDER, 'noise')
IMPULSE_RESPONSE_FOLDER = os.path.join(DATA_FOLDER, 'impulse_response')
INPUT_TRN = os.path.join(DATA_FOLDER, 'train.tfrecords')
INPUT_VLD = os.path.join(DATA_FOLDER, 'validation.tfrecords')

# DATA PRE-PROCESSING
NUM_MFCCS = 20
MAX_SEQUENCE_LENGTH = 800
SR = 22050
N_FFT = 2048
HOP_LENGTH = 512
FRAME_SIZE = N_FFT / SR  # in s
FRAME_DELTA = HOP_LENGTH / SR  # distance between frames in s
MAX_AUDIO_LENGTH = MAX_SEQUENCE_LENGTH * HOP_LENGTH - 1

# MODEL PARAMETERS
N_HIDDEN_LAYER = 32
N_EMBEDDINGS = 16
BATCH_SIZE = 16
SHUFFLE_BUFFER = 2000

# PREDICTION STAGE
SMOOTHING = True
SMOOTHING_WINDOW_SIZE = 3
SMOOTHING_WINDOW = 'boxcar'
MIN_BEEP_LENGTH = 4

# TRAINING DURATION
N_EPOCHS = 50
try:
    N_TRN_EXAMPLES = sum(1 for _ in tf.python_io.tf_record_iterator(INPUT_TRN))
    TRN_STEPS_PER_EPOCH = math.ceil(N_TRN_EXAMPLES / BATCH_SIZE)
except NotFoundError:
    N_TRN_EXAMPLES, TRN_STEPS_PER_EPOCH = None, None
try:
    N_VLD_EXAMPLES = sum(1 for _ in tf.python_io.tf_record_iterator(INPUT_VLD))
    VLD_STEPS_PER_EPOCH = math.ceil(N_VLD_EXAMPLES / BATCH_SIZE)
except NotFoundError:
    N_VLD_EXAMPLES, VLD_STEPS_PER_EPOCH = None, None
try:
    N_STEPS = (TRN_STEPS_PER_EPOCH + 1) * N_EPOCHS
except TypeError:
    N_STEPS = None
# N_TRN_EXAMPLES = len(os.listdir(TRN_FOLDER)) // 2
# N_VLD_EXAMPLES = len(os.listdir(VLD_FOLDER)) // 2
