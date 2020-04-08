# Voice activity detector
# Candidate: Gianluca MICCHI

## The goal
Create a voice activity detector. I implemented it as a program that takes an audio file as an input and creates two files as an output: a json file containing the start and end time of the various speech segments (to mimic the provided dataset) and an audio file where a beep is superimposed to the original file when speech is detected, which allows for fast human verification of the quality of the software.

## The model
Since an annotated dataset was provided, I trained a supervised machine learning model. In particular, I used a Long Short-Term Memory Recurrent Neural Network (LSTM-RNN). I actually trained two models, one with a single-layer LSTM-RNN and the other with two stacked layers: the second is certainly more powerful but also requires more memory and more training time because it has a larger number of weights.

The idea is the following: the dataset is divided in two parts, training and validation; each audio file is read and a time series of 20 Mel Frequency Cepstral Coefficients (MFCCs) is extracted from it; the first feature is discarded because it contains the average energy of the audio file, which might confuse the network to associate speech with a certain volume regardless of the content; the series of the 19 remaining MFCCs is then given as an input to the rolled-out version of the LSTM-RNN which guesses once per each frame if speech is present; the network is trained, accuracy and loss on the validation set are calculated during training, and then finally the weights are saved, ready to be loaded by a second program to detect the speech in no matter what audio file. At prediction time, a 'glitch removal' algorithm is run after the neural networks outputs its predictions; the algorithms removes all predicted speech (or lack of speech) that are too short-lived to represent a real signal. A small test dataset with personal recordings is provided to show the result of the software.

The loss function is a standard cross entropy where positive and negative examples are given the same weight.

The code of the program is written in Python and Tensorflow.

## Technical considerations
Each paragraph of this section is devoted to a different aspect that deserves further discussion. For sake of brevity, not everything can be discussed here: I will be glad to answer to questions regarding other aspects and to delve deeper on the ones presented.

### Data augmentation
The dataset provided was of small size, containing around a thousand training examples. More importantly, it was made mostly (only?) of high-quality recordings of people talking. It was very easy for the model to learn parameters that allow for an extremely high accuracy on both training and validation set. However, when tested on a real-life recording, the software didn't achieve great results. Before improving the model I should have improved the data I was training on. 

Creating new data points manually is a long process, especially due to the necessity of labeling them: it would be maybe possible to write a second small piece of software to help one in such a task (you keep a button pressed when you hear the voice and the program automatically transcribes the annotation) but I didn't have the time to do that. I then thought of augmenting the data that I had. This was done in four different ways:
 - I added to the dataset some short excerpts of instrumental music (downloaded thanks to the Spotify API). They are very easy to label because they have an output that is constantly zero.
 - Furthermore, each original voice recording was associated with a random music excerpt (different in each case) and the two signals were added in order to reproduce the situation in which somebody is talking over some background music (the volume of the music was reduced). As labels, the labels for the original text are kept.
 - A similar procedure was done for noise recordings and I used the SLR28 dataset suggested in your email to obtain them. I actually didn't train the network to listen only to noise and outputting zero. Now that I think of it, I should have probably done it.
 - In the same dataset there are also recorded impulse responses. I then thought of taking them and convolving them with the original voice inputs in order to augment the data a little more. Once again, each voice recording had its own impulse response chosen at random. I honestly don't know how much this helps but the price to pay in terms of time was small so I did it anyway.

The dataset was pre-processed then stored as a .tfrecords file ready to be used.

### Why MFCCs?
First of all, one should answer another question: why not using the raw audio? The answer is that raw audio has very short time steps: the sampling rate used in this project is 22050 Hz. An LSTM-RNN typically can not retain information over thousands of steps and so the model becomes less powerful. One could maybe study a more complex architecture that is able to deal with raw audio but the requirements in terms of memory and computational power would explode, which would make the model impossible to train on my local machine and also maybe impossible to use on embedded chips.

MFCCs are features that derive from the spectral analysis of the signal and that synthesize the information about the frequency content in a way that is inspired by the biological perception of sound in the human ear. There are several research papers that show how using MFCCs instead of plain STFT coefficients improves the quality of the model. 

It is importance to notice that the choice of the features to describe the data point is of paramount importance for the quality of the resulting program. Up to my knowledge, MFCCs are sort of a standard for speech recognition and for this reason I chose them. However, a more extensive research on the subject would be something very important to do.

### Data pre-processing
In theory, RNNs work with sequences of variable length. In practice, at least in tensorflow, one establishes a sequence length and then fits the input data to have the desired size. What I've done is to cut an excerpt from the audio file if it's too long and to place it at a random position in the sequence if it's too short. I chose a sequence length of 800 sets of MFCCs, which corresponds to just shy of 20 seconds of audio with the parameters I'm using.

### No hangover scheme
Hangover schemes are sometimes used to help the program detect weak tails in the speech. After a brief analysis of the results without hangover scheme I decided that this was not the biggest of my issues so I focused on other things. As a consequence, no hangover scheme is implemented.

### No bi-directional LSTMs
There exists a variant of LSTMs that determines the output at a certain point of the sequence by using information not only from previous elements of the input sequencebut also from the following ones. This is equivalent to say that the network uses information from the future. I thought that, in a real-life product, this application needs to run in real time, therefore I have discarded this architecture. There might be tricks to reinsert it, however: in particular, if one works with very short sequences, one could predict the speech with a very short delay that might be acceptable. However, I didn't analyse this option further because it didn't look very promising.

### Other architectures:  WaveNet and Wave-U-Net
I just want to mention here that there alternatives to the architecture that I chose. For example, one could have a CNN-based analysis such as WaveNet or Wave-U-Net. Up to my knowledge, these architectures have been trained for other tasks (speech generation, audio source separation) but they might be adapted to the case at hand. However, I have never actually implemented them myself and I didn't try this time either, thinking I would need a bit more time to understand if they are really a viable alternative and, eventually, to code them.

### Tutorial for the code
The code has three entry points. When **dataset_creation.py** is run, two new tfrecords files are produced (one for training and the other for validation) with or without augmentation, with or without smoothing of the input data according to the configuration. **train.py** and **predict.py** are then used, their goal should be quite self-explanatory: **train.py** produces a model folder as an output, containing the saved trained model, some logging information for TensorBoard, and a file describing the architecture; **predict.py**, instead, produces for every test audio the two files that I mentioned at the beginning, the json file with the speech segments and the beeped audio file. All the configuration is stored in **config.py** and the models are defined in **models.py**. Finally, the **data_loading.py** file contains function to parse the tfrecords and import them in tensorflow.

## Results
In the same folder you will find some example of the audio output of the 2-layer LSTM-RNN model trained on augmented data. As one can hear, the program works quite well on clean audio inputs. It also deals quite well with music, ignoring it while still recognizing when somebody talks on top of it. Unfortunately, a noisy situation is still too much to handle and the detector is often on even when nobody is talking. **Probably, a better denoising system is necessary before calculating the MFCCs. So far I've tried to smooth the input data with a rectangular box of window size 3, but it didn't bring any quantifiable improvement.**

Regarding the tensorboard plots: they represent accuracy and loss on the validation set. The dark blue model is the single-layer LSTM-RNN on the initial, clean data set. We can see that it has a very high accuracy but the result of the prediction on the test files was not so great because the dataset was too easy and with not enough variance. The green line represents the same model trained on the augmented data. We can see that the performance has decreased considerably but only because the task has become harder. On a side note, there is an unexpected sudden decay midway through the training. Honestly, I didn't spend much time trying to understand it because I already had the results from the light blue line, that represents a stacked 2-layers LSTM-RNN model trained on the augmented data. We can see that it does much better than the single layer LSTM-RNN and it doesn't have the puzzling decay in performance.

## Possible improvements
I see several different ways of improving the results, some of which I mentioned already: training different models, introducing a better denoising algorithm, increasing the size of the dataset. For what concerns the metrics and the visualization of the results, one could definitely show the ROC curve and the area under it: I haven't done it so far because I implemented only two models and their comparison seems to me obvious enough not to need other metrics. Also, TensorFlow doesn't integrate them in an easy way and obtaining this result is a bit annoying (I've done it already, but it requires some time and lots of attention).
