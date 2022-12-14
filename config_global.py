# logmel spectrogram values
sample_rate = 44100
n_fft=(2560*sample_rate)//44100
# hop_length = 251
hop_length=(694*sample_rate)//44100
n_mels = 128
fmin = 20
remove_codec_from_filename = True
fmax = 8000
num_cores = -1 # -1 implies take all available cores.