import os
from pathlib import Path
import librosa
import dill as pickle
import numpy as np
# from threading import Parallel, delayed
from joblib import  Parallel,delayed
from glob import glob
from tqdm import tqdm
import argparse
from loguru import logger
from config_global import n_fft, hop_length, n_mels, fmin, fmax, sample_rate, num_cores, remove_codec_from_filename
import warnings

import torch
import laion_clap


number_of_files_success = 0
logger.add(f'compute_logmel_sr={sample_rate}.log')

@logger.catch
def remove_codec_substr(filename: str, remove_codec_from_filename: bool = True):
    """Utility function to remove codec substring from audio files in audioset dataset.

    Args:
        filename (str): Full filepath of audio file
        remove_codec_from_filename (bool, optional): If true will remove the codec substring. Defaults to remove_codec_from_filename.

    Returns:
        str: Final filepath to be used.
    """
    output_filename = os.path.basename(filename)
    if remove_codec_from_filename:
        output_filename = output_filename[:output_filename.rindex('_')]+'.wav'
    return output_filename


@logger.catch
def compute_melspec(filename, outdir):
    global number_of_files_success
    try:
        embedding = model.get_audio_embedding_from_filelist([filename],use_tensor=False)
        filename = Path(filename).name
        save_path = os.path.join(outdir, filename + '.npy')

        np.save(save_path, embedding)
        logger.success(save_path)
        number_of_files_success+=1
    except ValueError:
        print('ERROR IN:', filename)
        logger.error(f"{filename} - {save_path}")


@logger.catch
def main(input_path, output_path):
    logger.info(f"PARAMS:")
    logger.info(f"n_fft = {n_fft}")
    logger.info(f"hop_length = {hop_length}")
    logger.info(f"n_mels = {n_mels}")
    logger.info(f"fmin = {fmin}")
    logger.info(f"fmax = {fmax}")
    logger.info(f"sample_rate = {sample_rate}")
    logger.info(f"num_cores = {num_cores}")
    logger.info(f"remove_codec_from_filename = {remove_codec_from_filename}")
    logger.info(f'Starting computing logmels using above params.')
    file_list = glob(input_path + '/*.wav')
    output_path = os.path.join(output_path)
    os.makedirs(output_path, exist_ok=True)
    _ = Parallel(n_jobs=num_cores, prefer="threads")(
        delayed(lambda x: compute_melspec(
            x, output_path))(x)
        for x in tqdm(file_list))
    global number_of_files_success
    logger.success(f'Finished computing logmels using sr = {sample_rate}, total successfully converted to logmels = {number_of_files_success}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Input and Output Paths')
    parser.add_argument('input_path', type=str,
                        help="Specifies directory of audio files.")
    parser.add_argument('output_path', type=str,
                        help="Specifies directory for generated spectrograms.")
    args = parser.parse_args()

    model = laion_clap.CLAP_Module(enable_fusion=False)
    model.load_ckpt()
    # print(args.output_path)
    main(args.input_path, args.output_path)
