import os
import numpy as np
import tensorflow as tf
import tensorflow_io as tfio
from librosa.filters import mel

from scipy import signal
from scipy.signal import get_window


from typing import List, Tuple, Dict, Any


def build_file_dataset(file_paths: List[str]) -> tf.data.Dataset:
    """Create a DataSet object containing all file paths"""
    file_paths = tf.convert_to_tensor(file_paths, dtype=tf.string)
    dataset = tf.data.Dataset.list_files(file_paths).prefetch(1)
    return dataset


def transform_files(files: tf.data.Dataset, process_fn: Any) -> tf.data.Dataset:
    """Apply processing step to dataset"""
    files = files.map(process_fn, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    return files


def load_audio(file_path: str) -> tf.Tensor:
    """Load and decode flac files"""
    audio = tf.io.read_file(file_path)
    audio = tfio.audio.decode_flac(audio, dtype=tf.int16)  # TODO: check datatype
    return audio


def _bytes_feature(value: Any) -> tf.train.Feature:
    """Returns a bytes_list from a string / byte."""
    if isinstance(value, type(tf.constant(0))):
        value = value.numpy()  # BytesList won't unpack a string from an EagerTensor.
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def _int64_feature(value: int) -> tf.train.Feature:
    """Returns an int64_list from a bool / enum / int / uint."""
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


# Create a dictionary with features that may be relevant.
def spectrogram_example(
    spectrogram_string: str, label: int, subset: bytes
) -> tf.train.Example:
    serialized_tensor = tf.io.serialize_tensor(spectrogram_string)
    feature = {
        "label": _int64_feature(label),
        "mel_spectrogram": _bytes_feature(serialized_tensor),
        "subset": _bytes_feature(subset),
    }

    return tf.train.Example(features=tf.train.Features(feature=feature))


def spectrogram_example_vctk(example: tf.train.Example) -> tf.train.Example:
    serialized_tensor = tf.io.serialize_tensor(example["speech"])

    feature = {
        "id": _bytes_feature(example["id"]),
        "speaker": _int64_feature(example["speaker"]),
        "gender": _int64_feature(example["gender"]),
        "accent": _int64_feature(example["accent"]),
        "speech": _bytes_feature(serialized_tensor),
    }

    return tf.train.Example(features=tf.train.Features(feature=feature))


def find_speaker_paths(dataset_path: str) -> Tuple[List, List]:
    """Find all speaker file paths for a certain dataset."""
    speaker_ids = [
        d
        for d in os.listdir(dataset_path)
        if os.path.isdir(os.path.join(dataset_path, d))
    ]
    speaker_file_paths = [os.path.join(dataset_path, sp_id) for sp_id in speaker_ids]

    return speaker_ids, speaker_file_paths


def find_flac_paths(speaker_path: str) -> List[str]:
    """Find all flac file paths for a certain speaker."""
    return get_all_files(speaker_path, extension=".flac")


def get_all_files(path: str, extension: str) -> List[str]:
    """Find all files within the path and deeper that end with .flac"""
    file_names = []
    for root, dirs, files in os.walk(path):
        for file in files:
            if file.endswith(extension):
                file_names.append(os.path.join(root, file))
    return file_names


def parse_spectrograms(example: Dict) -> Dict:
    """Convert the serialized tensor back to a tensor."""
    example["mel_spectrogram"] = tf.io.parse_tensor(
        example["mel_spectrogram"].numpy()[0], out_type=tf.float32
    )
    return example


def parse_spectrograms_vctk(example: Dict) -> Dict:
    example["speech"] = tf.io.parse_tensor(
        example["speech"].numpy()[0], out_type=tf.float32
    )
    return example


def raw_audio_to_spectrogram(speech_tensor: tf.Tensor, config: Dict) -> tf.Tensor:
    spectrogram = tfio.experimental.audio.spectrogram(
        speech_tensor,
        nfft=config["nfft"],
        window=config["window"],
        stride=config["stride"],
    )

    mel_spectrogram = tfio.experimental.audio.melscale(
        spectrogram,
        rate=config["rate"],
        mels=config["mels"],
        fmin=config["fmin"],
        fmax=config["fmax"],
    )

    return mel_spectrogram


def butter_highpass(cutoff, fs, order=5):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = signal.butter(order, normal_cutoff, btype='high', analog=False)
    return b, a


def pySTFT(x, fft_length=1024, hop_length=256):
    x = np.pad(x, int(fft_length // 2), mode='reflect')

    noverlap = fft_length - hop_length
    shape = x.shape[:-1] + ((x.shape[-1] - noverlap) // hop_length, fft_length)
    strides = x.strides[:-1] + (hop_length * x.strides[-1], x.strides[-1])
    result = np.lib.stride_tricks.as_strided(x, shape=shape,
                                             strides=strides)

    fft_window = get_window('hann', fft_length, fftbins=True)
    result = np.fft.rfft(fft_window * result, n=fft_length).T

    return np.abs(result)


def get_filters(config: Dict):
    mel_basis = mel(config["rate"], config["window"], fmin=90, fmax=config["fmax"], n_mels=config["mels"]).T
    min_level = np.exp(-100 / 20 * np.log(10))
    b, a = butter_highpass(30, config["rate"], order=5)

    return mel_basis, min_level, b, a


def raw_audio_to_spectrogram_np(speech_tensor: tf.Tensor, mel_basis, min_level, b, a):
    y = signal.filtfilt(b, a, speech_tensor.numpy())

    # Ddd a little random noise for model robustness
    # wav = y * 0.96 + (prng.rand(y.shape[0])-0.5)*1e-06

    # Compute spectrogram
    D = pySTFT(y).T

    # Convert to mel and normalize
    D_mel = np.dot(D, mel_basis)
    D_db = 20 * np.log10(np.maximum(min_level, D_mel)) - 16
    S = np.clip((D_db + 100) / 100, 0, 1)

    return S
