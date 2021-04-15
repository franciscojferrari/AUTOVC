import os
import tensorflow as tf
import tensorflow_io as tfio

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
def spectrogram_example(spectrogram_string: str, label: int, subset:str) -> tf.train.Example:
    serialized_tensor = tf.io.serialize_tensor(spectrogram_string)
    feature = {
        "label": _int64_feature(label),
        "mel_spectrogram": _bytes_feature(serialized_tensor),
        "subset": _bytes_feature(tf.io.serialize_tensor(subset))
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


def _parse_spectrograms(example: Dict) -> Dict:
    """Convert the serialized tensor back to a tensor."""
    example["mel_spectrogram"] = tf.io.parse_tensor(
        example["mel_spectrogram"], out_type=tf.float32
    )
    return example
