import tensorflow_io as tfio
import matplotlib.pyplot as plt
from data_processing.utils import *


class DataWriter:
    def __init__(self, bucket_name, datasets, config):
        self.bucket_path = bucket_name
        self.datasets = datasets
        self.config = config

    def process_datasets(self, verbose=False, train_split=0.8):
        """Process datasets"""

        for dataset_name in self.datasets:
            dataset_path = os.path.join(
                self.bucket_path, self.config["dataset"][dataset_name]
            )
            speaker_ids, speaker_paths = find_speaker_paths(dataset_path)

            for speaker_path, speaker_id in zip(speaker_paths, speaker_ids):
                print(f"Processing data for speaker: {speaker_id}")

                label = int(speaker_id)
                file_paths = find_flac_paths(speaker_path)

                file_data_set = build_file_dataset(file_paths)
                processed_files = transform_files(file_data_set, self.process_files)

                record_file_name = f"{dataset_name}/{label}.tfrecords"
                write_path = f"{self.bucket_path}/processed_datasets/librispeech"
                record_file = os.path.join(write_path, record_file_name)
                n_samples = tf.data.experimental.cardinality(
                    processed_files).numpy()
                with tf.io.TFRecordWriter(record_file) as writer:
                    for i, processed_file in processed_files.enumerate():
                        subset = "train" if i <= train_split * n_samples else "test"
                        if verbose:
                            plt.figure(figsize=(15,4))
                            data = tf.math.log(processed_file).numpy()
                            plt.imshow(data, aspect="auto")
                            plt.show()
                        tf_example = spectrogram_example(processed_file, label, subset)
                        writer.write(tf_example.SerializeToString())

    def process_files(self, file_path: str) -> tf.Tensor:
        """Load audio file from disk and perform processing step to get mel spectorgram"""
        audio_tensor = load_audio(file_path)

        tensor = tf.cast(audio_tensor, tf.float32) / 32768.0
        tensor = tensor[:, 0]

        spectogram = tfio.experimental.audio.spectrogram(
            tensor,
            nfft=self.config["nfft"],
            window=self.config["window"],
            stride=self.config["stride"],
        )
        mel_spectogram = tfio.experimental.audio.melscale(
            spectogram,
            rate=self.config["rate"],
            mels=self.config["mels"],
            fmin=self.config["fmin"],
            fmax=self.config["fmax"],
        )

        return mel_spectogram


class DataReader:
    def __init__(self, base_path, dataset):
        self.base_path = base_path
        self.dataset = dataset
        self.feature_description = {
            "label": tf.io.FixedLenFeature([], tf.int64),
            "mel_spectrogram": tf.io.RaggedFeature(tf.string),
            "sub_set": tf.io.FixedLenFeature([], tf.string)
        }  # Assuming that the data only contains these two attributes

        self.speaker_files = None
        self.speaker_ids = None

        self.datasets = {}

    def _parse_function(self, example_proto):
        # Parse the input tf.train.Example proto using the provided dictionary
        return tf.io.parse_single_example(example_proto, self.feature_description)

    def read_data_set(self, tfrecord_path: str) -> tf.data.TFRecordDataset:
        """Read speaker dataset and parse it."""
        processed_dataset = tf.data.TFRecordDataset(tfrecord_path)
        parsed_dataset = processed_dataset.map(self._parse_function)
        return parsed_dataset

    def find_speaker_datasets(self) -> None:
        """Find all speaker datasets in dataset directory."""
        path = os.path.join(self.base_path, self.dataset)
        for root, dirs, files in os.walk(path):
            for file in files:
                if file.endswith(".tfrecords"):
                    self.speaker_files.append(os.path.join(root, file))
                    self.speaker_ids.append(file.split(".")[0])

    def load_speaker_datasets(self) -> None:
        """Load all speaker datasets into a dictionary."""
        for speaker_file, speaker_id in zip(self.speaker_files, self.speaker_ids):
            self.datasets[speaker_id] = self.read_data_set(speaker_file)

    def get_dataset(self, speaker_id) -> tf.data.TFRecordDataset:
        """Getter for a specific dataset."""
        return self.datasets[speaker_id]

    def get_datasets(self) -> Dict:
        "Getter for all dartasets."
        return self.datasets
