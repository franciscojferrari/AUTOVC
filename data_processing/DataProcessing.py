import matplotlib.pyplot as plt
import random
import torch

from data_processing.utils import *
from collections import OrderedDict
from pre_trained_models.speaker_embedder import SpeakerEmbedder


class DataWriter:
    def __init__(self, bucket_name, datasets, config):
        self.bucket_path = bucket_name
        self.datasets = datasets
        self.config = config
        self.mel_basis = None
        self.min_level = None
        self.b = None
        self.a = None
        self.C = None

        self.load_speaker_embedder()

    def process_datasets(self, verbose=False, train_split=0.8):
        """Process datasets"""

        self.mel_basis, self.min_level, self.b, self.a = get_filters(config=self.config)

        if self.config["dataset_tf"] == "librispeech":

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
                    processed_files = transform_files(
                        file_data_set, self.process_files_librispeech
                    )

                    record_file_name = f"{label}.tfrecords"
                    write_path = f"{self.bucket_path}/processed_datasets/{self.config['dataset_tf']}"
                    record_file = os.path.join(write_path, record_file_name)
                    n_samples = tf.data.experimental.cardinality(
                        processed_files
                    ).numpy()
                    with tf.io.TFRecordWriter(record_file) as writer:
                        for i, processed_file in processed_files.enumerate():

                            processed_file = raw_audio_to_spectrogram_np(processed_file,
                                                                         self.mel_basis,
                                                                         self.min_level,
                                                                         self.b,
                                                                         self.a)

                            subset = (
                                b"train" if i <= train_split * n_samples else b"test"
                            )
                            if verbose:
                                plt.figure(figsize=(15, 4))
                                data = tf.math.log(processed_file).numpy()
                                plt.imshow(data, aspect="auto")
                                plt.show()

                            # Don't process utterance that are shorter then
                            if processed_file.shape[0] < self.config["max_length"]:
                                print(f"{processed_file.shape[0]}")
                                continue

                            # Create the speaker embedding here
                            speaker_embedding = self.get_speaker_embedding(processed_file)

                            tf_example = spectrogram_example(
                                processed_file, label, subset, speaker_embedding
                            )
                            writer.write(tf_example.SerializeToString())
        else:
            dataset_path = os.path.join(
                self.config["bucket_name"], self.config["dataset"]["vctk"]
            )
            vctk_files = get_all_files(dataset_path, "00512")
            for vctk_file in vctk_files:

                dataset = tf.data.TFRecordDataset(vctk_file)
                parsed_dataset = dataset.map(self._parse_function)
                processed = parsed_dataset.map(self.process_files_vctk)

                write_path = (
                    f"{self.bucket_path}/processed_datasets/{self.config['dataset_tf']}"
                )
                file_nm = vctk_file.split(".")[-1]

                record_file = os.path.join(write_path, file_nm)

                with tf.io.TFRecordWriter(record_file) as writer:
                    for example in processed:
                        example["speech"] = raw_audio_to_spectrogram_np(example["speech"], self.mel_basis, self.min_level, self.b, self.a)

                        # Don't process utterance that are shorter then
                        if example["speech"].shape[0] < self.config["max_length"]:
                            print(f"{example['speech'].shape[0]}")
                            continue

                        # Create speaker embedding
                        example["speaker_embedding"] = self.get_speaker_embedding(example['speech'])
                        tf_example = spectrogram_example_vctk(example)
                        writer.write(tf_example.SerializeToString())

    def process_files_librispeech(self, file_path: str) -> tf.Tensor:
        """Load audio file from disk and perform processing step to get mel spectrogram"""
        audio_tensor = load_audio(file_path)

        tensor = tf.cast(audio_tensor, tf.float32) / 32768.0
        tensor = tensor[:, 0]

        return tensor

    def process_files_vctk(self, example):
        speech_tensor = example["speech"]

        speech_tensor = (
            tf.cast(tf.sparse.to_dense(speech_tensor), dtype=tf.float32) / 32768.0
        )

        example["speech"] = speech_tensor
        return example

    @staticmethod
    def _parse_function(example_proto):
        feature_description_vctk = {
            "id": tf.io.FixedLenFeature([], tf.string),
            "speaker": tf.io.FixedLenFeature([], tf.int64),
            "gender": tf.io.FixedLenFeature([], tf.int64),
            "accent": tf.io.FixedLenFeature([], tf.int64),
            "speech": tf.io.VarLenFeature(tf.int64),
            "speaker_embedding": tf.io.VarLenFeature(tf.float32),
        }
        return tf.io.parse_single_example(example_proto, feature_description_vctk)

    def get_speaker_embedding(self, mel_spectrogram: tf.Tensor) -> tf.Tensor:
        embeddings = []
        for _ in range(self.config["nr_sample"]):
            length_spectrogram = mel_spectrogram.shape[0]
            idx = random.randint(0, length_spectrogram - self.config["max_length"])
            mel_temp = np.array(mel_spectrogram[idx:idx + self.config["max_length"]])
            mel_temp = torch.from_numpy(np.expand_dims(mel_temp, axis=0))
            embedding = self.C(mel_temp)
            embeddings.append(embedding.detach().numpy())

        speaker_embedding = np.array(embeddings).mean(axis=0)
        return tf.convert_to_tensor(speaker_embedding)

    def load_speaker_embedder(self) -> None:
        C = SpeakerEmbedder(dim_input=80, dim_cell=768, dim_emb=256).eval()  # .cuda()
        c_checkpoint = torch.load('/content/DataSet/pre_trained_models/3000000-BL.ckpt',
                                  map_location=torch.device('cpu'))
        new_state_dict = OrderedDict()
        for key, val in c_checkpoint['model_b'].items():
            new_key = key[7:]
            new_state_dict[new_key] = val
        C.load_state_dict(new_state_dict)

        self.C = C


class DataReader:
    def __init__(self, config):
        self.speaker_files = []
        self.speaker_ids = []
        self.dataset_files = []
        self.config = config

        self.base_path = os.path.join(
            self.config["base_read_path"], self.config["dataset_tf"]
        )
        self.dataset = self.config[
            "subdataset"
        ]  # For librispeech these are dev-clean, test-clean, etc.
        self.datasets = {}

    def _parse_function(self, example_proto):
        # Parse the input tf.train.Example proto using the provided dictionary
        if self.config["dataset_tf"] == "librispeech":
            feature_description = {
                "label": tf.io.FixedLenFeature([], tf.int64),
                "mel_spectrogram": tf.io.RaggedFeature(tf.string),
                "subset": tf.io.FixedLenFeature([], tf.string),
                "speaker_embedding": tf.io.RaggedFeature(tf.string),
            }
        elif self.config["dataset_tf"] == "vctk":
            feature_description = {
                "id": tf.io.FixedLenFeature([], tf.string),
                "speaker": tf.io.FixedLenFeature([], tf.int64),
                "gender": tf.io.FixedLenFeature([], tf.int64),
                "accent": tf.io.FixedLenFeature([], tf.int64),
                "speech": tf.io.RaggedFeature(tf.string),
                "speaker_embedding": tf.io.RaggedFeature(tf.string),
            }
        else:
            raise NotImplemented
        return tf.io.parse_single_example(example_proto, feature_description)

    def read_data_set(self, tfrecord_path: str) -> tf.data.TFRecordDataset:
        """Read speaker dataset and parse it."""
        processed_dataset = tf.data.TFRecordDataset(tfrecord_path)
        parsed_dataset = processed_dataset.map(self._parse_function)
        return parsed_dataset

    def find_vctk_datasets(self):
        """Find all vctk datasets in dataset directory."""
        path = os.path.join(
            self.config["bucket_name"], self.base_path
        )
        for root, dirs, files in os.walk(path):
            for file in files:
                if file.endswith("00512"):
                    self.dataset_files.append(os.path.join(root, file))

    def find_speaker_datasets(self) -> None:
        """Find all speaker datasets in dataset directory."""
        path = os.path.join(
            self.config["bucket_name"], self.base_path
        )
        for root, dirs, files in os.walk(path):
            for file in files:
                if file.endswith(".tfrecords"):
                    self.speaker_files.append(os.path.join(root, file))
                    self.speaker_ids.append(file.split(".")[0])

    def find_data_sets(self) -> None:
        if self.config["dataset_tf"] == "librispeech":
            self.find_speaker_datasets()
        elif self.config["dataset_tf"] == "vctk":
            self.find_vctk_datasets()
        else:
            raise NotImplemented

    def load_datasets(self) -> None:
        """Load all speaker datasets into a dictionary."""
        self.find_data_sets()
        if self.config["dataset_tf"] == "librispeech":
            # Load the data for each speaker
            for speaker_file, speaker_id in zip(self.speaker_files, self.speaker_ids):
                self.datasets[speaker_id] = self.read_data_set(speaker_file)
        elif self.config["dataset_tf"] == "vctk":
            # Load the data per sub dataset
            for dataset_file in self.dataset_files:
                self.datasets[dataset_file] = self.read_data_set(dataset_file)
        else:
            raise NotImplemented

    def get_dataset(self, speaker_id) -> tf.data.TFRecordDataset:
        """Getter for a specific dataset."""
        if self.config["dataset_tf"] == "vctk":
            raise NotImplemented
        return self.datasets[speaker_id]

    def get_datasets(self) -> Dict:
        """Getter for all datasets."""
        return self.datasets
