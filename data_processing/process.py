import yaml

from pathlib import Path
from data_processing.DataProcessing import DataWriter, DataReader


def main(config):

    # Convert dataset and write the processed files.
    bucket_name = config["bucket_name"]  # Name of how bucket is mounted
    datasets = ["dev-clean"]
    writer = DataWriter(bucket_name, datasets, config)
    # writer.process_datasets()

    # Read dataset.
    reader = DataReader(config)
    reader.load_datasets()


if __name__ == "__main__":
    config = yaml.load(Path("config.yml").read_text(), Loader=yaml.SafeLoader)

    main(config)
