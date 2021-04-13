import yaml

from pathlib import Path
from data_processing.DataProcessing import DataWriter, DataReader

def main(config):

    # Convert dataset and write the processed files.
    bucket_name = "DataSet"  # Name of how bucket is mounted
    datasets = ["dev-clean"]
    writer = DataWriter(bucket_name, datasets, config)
    # writer.process_datasets()

    # Read dataset.
    base_path = config["base_read_path"]
    dataset = "dev-clean"
    assert dataset in config["dataset"]
    reader = DataReader(base_path, dataset)
    # reader.read_data_set()


if __name__ == "__main__":
    config = yaml.load(
        Path("config.yml").read_text(), Loader=yaml.SafeLoader
    )

    main(config)
