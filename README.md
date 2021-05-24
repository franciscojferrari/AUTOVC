# AUTOVC
This repository is a part of our master's course in Speech and Speaker Recognition (DT2119) taken in Spring 2021. In this project, we aim to implement a Voice conversor that only uses an autoencoder, being inspired by this paper. Voice conversion (VC) is a technique where the speaker characteristics of a source speaker are copied onto the speech contents of a target speaker, transforming the utterance of the target speaker such that it sounds like the source speaker.

## Installation

Use the package manager [pip](https://pip.pypa.io/en/stable/) to install the requirements.

```bash
pip install -r requirements.txt

```

## Usage
There are two main parts of the project: the training of the speaker encoder and the training of the whole AUTOVC. 
The training of the speaker encoder can be found in AUTOVC/speaker_encoder/experiments_speech.ipynb
The training of the whole AUTOVC can be found in AUTOVC/experiments/test_training.ipynb


## Contributing
Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.

## License
[MIT](https://choosealicense.com/licenses/mit/)
