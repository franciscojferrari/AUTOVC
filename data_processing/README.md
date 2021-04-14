# Processing data
 In order to process the dataset the following variable need to be set in the `config.yml` file:
 - nfft: Size of FFT (fast fourier transform).
 - window: Window size.
 - stride: Size of hops between windows.
 - rate: Sample rate of the audio.
 - mels: Number of mel filterbanks. (Dimensionality of data)
 - fmin:  Minimum frequency.
 - fmax: Maximum frequency.
 
Besides, the dataset name should also be specified. For librispeech these are:
- dev-clean
- test-clean
- train-clean-100
- train-clean-360
- train-clean-500

The script can be run by executing:
`python process.py`

The created datasets will be located in `processed_datasets/*`, where a datset is created per speaker.

**For now only works with the librispeech dataset** 
 
 
# Reading data
In order to read the data, one should select a dataset (one of  the above specified datasets) and run the following 
command: `python process.py`

Running this will create a dictionary within the `DataReader` object that cotains a mapping between speaker id and its 
dataset.