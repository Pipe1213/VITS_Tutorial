# VITS: Conditional Variational Autoencoder with Adversarial Learning for End-to-End Text-to-Speech

## 1. Clone repository and dependencies
Clone this repository on your system:

``git clone https://github.com/Pipe1213/VITS_Tutorial.git``

## 2. Enviroment setup

Create a new enviroment with a python version = 3.6, then install all the dependencies from the requirements.txt file (some of then are available using conda install but other only using pip install so i recommend to install them one by one)

Additionally, it is recommended to install the appropriate driver for the cuda version of your system, which can be found at the following link: https://pytorch.org/get-started/locally/

## 3. Monotonic Alignment Search
Build Monotonic Alignment Search

```sh
# Cython-version Monotonoic Alignment Search

cd monotonic_align
python setup.py build_ext --inplace

```

## 4. Preprocess Dataset
Download datasets
    1. Download and extract the LJ Speech dataset, then rename or create a link to the dataset folder: `ln -s /path/to/LJSpeech-1.1/wavs DUMMY1`
    1. For mult-speaker setting, download and extract the VCTK dataset, and downsample wav files to 22050 Hz. Then rename or create a link to the dataset folder: `ln -s /path/to/VCTK-Corpus/downsampled_wavs DUMMY2`
    
You can use your own dataset but it must have the same format as the previous ones mentioned (below I will indicate in more detail how to prepare the dataset as well as provide some tools).

If the dataset is already organized in the way required by the model (The LJ Speech Dataset for single speaker and VCTK Dataset for multi-speaker) use the following command to preprocess the text files containing the different subsets (train, validation, test).

```sh
# Preprocessing (g2p) for your own datasets. Preprocessed phonemes for LJ Speech and VCTK have been already provided.

# LJ Speech
python preprocess.py --text_index 1 --filelists filelists/ljs_audio_text_train_filelist.txt filelists/ljs_audio_text_val_filelist.txt filelists/ljs_audio_text_test_filelist.txt

# VCTK
python preprocess.py --text_index 2 --filelists filelists/vctk_audio_sid_text_train_filelist.txt filelists/vctk_audio_sid_text_val_filelist.txt filelists/vctk_audio_sid_text_test_filelist.txt
```

## 5. Training
Run the following line depending on the type of training to be performed:

```sh
# LJ Speech (single speaker)
python train.py -c configs/ljs_base.json -m ljs_base

# VCTK (multi-speaker)
python train_ms.py -c configs/vctk_base.json -m vctk_base
```
