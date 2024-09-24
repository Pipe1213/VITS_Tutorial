# VITS: Conditional Variational Autoencoder with Adversarial Learning for End-to-End Text-to-Speech

## 1. Clone repository and dependencies
Clone this repository onto your system:

``git clone https://github.com/Pipe1213/VITS_Tutorial.git``

## 2. Enviroment setup

Create a new environment with Python version 3.6, then install all dependencies from the `requirements.txt` file. Note that some dependencies are available via `conda install`, while others can only be installed using `pip`. Therefore, I recommend installing them one by one.

Additionally, it is advised to install the appropriate CUDA driver for your system. You can find the suitable driver at the following link: https://pytorch.org/get-started/locally/

## 3. Monotonic Alignment Search
Build Monotonic Alignment Search module:

```sh
# Cython-version Monotonoic Alignment Search

cd monotonic_align
python setup.py build_ext --inplace

```

## 4. Preprocess Dataset
Download datasets:

1. Download and extract the LJ Speech dataset, then rename or create a symbolic link to the dataset folder: 
    `ln -s /path/to/LJSpeech-1.1/wavs DUMMY1`
   
2. For mult-speaker setting, download and extract the VCTK dataset, and downsample wav files to 22050 Hz. Then rename or create a symbolic link to the dataset folder: 
    `ln -s /path/to/VCTK-Corpus/downsampled_wavs DUMMY2`
    
You can also use your own dataset, but it must have the same format as the ones mentioned above. Below, I will provide more details on how to prepare your dataset, along with some tools.

If your dataset is already organized in the required format (LJ Speech for single speaker and VCTK for multi-speaker), use the following commands to preprocess the text files containing the different subsets (train, validation, test).

```sh
# Preprocessing (g2p) for your own datasets. Preprocessed phonemes for LJ Speech and VCTK have been already provided.

# LJ Speech
python preprocess.py --text_index 1 --filelists filelists/ljs_audio_text_train_filelist.txt filelists/ljs_audio_text_val_filelist.txt filelists/ljs_audio_text_test_filelist.txt

# VCTK
python preprocess.py --text_index 2 --filelists filelists/vctk_audio_sid_text_train_filelist.txt filelists/vctk_audio_sid_text_val_filelist.txt filelists/vctk_audio_sid_text_test_filelist.txt
```

## 5. Training
Run the following command depending on the type of training to be performed:

```sh
# LJ Speech (single speaker)
python train.py -c configs/ljs_base.json -m ljs_base

# VCTK (multi-speaker)
python train_ms.py -c configs/vctk_base.json -m vctk_base
```

# Dataset Formatting



