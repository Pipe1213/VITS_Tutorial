# VITS: Conditional Variational Autoencoder with Adversarial Learning for End-to-End Text-to-Speech

This project provides a step-by-step guide on training VITS, a state-of-the-art model designed for high-quality speech synthesis. The goal of this project is to help others use VITS on their custom datasets and generate high-quality TTS outputs from scratch.

### Project Overview
- **Goal**: Train a VITS model from scratch using custom datasets, focusing on high-quality TTS generation.
- **Key Techniques**: Variational autoencoding, adversarial learning, and TTS.
- **Tools & Libraries**: Python, PyTorch, Jupyter, Hugging Face Transformers.

### Key Features
- **Dataset Setup**: Instructions on dataset preparation and preprocessing for VITS.
- **Model Training**: End-to-end steps on configuring and training the model.
- **Inference**: Guide to generating TTS outputs from text input using the trained model.

## Results
- Due to an upcoming publication, specific results are not yet publicly available.
- **Model Access**: You can try the trained models on my [Hugging Face](https://huggingface.co/Pipe1213).

### Usage
- Follow the tutorial below.


## Training from Scratch

### 1. Clone repository and dependencies
Clone this repository onto your system:

``git clone https://github.com/Pipe1213/VITS_Tutorial.git``

### 2. Enviroment setup

Create a new environment with Python version 3.6, then install all dependencies from the `requirements.txt` file. Note that some dependencies are available via `conda install`, while others can only be installed using `pip`. Therefore, I recommend installing them one by one.

Additionally, it is advised to install the appropriate CUDA driver for your system. You can find the suitable driver at the following link: https://pytorch.org/get-started/locally/

### 3. Monotonic Alignment Search
Build Monotonic Alignment Search module:

```sh
# Cython-version Monotonoic Alignment Search

cd monotonic_align
python setup.py build_ext --inplace

```

### 4. Preprocess Dataset
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
### 5. Symbols and Clearner files

In the `text` folder, there are two important files: `symbols.py` and `cleaners.py`.

- **`symbols.py`** contains all the alphabet characters, numbers, and symbols used in the dataset. Ensure that all characters present in your dataset are included in this file to avoid errors. Additionally, include any potential characters you plan to use during inference, as this cannot be changed after training. Failing to do so will require you to preprocess the input text before inference.

- **`cleaners.py`** provides default text preprocessing routines that are applied to the training, validation, and test data before training, as well as during inference. Modify these routines according to your specific requirements.

### 6. Configuration file

In the `configs` folder, there are example configuration files. Review them and modify anything necessary to adapt to your training setup.

Some of the most important parameters you should change include:

- **Training and validation file paths**
- **Text cleaners**
- **Number of speakers** (speaker IDs start counting from zero, but the number of speakers is counted normally. For example, with 2 speakers, the IDs are 0 and 1, and the number of speakers is 2)
- **Number of epochs**
- **Batch size** (adjust this according to your hardware)

### 7. Training
Run the following command depending on the type of training to be performed:

```sh
# LJ Speech (single speaker)
python train.py -c configs/ljs_base.json -m ljs_base

# VCTK (multi-speaker)
python train_ms.py -c configs/vctk_base.json -m vctk_base
```
### 8. Inference

Use the `inference.ipynb` file from the original repository, or the [multi-speaker inference](inference_example_ms.py) file provided, to test inference. To do this, you need to specify the following in the file:

- The model path
- The configuration file path
- The folder where the generated audio files will be stored
- The path to the trained model (after training, this is stored in the `logs` folder)

## Dataset Formatting

To utilize the VITS configuration, the dataset must be formatted in a specific way. The audio files should have a sampling frequency of 22 kHz and be in WAV mono format. Additionally, audio files need to be segmented by phrases for two main reasons: longer audio files can reduce alignment quality, and the model automatically excludes files that are too long. To accommodate longer files, you can modify `train.py` at line 70 to increase the threshold for what constitutes a "long" file.

### Transcripts Formatting

Transcripts should be organized into three text files (`train`, `test`, `validation`), each in TXT format. Each line in these files should start with the audio file path, followed by the speaker ID (for multi-speaker configuration), and the corresponding transcription.

#### For Single-Speaker:
`
DUMMY/path_to_your_audio_folder/audio_001.wav|transcription_audio_001 
DUMMY/path_to_your_audio_folder/audio_999.wav|transcription_audio_999
`
#### For Multi-Speaker:
`
DUMMY/path_to_your_audio_folder/audio_001.wav|speaker_id|transcription_audio_001 
DUMMY/path_to_your_audio_folder/audio_999.wav|speaker_id|transcription_audio_999
`

Examples of how the alignment file should look can be found in the `filelists` folder of this repository.

#### Remarks:

- Each audio file must correspond to a transcription, so ensure you split the audio correctly beforehand.
- All audio files should have a sample rate of **22050 Hz** for optimal results. If any audio is in stereo, the system will not work, so you must convert them to mono.
- Speaker IDs start from 0, so keep this in mind when assigning the IDs.

## Contributing
Contributions are welcome. Please open an issue to discuss your ideas.
