''' THIS SCRIPT IS FOR CALCULATING MCD, PESQ, AND SECS METRICS FOR EVALUATION OF AUDIO FILES

    Replace the paths with the paths to the true and generated audio files
    and the paths to the report files for MCD, PESQ, and SECS
    
    true_audio_folder: Path to the folder containing the true audio files
    generated_audio_folder: Path to the folder containing the generated audio files
    report_file_path_mcd: Path to the report file for MCD
    report_file_path_pesq: Path to the report file for PESQ
    report_file_path_cosine: Path to the report file for SECS
    
    The script will calculate the MCD, PESQ, and SECS metrics and write the results to the respective report files
    
    If the generated audio files have the same name as the true audio files, the script will automatically pair them
    based on the file names. If the naming convention is different, you may need to modify the script accordingly.

    '''

import torch
import numpy as np
from torchmetrics.audio import PerceptualEvaluationSpeechQuality
import librosa
from pymcd.mcd import Calculate_MCD
import os
from huggingface_hub import hf_hub_download
import scipy.spatial
import torch.nn.functional as F

###################################################
##### Calculate Mel Cepstral Distortion (MCD) #####
###################################################

def calculate_mcd(true_dir, generated_dir, report_file):
    mcd_values = []
    mcd_toolbox = Calculate_MCD(MCD_mode="dtw")
    with open(report_file, 'w') as report:
        for true_file in os.listdir(true_dir):
            if true_file.endswith(".wav"):
                # Construct corresponding generated file name
                generated_file = true_file.replace(".wav", "_generated.wav") # Assuming the generated files have the same name as the true files
                # Construct full paths
                true_file_path = os.path.join(true_dir, true_file)
                generated_file_path = os.path.join(generated_dir, generated_file)
                # Check if the corresponding generated file exists
                if os.path.exists(generated_file_path):
                    # Calculate MCD and save to the report
                    mcd_value = mcd_toolbox.calculate_mcd(true_file_path, generated_file_path)
                    mcd_values.append(mcd_value)
                    report.write(f'MCD for {true_file} and {generated_file}: {mcd_value}\n')

        # Calculate mean and std
        mean_mcd = np.mean(mcd_values)
        std_mcd = np.std(mcd_values)
        # Write the mean and std to the report
        report.write(f'\nMean MCD: {mean_mcd}\n')
        report.write(f'Standard Deviation of MCD: {std_mcd}\n')

    return mean_mcd, std_mcd

####################################################################
##### Calculate Perceptual Evaluation of Speech Quality (PESQ) #####
####################################################################

def load_and_resample_wav(file_path, target_fs=16000):
    data, fs = librosa.load(file_path, sr=None)
    if fs != target_fs:
        data = librosa.resample(data, orig_sr=fs, target_sr=target_fs)
    return torch.tensor(data, dtype=torch.float32), target_fs

def calculate_pesq(true_dir, generated_dir, report_file, target_fs=16000):
    pesq_values = []
    pesq_metric = PerceptualEvaluationSpeechQuality(fs=target_fs, mode='wb')

    with open(report_file, 'w') as report:
        # Iterate over files in the true audio directory
        for true_file in os.listdir(true_dir):
            if true_file.endswith(".wav"):
                # Construct corresponding generated file name
                generated_file = true_file.replace(".wav", "_generated.wav") # Assuming the generated files have the same name as the true files

                # Construct full paths
                true_file_path = os.path.join(true_dir, true_file)
                generated_file_path = os.path.join(generated_dir, generated_file)

                # Check if the corresponding generated file exists
                if os.path.exists(generated_file_path):
                    # Load and resample wav files
                    original_audio, _ = load_and_resample_wav(true_file_path, target_fs)
                    generated_audio, _ = load_and_resample_wav(generated_file_path, target_fs)

                    # Ensure both audio files have the same length
                    min_length = min(original_audio.size(0), generated_audio.size(0))
                    original_audio = original_audio[:min_length]
                    generated_audio = generated_audio[:min_length]

                    # Compute the PESQ score
                    pesq_score = pesq_metric(generated_audio, original_audio)
                    pesq_values.append(pesq_score.item())
                    # Write the individual PESQ value to the report
                    report.write(f'PESQ for {true_file} and {generated_file}: {pesq_score.item()}\n')

        # Calculate mean and std
        mean_pesq = np.mean(pesq_values)
        std_pesq = np.std(pesq_values)

        # Write the mean and std to the report
        report.write(f'\nMean PESQ: {mean_pesq}\n')
        report.write(f'Standard Deviation of PESQ: {std_pesq}\n')

    return mean_pesq, std_pesq


##############################################################
##### Calculate Speaker Encoder Cosine Similarity (SECS) #####
##############################################################

# Download model file and load it
model_file = hf_hub_download(repo_id='Jenthe/ECAPA2', filename='ecapa2.pt', cache_dir=None)
ecapa2 = torch.jit.load(model_file, map_location='cuda')

# Function to load and resample audio
def load_and_resample(file_path, target_sr=16000):
    audio, sr = librosa.load(file_path, sr=None)
    return librosa.resample(audio, orig_sr=sr, target_sr=target_sr)

# Function to calculate cosine similarity
def cosine_similarity_gpu(embedding1, embedding2):
    embedding1 = F.normalize(embedding1, p=2, dim=1)
    embedding2 = F.normalize(embedding2, p=2, dim=1)
    return torch.mm(embedding1, embedding2.transpose(0, 1)).item()

# Function to calculate cosine similarity for a set of audio files and generate a report
def calculate_cosine_similarity(true_audio_folder, generated_audio_folder, report_file_path):
    results = []
    for true_file in os.listdir(true_audio_folder):
        if true_file.endswith('.wav'):
            # Load and process true audio
            true_audio_path = os.path.join(true_audio_folder, true_file)
            audio_t = load_and_resample(true_audio_path)
            audio_t = torch.tensor(audio_t).unsqueeze(0).to('cuda')
            embedding_t = ecapa2(audio_t)

            # Construct corresponding generated audio file path
            generated_file = true_file.replace('.wav', '_generated.wav') # Assuming the generated files have the same name as the true files
            generated_audio_path = os.path.join(generated_audio_folder, generated_file)
            
            if os.path.exists(generated_audio_path):
                # Load and process generated audio
                audio_g = load_and_resample(generated_audio_path)
                audio_g = torch.tensor(audio_g).unsqueeze(0).to('cuda')
                embedding_g = ecapa2(audio_g)

                # Calculate cosine similarity
                similarity_score = cosine_similarity_gpu(embedding_t, embedding_g)
                results.append((true_file, generated_file, similarity_score))
            else:
                results.append((true_file, generated_file, None))

    similarity_scores = [score for _, _, score in results if score is not None]
    mean_similarity = np.mean(similarity_scores)
    std_similarity = np.std(similarity_scores)

    with open(report_file_path, 'w') as f:
        for true_file, generated_file, score in results:
            f.write(f'SECS for {true_file} vs {generated_file}: {score}\n')
        f.write(f'\nMean Similarity: {mean_similarity}\n')
        f.write(f'Standard Deviation of Similarity: {std_similarity}\n')

    return mean_similarity, std_similarity


###################################################################
##### Metric Calculation and Report Generation for Evaluation #####
###################################################################

true_audio_folder = "true"
generated_audio_folder = "gen"
report_file_path_mcd = "results_MCD.txt"
report_file_path_pesq = "results_PESQ.txt"
report_file_path_cosine = "results_SECS.txt"
mean_mcd, std_mcd = calculate_mcd(true_audio_folder, generated_audio_folder, report_file_path_mcd)
mean_pesq, std_pesq = calculate_pesq(true_audio_folder, generated_audio_folder, report_file_path_pesq)
mean_cosine, std_cosine = calculate_cosine_similarity(true_audio_folder, generated_audio_folder, report_file_path_cosine)
print(f'Mean MCD: {mean_mcd}, Std MCD: {std_mcd}')
print(f'Mean PESQ: {mean_pesq}, Std PESQ: {std_pesq}')
print(f'Mean SECS: {mean_cosine}, Std SECS: {std_cosine}')