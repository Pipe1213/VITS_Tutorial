import xml.etree.ElementTree as ET
import glob
from pydub import AudioSegment
import os
import argparse

""" THIS SCRIPT IS USED TO CREATE A DATASET FOR MONOTONIC ALIGNMENT SEARCH
FOLLOWING THE STRUCTURE OF LJSPEECH DATASET
THE SCRIPT PARSES .TRS FILES AND SPLITS THE AUDIO INTO SEGMENTS
AND CREATES TRANSCRIPTION FILES FOR EACH SEGMENT
IT ALSO CREATES AN ALIGNMENT FILE FOR MONOTONIC ALIGNMENT SEARCH

The script assumes that the complete audio files are in the same folder as the .trs files
The script will create 'wavs' and 'transcriptions' folders in the same directory as the script
The input audio files should be in .wav format
The script will create a .wav file for each segment and a .txt file for each transcription 
The encoding of the transcription files is iso-8859-1
The audio files are converted to mono and the sample rate is set to 22050

Example usage:
python dataset_creator.py /path_to_desired_folder
This will process all .trs files in the 'desire folder' and create the dataset

Last modified: 24/06/2024
"""


def create_transcription_file(output_path, text):
    corrected_text = correct_transcription_errors(text)
    with open(output_path, 'w', encoding='iso-8859-1') as text_file:
        text_file.write(corrected_text)
    print(f"Created transcription file {output_path}")

def correct_transcription_errors(text):
    """
    Corrects transcription errors in the text.
    Replaces '?' with "'" unless '?' is preceded by a space.
    """
    corrected_text = ""
    previous_char = None
    for char in text:
        if char == '?' and previous_char != ' ':
            corrected_text += "'"
        else:
            corrected_text += char
        previous_char = char
    return corrected_text

def parse_and_split_audio(trs_file_path, audio_folder, wav_output_folder, transcription_output_folder, alignment_data):
    audio_file_path = os.path.join(audio_folder, os.path.basename(trs_file_path).replace('.trs', '.wav'))
    audio = AudioSegment.from_wav(audio_file_path)

    tree = ET.parse(trs_file_path)
    root = tree.getroot()

    turn = root.find('.//Turn')
    if turn is not None:
        sync_elements = list(turn.findall('Sync'))
        for i, sync in enumerate(sync_elements):
            sentence_start_time = float(sync.attrib.get('time')) * 1000  # Convert to milliseconds

            if i + 1 < len(sync_elements):
                sentence_end_time = float(sync_elements[i + 1].attrib.get('time')) * 1000
            else:
                sentence_end_time = float(turn.attrib.get('endTime')) * 1000

            segment = audio[sentence_start_time:sentence_end_time]
            # Convert to mono and set the sample rate to 22050
            segment = segment.set_channels(1).set_frame_rate(22050)
            
            num_digits = 4  # Adjust depending on the number of segments
            segment_number = f"{i+1:0{num_digits}}"
            base_filename = f"{os.path.splitext(os.path.basename(audio_file_path))[0]}_{segment_number}"
            audio_output_path = os.path.join(wav_output_folder, f"{base_filename}.wav")

            segment.export(audio_output_path, format="wav")
            print(f"Exported {audio_output_path}")

            segment_text = ''
            is_collecting = False
            for elem in turn:
                if elem == sync:
                    is_collecting = True
                elif elem.tag == 'Sync' and is_collecting:
                    break
                if is_collecting and elem.tail:
                    segment_text += elem.tail.strip() + ' '

            transcription_output_path = os.path.join(transcription_output_folder, f"{base_filename}.txt")
            create_transcription_file(transcription_output_path, segment_text.strip())
            
            aligment_text = correct_transcription_errors(segment_text.strip())
            audio_file_relative_path = os.path.relpath(audio_output_path, start=wav_output_folder)
            alignment_data.append((audio_file_relative_path, aligment_text))

def generate_alignment_txt(output_folder, alignment_data):
    """
    Creates a .txt file for Monotonic Alignment Search.
    Each line contains the path of an audio file followed by the transcription.
    """
    alignment_file_path = os.path.join(output_folder, 'alignment.txt')
    with open(alignment_file_path, 'w') as file:
        for audio_path, transcription in alignment_data:
            line = f"{audio_path}|{transcription}\n"
            file.write(line)
    print(f"Created alignment file {alignment_file_path}")

def process_folder(folder_path):
    trs_files = glob.glob(f'{folder_path}/*.trs')
    script_dir = os.path.dirname(os.path.abspath(__file__))
    wav_output_folder = os.path.join(script_dir, 'wavs')
    transcription_output_folder = os.path.join(script_dir, 'transcriptions')
    alignment_data = []

    if not os.path.exists(wav_output_folder):
        os.makedirs(wav_output_folder)
    
    if not os.path.exists(transcription_output_folder):
        os.makedirs(transcription_output_folder)

    for trs_file in trs_files:
        print(f"Processing {trs_file}...")
        parse_and_split_audio(trs_file, folder_path, wav_output_folder, transcription_output_folder, alignment_data)

    generate_alignment_txt(transcription_output_folder, alignment_data)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process TRS files and split audio into segments.")
    parser.add_argument("folder_path", type=str, help="Path to the folder containing TRS files and complete audio files.")
    
    args = parser.parse_args()
    
    process_folder(args.folder_path)
