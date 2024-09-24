import os
from pydub import AudioSegment

"""
This script converts all .wav files in a folder to mono and changes their sample rate to 22050 Hz.
The input folder should contain only .wav files.
The output folder will be created if it doesn't exist.
Change the sample rate by modifying the target_sample_rate variable as needed.
Replace the input_folder and output_folder variables with the paths to the input and output folders.

Last modified: 24/06/2024
"""

def change_sample_rate_and_convert_to_mono(input_folder, output_folder, target_sample_rate=22050):
    # Create the output folder if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)

    # Iterate over all files in the input folder
    for filename in os.listdir(input_folder):
        if filename.endswith(".wav"):
            # Construct full file path
            input_file_path = os.path.join(input_folder, filename)
            output_file_path = os.path.join(output_folder, filename)

            # Load the audio file
            audio = AudioSegment.from_wav(input_file_path)

            # Set the frame rate and convert to mono
            audio = audio.set_frame_rate(target_sample_rate).set_channels(1)

            # Export the modified audio file
            audio.export(output_file_path, format="wav")

            print(f"Processed {filename}")


# Replace input and output folder paths as needed
input_folder = 'path_to_input_folder'
output_folder = 'path_to_output_folder'
change_sample_rate_and_convert_to_mono(input_folder, output_folder)