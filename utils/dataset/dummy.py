def prepend_prefix(prefix, input_file, output_file, enconding_input='utf-8', enconding_output='utf-8'):

    """ This function prepends a prefix to each line in the input file and writes the result to the output file."
    Modify this function to prepend 'DUMMY2/' or 'DUMMY1/' and
    also to add the speaker folder to the beginning of each line
    for example /DUMMY2/speaker1/audio_path.wav|transcription """

    with open(input_file, 'r', encoding=enconding_input) as infile, open(output_file, 'w', encoding=enconding_output) as outfile:
        for line in infile:
            # Prepend "DUMMY1/" to each line
            outfile.write(f'{prefix}{line}')

input_file = 'transcriptions/alignment.txt'
processed_file = 'alignment.txt'
prefix = 'DUMMY1/'
prepend_prefix(prefix, input_file, processed_file, enconding_input='iso-8859-1', enconding_output='utf-8')