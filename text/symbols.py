""" from https://github.com/keithito/tacotron """

'''
Defines the set of symbols used in text input to the model.
'''
_punctuation = '"'
_full_alphabet = " !*&'()+,-./0123456789:;?ABCDEFGHIJKLMNOPQRSTUVWXYZ_abcdefghijklmnopqrstuvwxyz|«°»ÀÂÇÈÉÊÎÏÔÙÛàâæçèéêëîïôöùúûüœ–’“”…ÅåûPRSTVWYZ"

# Export all symbols:
symbols = [_punctuation] + list(_full_alphabet)


# Special symbol ids
SPACE_ID = symbols.index(" ")
