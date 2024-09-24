import matplotlib.pyplot as plt
import IPython.display as ipd

import os
import json
import math
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader

import commons
import utils
from data_utils import TextAudioLoader, TextAudioCollate, TextAudioSpeakerLoader, TextAudioSpeakerCollate
from models import SynthesizerTrn
from text.symbols import symbols
from text import text_to_sequence

from scipy.io.wavfile import write
import soundfile as sf

def get_text(text, hps):
    text_norm = text_to_sequence(text, hps.data.text_cleaners)
    if hps.data.add_blank:
        text_norm = commons.intersperse(text_norm, 0)
    text_norm = torch.LongTensor(text_norm)
    return text_norm

### REPLACE WITH YOUR OWN PATH ###

hps = utils.get_hparams_from_file("./configs/vctk_base.json") # Load the hyperparameters
output_folder = './gen_audios/wallon_mspho_110624/male_fab_54k' # Output folder
speaker_id = 0 # Speaker ID 

net_g = SynthesizerTrn(
    len(symbols),
    hps.data.filter_length // 2 + 1,
    hps.train.segment_size // hps.data.hop_length,
    n_speakers=hps.data.n_speakers,
    **hps.model).cuda()
_ = net_g.eval()

### REPLACE WITH YOUR OWN PATH TO THE MODEL ###

_ = utils.load_checkpoint("./logs/wallon_mspho_110624/G_54000.pth", net_g, None) # Load the generator model

### REPLACE WITH YOUR OWN TEXT ###

stn_tst = get_text("Li bijhe et l’ solea estént ki s’ margayént po sawè kî çki, des deus, esteut l’ pus foirt. Mins ç’ côp la, la k’ i veyèt on tchminåd k' arivéve pyim piam, dins on bea noû tchôd paltot. Ladsu, i s' metèt d' acoird po çci: li ci ki parvénreut l' prumî a lyi haper s' mantea, ça sreut lu ki sreut rwaitî come li pus foirt. Adon, la k' li bijhe si mete a shofler tant k' ele pout. Mins non pus, pus çk' ele shofléve, pus çki l' roteu s' racafûléve dins s' grand paltot. Ça fwait k' elle a laixhî ouve. A ç' moumint la, la ki l' solea s' mete a lure come cwand i lût des cwate costés. Et c' est ki: après ene hapêye, la ki l' pormoenneu a stî reschandi, et il a tiré s' mantea. Ça fwait k' li bijhe a bén dvou rconoxhe ki l' solea esteut l' pus foirt di zels deus.", hps)
with torch.no_grad():
    x_tst = stn_tst.cuda().unsqueeze(0)
    x_tst_lengths = torch.LongTensor([stn_tst.size(0)]).cuda()
    sid = torch.LongTensor([speaker_id]).cuda()
    audio = net_g.infer(x_tst, x_tst_lengths, sid=sid, noise_scale=.667, noise_scale_w=0.8, length_scale=1)[0][0,0].data.cpu().float().numpy()
output_file_path = f"{output_folder}/transinne1.wav"
sf.write(output_file_path, audio, hps.data.sampling_rate)



