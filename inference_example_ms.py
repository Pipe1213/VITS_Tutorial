from pathlib import Path

import torch
import soundfile as sf

import commons
import utils
from models import SynthesizerTrn
from text.symbols import symbols
from text import text_to_sequence


def get_text(text, hps):
    text_norm = text_to_sequence(text, hps.data.text_cleaners)
    if hps.data.add_blank:
        text_norm = commons.intersperse(text_norm, 0)
    text_norm = torch.LongTensor(text_norm)
    return text_norm


PATH = Path("~/Documents/LISN/DATA/TTS/VITS").expanduser()
CONFIG = Path("configs/vctk_base.json")

MODEL = PATH / "wa_graphemes_v3/G_394000.pth"
# OUTPUT = PATH / "gen_audios/wallon_mspho_110624/male_fab_54k/transinne1.wav"
OUTPUT = PATH / "gen_audios/test1.wav"


### REPLACE WITH YOUR OWN PATH ###

hps = utils.get_hparams_from_file(CONFIG)  # Load the hyperparameters
speaker_id = 0  # Speaker ID

net_g = SynthesizerTrn(
    len(symbols),
    hps.data.filter_length // 2 + 1,
    hps.train.segment_size // hps.data.hop_length,
    n_speakers=hps.data.n_speakers,
    **hps.model,
)
# ).cuda()

_ = net_g.eval()

### REPLACE WITH YOUR OWN PATH TO THE MODEL ###

_ = utils.load_checkpoint(MODEL, net_g, None)  # Load the generator model

### REPLACE WITH YOUR OWN TEXT ###

stn_tst = get_text(
    "Li bijhe et l’ solea estént ki s’ margayént po sawè kî çki, des deus, ",
    # "esteut l’ pus foirt. ",
    # "Mins ç’ côp la, la k’ i veyèt on tchminåd k' arivéve pyim piam, "
    # "dins on bea noû tchôd paltot. "
    # "Ladsu, i s' metèt d' acoird po çci: li ci ki parvénreut "
    # "l' prumî a lyi haper s' mantea, ça sreut lu ki sreut rwaitî "
    # "come li pus foirt. Adon, la k' li bijhe si mete a shofler tant "
    # "k' ele pout. Mins non pus, pus çk' ele shofléve, pus çki l' roteu "
    # "s' racafûléve dins s' grand paltot. Ça fwait k' elle a laixhî ouve. "
    # "A ç' moumint la, la ki l' solea s' mete a lure come cwand i lût des "
    # "cwate costés. Et c' est ki: après ene hapêye, la ki l' pormoenneu a stî "
    # "reschandi, et il a tiré s' mantea. Ça fwait k' li bijhe a bén dvou "
    # "rconoxhe ki l' solea esteut l' pus foirt di zels deus.",
    hps,
)
with torch.no_grad():
    # x_tst = stn_tst.cuda().unsqueeze(0)
    # x_tst_lengths = torch.LongTensor([stn_tst.size(0)]).cuda()
    # sid = torch.LongTensor([speaker_id]).cuda()
    x_tst = stn_tst.unsqueeze(0)
    x_tst_lengths = torch.LongTensor([stn_tst.size(0)])
    sid = torch.LongTensor([speaker_id])
    audio = (
        net_g.infer(
            x_tst,
            x_tst_lengths,
            sid=sid,
            noise_scale=0.667,
            noise_scale_w=0.8,
            length_scale=1,
        )[0][0, 0]
        .data.cpu()
        .float()
        .numpy()
    )
output_file_path = OUTPUT
sf.write(output_file_path, audio, hps.data.sampling_rate)
