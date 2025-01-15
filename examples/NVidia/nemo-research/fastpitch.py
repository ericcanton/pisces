import random
import argparse
import pickle

import numpy as np
from matplotlib import pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from torch.utils.data import Dataset, DataLoader
import nemo
from nemo.core import NeuralModule
from nemo.core.config import hydra_runner
from nemo.collections.tts.models import FastPitchModel, SpectrogramEnhancerModel

fastpitch = FastPitchModel.from_pretrained(model_name="tts_en_fastpitch")
# enhancer = SpectrogramEnhancerModel.from_pretrained(model_name="tts_en_spectrogram_enhancer_for_asr_finetuning")

language = 'en'


def test_inference(pretrained_model, text):
    model = pretrained_model
    parsed_text = model.parse(text)

    print("@@@@ parsed_text dtype:", parsed_text.dtype)
    print("@@@@ parsed_text shape:", parsed_text.shape)


    # Multi-Speaker
    speaker_id = None
    reference_spec = None
    reference_spec_lens = None

    if hasattr(model.fastpitch, 'speaker_emb'):
        speaker_id = 0

    if hasattr(model.fastpitch, 'speaker_encoder'):
        if hasattr(model.fastpitch.speaker_encoder, 'lookup_module'):
            speaker_id = 0
        if hasattr(model.fastpitch.speaker_encoder, 'gst_module'):
            bs, lens, t_spec = parsed_text.shape[0], random.randint(50, 100), model.cfg.n_mel_channels
            reference_spec = torch.rand(bs, lens, t_spec)
            reference_spec_lens = torch.tensor([lens]).long().expand(bs)

    # convert to float
    parsed_text = parsed_text.float()
    print(">>>>>", type(parsed_text.shape[1]))
    
    cos_like_parsed = np.cos(np.linspace(0, 4 * np.pi, parsed_text.shape[1])).reshape(1, -1)
    parsed_text = parsed_text * 0.1 + torch.Tensor(cos_like_parsed).to(parsed_text.device)

    # return to int64
    parsed_text = parsed_text.long()

    X = model.generate_spectrogram(
        tokens=parsed_text, speaker=speaker_id, reference_spec=reference_spec, reference_spec_lens=reference_spec_lens
    )
    print("Spectrogram has shape:", X.shape)

    # now save X to disk
    np_X = X.detach().cpu().numpy()
    np_parsed_text = parsed_text.detach().cpu().numpy()

    np.save(f"{language}_spectrogram.npy", np_X)

    fig, ax = plt.subplots(nrows=2, ncols=1, figsize=(20, 10))

    fig.tight_layout(pad=0.1)
    ax[0].plot(np_parsed_text[0])
    ax[0].set_title("Embedded Text")
    ax[0].set_xlim(0, len(np_parsed_text[0]))
    ax[0].set_xticks(range(len(np_parsed_text[0])))
    ax[1].imshow(np_X[0], aspect='auto')
    fig.savefig(f"{language}_spectrogram.png", dpi=200)

    # enhanced_X = enhancer.normalize_spectrograms(X, torch.tensor([X.shape[2]]).to('cuda:0'))
    # print(type(enhanced_X))
    # print(enhanced_X.shape)
    # np_enh_X = enhanced_X.detach().cpu().numpy()
    # plt.imshow(np_enh_X[0], aspect='auto')
    # plt.savefig(f"{language}_enhanced_spectrogram.png")


def int_psg_to_WLDM(psg_ints: str) -> str:
    def mapper(s) -> str:
        return {
            '0': 'W',
            '1': 'L',
            '2': 'L',
            '3': 'D',
            '4': 'D',
            '5': 'M',
        }.get(s, '')
    # convert psg_ints to WLDM
    # psg_ints = psg_ints.split('')
    psg_strs = [mapper(s) for s in psg_ints]
    return ''.join(psg_strs)



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='FastPitch Inference')
    parser.add_argument('text', type=str, help='Text to synthesize')
    args = parser.parse_args()

    psg_string = int_psg_to_WLDM(args.text)
    print("@@@@ Generating spectrogram for:\n@@@@\t", psg_string)
    test_inference(fastpitch, psg_string)