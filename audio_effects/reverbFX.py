import torch
import numpy as np
from audio_effects.reverb import *
from audio_effects.base import FXBase
from audio_effects.dsp.reverb import *

class ReverbFX(FXBase):
    def __init__(self, samplerate=44100):
        super(FXBase,self).__init__()
        self.num_controls = 2 
        self.controls_names = ["decay_time", "dry_wet"]
        self.controls_ranges = torch.tensor([[0.1, 3.0],[0.0, 1.0]])
        self.effect_name = "colorless_reverb"
        self.samplerate = samplerate


    def process(self, x,p):
        decay_time = p[:, 0].item()
        dry_wet = p[:,1].item()

        x_numpy = x.cpu().detach().numpy().astype(np.float64).flatten()

        temp = colorlessReverb(x_numpy,decay_time,dry_wet,self.samplerate)

        out = torch.tensor(temp, dtype=x.dtype, device=x.device)

        return out

    def _apply(self, fn):
        self.noise = fn(self.noise)
        self.time = fn(self.time)