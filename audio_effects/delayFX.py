import torch
from audio_effects.delay import *
from audio_effects.base import FXBase

class DelayFX(FXBase):
    def __init__(self, samplerate=44100):
        super(FXBase, self).__init__()
        self.num_controls = 3 
        self.controls_names = ["delay_time_sec", "feedback", "dry_wet"]
        self.controls_ranges = torch.tensor([[0.1, 2.0],[0.0, 1.0],[0.0, 1.0]])
        self.effect_name = "delay"
        self.samplerate = samplerate

    def process(self, x, p):
        """
        Applique l'effet de délai avec les paramètres dénormalisés `p`.

        Args:
            x (torch.Tensor): Signal d'entrée.
            p (torch.Tensor): Paramètres dénormalisés.
        Returns:
            torch.Tensor: Signal traité.
        """
        delay_time_sec = p[:, 0].item()  # Temps de délai en secondes
        feedback = p[:, 1].item()       # Rétroaction
        dry_wet = p[:, 2].item()        # Ratio dry/wet

        # Appliquer l'effet de délai
        return fun_delay(x, delay_time_sec, feedback, dry_wet, self.samplerate)
