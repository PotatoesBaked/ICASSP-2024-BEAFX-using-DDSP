import torch
from audio_effects.base import FXBase

class SimpleReverb(FXBase):
    def __init__(
        self,
        samplerate: int = 44100,
        IR_length: int = None,
    ):
        # Appel du constructeur de la classe parente avant tout paramètre
        super(FXBase,self).__init__()

        # Définir les attributs
        self.controls_ranges = torch.tensor([[0.1, 3.0], [0.0, 1.0]])
        self.controls_names = ['decay_time', 'dry_wet']
        self.num_controls = 2

        self.samplerate = samplerate
        self.IR_length = IR_length if IR_length is not None else 2 * samplerate  # Default to 2 seconds

        # Appliquer la création du bruit après l'appel au constructeur parent
        self.noise = torch.nn.Parameter(torch.randn(2, self.IR_length))  # Learnable noise tensor

    def forward(self, x: torch.Tensor, decay_time: torch.Tensor, dry_wet: torch.Tensor):
        """
        x: Input audio tensor (batch_size, channels, samples)
        decay_time: Tensor of decay times (batch_size,)
        dry_wet: Tensor of dry/wet mix values (batch_size,)
        """
        batch_size, channels, _ = x.size()
        device = x.device

        # Time array for IR
        time = torch.arange(self.IR_length, device=device).float() / self.samplerate

        # Create impulse response (IR) for each batch
        IR = self.noise.to(device).unsqueeze(0) * torch.exp(
            -time.view(1, 1, -1) / decay_time.view(-1, 1, 1)
        )  # Shape: (batch_size, 2, IR_length)

        # Normalize IR
        IR = IR / IR.abs().max(dim=-1, keepdim=True).values

        # Convolve input with IR
        padding = self.IR_length // 2  # To ensure valid convolution output size
        convolved = torch.nn.functional.conv1d(
            x,
            IR.view(batch_size * 2, 1, self.IR_length),
            groups=channels,
            padding=padding,
        ).view(batch_size, channels, -1)

        # Apply dry/wet mix
        output = dry_wet.view(-1, 1, 1) * convolved + (1 - dry_wet.view(-1, 1, 1)) * x

        return output
