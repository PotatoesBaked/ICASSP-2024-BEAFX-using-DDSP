import lightning.pytorch as pl
import torch
from audio_effects.base import FXBase
import torch.nn.functional as F
from audio_effects.dsp.reverb import * 

def reverb(
        x:torch.Tensor,
        decay_time:torch.Tensor,
        dry_wet:torch.Tensor = .5,
        IR_length:int=None,
        samplerate:int = 44100
):
    device = x.device
    if IR_length==None:
        IR_length=3*decay_time*samplerate
    
    noise = torch.randn((2, IR_length), device=device)
    time = torch.arange(IR_length, device=device)/samplerate

    IR = torch.zeros((x.size(0), 2, IR_length), device=device)
    IR = noise*torch.pow(10, 3*time.view(1,1,-1)/decay_time.view(-1,1,1))

    IR = IR.mean(dim=0, keepdim=True)
    IR = IR.mean(dim=1, keepdim=True)

    print(f'{x.size()} et {IR.size()}')
    out = torch.nn.functional.conv1d(x, IR,padding=22050)
    out = out[:, :, :-1]
    print(f'out : {out.size()}')
    return out

#class SimpleReverb(pl.LightningModule):

class SimpleReverb(FXBase):
    def __init__(
            self,
            noise_length:int = 88200,
            samplerate:int = 44100,
            IR_length:int = 88200
    ):
        super(FXBase,self).__init__()
        self.num_controls = 2
        self.controls_ranges = torch.Tensor([[0.1, 3.0], [0.0, 1.0]])
        self.controls_names = ['decay_time', 'dry_wet']
        self.save_hyperparameters()   
 

        if IR_length==None:
            self.IR_length = 2*samplerate
        else : self.IR_length = IR_length
        
        self.noise = torch.rand((2,2*self.hparams.samplerate))
        self.time = torch.arange(IR_length)/self.hparams.samplerate
        
    def forward(self,x,decay_time,dry_wet=0.5):
        """
        ir = self.noise*torch.pow(10, -decay_time/self.hparams.samplerate*1000)


        out = reverb(x,decay_time,dry_wet,self.IR_length,self.hparams.samplerate)

        return out
        """
        out = reverb(x,decay_time,dry_wet,IR_length=self.IR_length,samplerate=self.hparams.samplerate)
        return out
        
    def _apply(self, fn):
        self.noise = fn(self.noise)
        self.time = fn(self.time)