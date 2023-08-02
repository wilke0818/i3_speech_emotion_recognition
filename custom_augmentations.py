from audiomentations.core.transforms_interface import BaseWaveformTransform
import sys
import numpy as np
import torch

class FrequencyMask(BaseWaveformTransform):
  supports_multichannel = True
  def __init__(
    self,
    freq_mask_param: int,
    p: float = 1.0
  ):
    super().__init__(p)
    assert freq_mask_param > 0
    
    self.freq_mask_max_length = freq_mask_param

  def apply(self, samples: np.ndarray, sample_rate: int):
    #print("in here")
    try:
      import torchaudio
    except ImportError:
      print(
        (
                    "Failed to import torchaudio. Maybe it is not installed? "
                    "To install torchaudio,"
                    " do `pip install torchaudio`"
        ), file=sys.stderr,
      )
      raise
    samples = torch.from_numpy(samples)
    #print(samples)
    spectrogram = torchaudio.transforms.Spectrogram()
    waveform = torchaudio.transforms.InverseSpectrogram()
    masking = torchaudio.transforms.FrequencyMasking(freq_mask_param=self.freq_mask_max_length)
    original = spectrogram(samples)
    #print(original)
    masked_spec = masking(original)
    #print(masked_spec)
    masked_spec = masked_spec.type(torch.cfloat)
    masked_wav = waveform(masked_spec)
    #print(masked_wav)
    return masked_wav


def frequency_mask(samples, sampling_rate):
  print("Fuck")
