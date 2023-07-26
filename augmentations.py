import torchaudio
import torch_audiomentations 
from torch_audiomentations import Compose
import importlib
import torch


def create_union_augmentation(class_to_import, class_properties):

  augmentation_class = getattr(torch_audiomentations, class_to_import)
  augmenter = augmentation_class(**class_properties)

  apply_augmentation = Compose(
      transforms=[
          augmenter
      ],
      output_type="dict"
  )
  return apply_augmentation

