import torchaudio
import torch_audiomentations 
from audiomentations import Compose
import importlib
import torch


def create_union_augmentation(module_to_import, class_to_import, class_properties):

  augmentation_module = importlib.import_module(module_to_import)
  augmentation_class = getattr(augmentation_module, class_to_import)
  augmenter = augmentation_class(**class_properties)

  
  apply_augmentation = Compose(
      transforms=[
          augmenter
      ]
  )
  return apply_augmentation

