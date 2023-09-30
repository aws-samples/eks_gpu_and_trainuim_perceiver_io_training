import os

import torch
from torch import Tensor
import torch.distributed as dist
from torchvision.datasets import Kinetics
from torchvision.transforms import Resize

from typing import Tuple


#_ROOT = "/dataset"
_ROOT = os.environ.get('DATASET_DIR')
_TRAIN_METADATA_FILE = "train_metadata.th"
_TRAIN_METADATA_FILE_SAMPLED = "train_metadata_sampled.th"
_VAL_METADATA_FILE = "val_metadata.th"
_VAL_METADATA_FILE_SAMPLED = "val_metadata_sampled.th"

_FRAMES_PER_CLIP = 16
_AUDIO_INPUT_LENGTH = 16*1920
_IMAGE_INPUT_SHAPE = (224, 224)
_NUM_CLASSES = 700

def is_dist_avail_and_initialized():
  if not dist.is_available():
    return False
  if not dist.is_initialized():
    return False
  return True

def get_rank():
  if not is_dist_avail_and_initialized():
    return 0
  return dist.get_rank()

def _get_cache_path(filepath):
    import hashlib

    h = hashlib.sha1(filepath.encode()).hexdigest()
    cache_path = os.path.join("~", ".torch", "vision", "datasets", "kinetics", h[:10] + ".pt")
    cache_path = os.path.expanduser(cache_path)
    print(cache_path)
    return cache_path

class FlattenAndZeroPadAudio(torch.nn.Module):

  def __init__(self, desired_length = _AUDIO_INPUT_LENGTH):
    super().__init__()
    self.desired_length = desired_length

  def forward(self, audio):
    audio = audio.reshape(-1)
    s = audio.shape[0]
    if s < self.desired_length:
      audio = torch.concat([audio, torch.zeros(self.desired_length-s, dtype=audio.dtype)])

    # TODO (@jmj): some of the audio is longer than 30720 elements...need to investigate why
    if s > self.desired_length:
      audio = audio[..., :self.desired_length]
    return audio

class PreprocessedKinetics(Kinetics):

  def __init__(
      self, 
      *args, 
      audio_input_length=_AUDIO_INPUT_LENGTH,
      image_input_shape=_IMAGE_INPUT_SHAPE,
      **kwargs
    ):
    super().__init__(*args, **kwargs)

    self.audio_transformer = FlattenAndZeroPadAudio(audio_input_length)
    self.video_transformer = Resize(image_input_shape)

  def __getitem__(self, idx: int) -> Tuple[Tensor, Tensor, int]:
    video, audio, label = super().__getitem__(idx)
    video = self.video_transformer(video) / 256.
    audio = self.audio_transformer(audio)
    #print(f"before torch.nn.functional.one_hot, label:{label},num_classes:{self.num_classes}")
    label = torch.nn.functional.one_hot(torch.tensor(label), num_classes=int(self.num_classes)).float()
    output = {
      "image": video,
      "audio": audio,
      "label": label,
    }
    #print(f"in PreprocessedKinetics: one_hot output:{output}")
    return output 


def build_dataset(split, root, metadata_filename, num_workers, debug=False, cache_dataset=True):
  metadata = None
  metadata_path = os.path.join(root, metadata_filename)
  if os.path.isfile(metadata_path):
    with open(metadata_path, "rb") as f:
      metadata = torch.load(f)

  data_dir = os.path.join(root, split)
  cache_path = _get_cache_path(data_dir)

  if cache_dataset and os.path.exists(cache_path):
    print(f"Loading dataset_train from {cache_path}")
    dataset, _ = torch.load(cache_path)
  else:
    print(f"No dataset_train from {cache_path}, going to create dataset from PreprocessedKinetics")
    dataset = PreprocessedKinetics(
      root=root,
      frames_per_clip=_FRAMES_PER_CLIP,
      num_classes=str(_NUM_CLASSES),
      num_workers=num_workers,
      split=split,
      download=False,
      _precomputed_metadata=metadata
    )

    if cache_dataset:
      print(f"Saving {split} dataset to {cache_path}")
      os.makedirs(os.path.dirname(cache_path), exist_ok=True)
      if get_rank() == 0:
        torch.save((dataset, data_dir), cache_path)

  return dataset

def build_train_dataset(num_workers=1, root=_ROOT, debug=False, cache_dataset=True):
  if debug:
    metadata_filename = _TRAIN_METADATA_FILE_SAMPLED
  else:
    metadata_filename = _TRAIN_METADATA_FILE
  print(f"build_train_dataset with metadata_filename:{metadata_filename}, num_workers:{num_workers}, root:{root}, debug:{debug}, cache_dataset:{cache_dataset}")
  return build_dataset("train", root, metadata_filename, num_workers, debug, cache_dataset)

def build_val_dataset(num_workers=1, root=_ROOT, debug=False, cache_dataset=True):
  if debug:
    metadata_filename = _VAL_METADATA_FILE_SAMPLED
  else:
    metadata_filename = _VAL_METADATA_FILE
  print(f"build_val_dataset with metadata_filename:{metadata_filename}, num_workers:{num_workers}, root:{root}, debug:{debug}, cache_dataset:{cache_dataset}")
  return build_dataset("val", root, metadata_filename, num_workers, debug, cache_dataset)
