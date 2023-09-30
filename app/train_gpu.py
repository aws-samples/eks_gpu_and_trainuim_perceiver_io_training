import argparse
import os
import pkg_resources
import sys
import logging

from lightning import Trainer
from lightning.pytorch.loggers import (
  TensorBoardLogger,
  WandbLogger,
)
import torch
torch.manual_seed(42)
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler
from transformers import PerceiverForMultimodalAutoencoding

from datasets.kinetics import build_train_dataset
from datasets.kinetics import build_val_dataset
from lit import (
  LitTransformersAutoencodingPerceiverIO, 
  MultimodalOutputSampler
)
from train_utils import load_config, ThroughputCallback

NUM_FRAMES = 16
NUM_CHANNELS = 3
IMAGE_SIZE = 224
AUDIO_SAMPLES_PER_FRAME = 1920
NUM_AUDIO_ELEMENTS = NUM_FRAMES * AUDIO_SAMPLES_PER_FRAME 
NUM_CLASSES = 700
NUM_OUTPUT_SAMPLES = 512

WANDB_DEFAULT_PROJECT = "Hugging Face Perceiver IO"

class SyntheticDataset(Dataset):

  def __len__(self):
    return 100000

  def __getitem__(self, idx):
    return {
      "image": torch.randn(NUM_FRAMES, NUM_CHANNELS, IMAGE_SIZE, IMAGE_SIZE),
      "audio": torch.randn(NUM_AUDIO_ELEMENTS),
      "label": torch.nn.functional.one_hot(torch.randint(low=0, high=NUM_CLASSES, size=()), num_classes=NUM_CLASSES).float(),
    }

def run(
    config_file_path,
    dataset="synthetic",
    dataset_dir="/home/ubuntu/dataset",
    batch_size=2, 
    batches_per_epoch=10,
    num_epochs=10,
    learning_rate=1e-2,
    wandb_enabled=False,
  ):
    
    if dataset == "synthetic":
      train_dataset = SyntheticDataset()
      valid_dataset = SyntheticDataset()
    elif dataset == "kinetics-small":
      train_dataset = build_train_dataset(num_workers=120, root=dataset_dir, debug=True)
      valid_dataset = build_val_dataset(num_workers=120, root=dataset_dir, debug=True)
      # train_dataset = valid_dataset
    elif dataset == "kinetics":
      train_dataset = build_train_dataset(num_workers=120, root=dataset_dir, debug=False)
      valid_dataset = build_val_dataset(num_workers=120, root=dataset_dir, debug=False)
      # train_dataset = valid_dataset
    else:
      raise NotImplementedError(f"unhandled dataset type: {dataset}")
    
    rank = int(os.environ['RANK'])
    world_size = int(os.environ['WORLD_SIZE'])

    # if os.environ.get("WORLD_SIZE"):
    #   world_size = int(os.environ['WORLD_SIZE'])
    # else:
    #   world_size = int(os.environ['GPU_NUM_DEVICES'])

    # rank = int(os.environ['RANK'])
    # train_sampler, valid_sampler = None, None
    # if world_size > 1:
    #   train_sampler = DistributedSampler(
    #     train_dataset,
    #     num_replicas=world_size,
    #     rank=rank,
    #     shuffle=True)
    #   valid_sampler = DistributedSampler(
    #     valid_dataset,
    #     num_replicas=world_size,
    #     rank=rank,
    #     shuffle=False)

    dataloader_train = DataLoader(
      train_dataset, 
      batch_size=batch_size, 
      num_workers=8, 
      prefetch_factor=8,
      pin_memory=True,
      shuffle=False,
    )
    dataloader_valid = DataLoader(
      valid_dataset, 
      batch_size=batch_size, 
      num_workers=8, 
      prefetch_factor=8,
      pin_memory=True,
    )

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    config = load_config(config_file_path)
    model = PerceiverForMultimodalAutoencoding(config)

    multimodal_output_sampler = MultimodalOutputSampler(
      image_size=NUM_FRAMES*IMAGE_SIZE*IMAGE_SIZE,
      audio_size=AUDIO_SAMPLES_PER_FRAME,
      audio_patch_size=config.samples_per_patch,
      num_samples=NUM_OUTPUT_SAMPLES
    )
    lit_model = LitTransformersAutoencodingPerceiverIO(
      model=model, 
      output_sampler=multimodal_output_sampler,
      learning_rate=learning_rate,
    )

    loggers = []
    if wandb_enabled and rank == 0:
      logger_tb = TensorBoardLogger(save_dir=os.getcwd(), version=1, name='lightning_logs')
      loggers.append(logger_tb)
      logger_wandb = WandbLogger()
      logger_wandb.log_hyperparams(dict(
          config_file_path=config_file_path,
          dataset=dataset,
          batch_size=batch_size,
          batches_per_epoch=batches_per_epoch,
          num_epochs=num_epochs,
          learning_rate=learning_rate,
        )
      )
      logger_wandb.log_hyperparams(config.to_dict())
      loggers.append(logger_wandb)

    trainer = Trainer(
      # accelerator="gpu",
      # devices=8, 
      # strategy="ddp",
      max_epochs=num_epochs,
      logger=loggers, 
      log_every_n_steps=20,
      limit_train_batches=batches_per_epoch, 
      limit_val_batches=batches_per_epoch,
      gradient_clip_val=10.,
      precision="bf16-mixed",
      callbacks=ThroughputCallback(batch_size=batch_size, world_size=world_size)
    )
    trainer.fit(lit_model, dataloader_train, dataloader_valid)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
      formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument(
      "-d",
      "--dataset",
      type=str,
      choices=["synthetic", "kinetics-small", "kinetics"],
      default="synthetic",
      help="Dataset type")
    parser.add_argument(
      "-r",
      "--dataset_dir",
      type=str,
      default="/home/ubuntu/dataset",
      help="Dataset directory")
    parser.add_argument(
      "-b",
      "--batch_size",
      type=int,
      default=6,
      help="Batch size")
    parser.add_argument(
      "-e",
      "--num_epochs",
      type=int,
      default=50,
      help="Number of training epochs")
    parser.add_argument(
      "-be",
      "--batches_per_epoch",
      type=int,
      default=10,
      help="Batch size")
    parser.add_argument(
      "-lr",
      "--learning_rate",
      type=float,
      default=0.0001,
      help="Learning rate")
    parser.add_argument(
      "-c",
      "--config_file_path",
      type=str,
      default="config/main.yaml",
      help="Model config file path")
    # parser.add_argument(
    #   "-ws",
    #   "--wandb_secret_name",
    #   type=str,
    #   default="jmaojones-wandb-key-gcp",
    #   help="Spookey secret name storing wandb API Key")
    parser.add_argument(
      "-wp",
      "--wandb_project",
      type=str,
      default=WANDB_DEFAULT_PROJECT,
      help="Wandb project name")

    config = vars(parser.parse_args())

    try:
      from snap.perceiver.util.wandb_util import set_env_vars
      set_env_vars(
        # secret_name=config.pop("wandb_secret_name"),
        project=config.pop("wandb_project"),
      )
      config["wandb_enabled"] = True

    except ModuleNotFoundError:
      # config.pop("wandb_secret_name"),
      config.pop("wandb_project"),
      config["wandb_enabled"] = False

    run(**config)
