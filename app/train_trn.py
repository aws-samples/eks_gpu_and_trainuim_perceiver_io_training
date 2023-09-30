import os
import sys
import time
from datetime import datetime
import numpy as np
import torch
torch.manual_seed(42)

import torch.distributed as dist
import torch_xla.distributed.xla_backend
dist.init_process_group('xla')
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torch_xla
import torch_xla.debug.metrics as met
import torch_xla.distributed.parallel_loader as pl
import torch_xla.utils.utils as xu
import torch_xla.core.xla_model as xm
import torch_xla.distributed.xla_multiprocessing as xmp

from transformers import PerceiverForMultimodalAutoencoding

from datasets.kinetics import build_train_dataset
from datasets.kinetics import build_val_dataset

from einops import rearrange

try:
    from utilities.reporting import Metric, post_metrics
except ImportError:
    Metric = post_metrics = lambda *args, **kwargs: None
    xm.master_print('Failed to import the Metrics API')

import train_utils

NUM_FRAMES = 16
NUM_CHANNELS = 3
IMAGE_SIZE = 224
AUDIO_SAMPLES_PER_FRAME = 1920
NUM_AUDIO_ELEMENTS = NUM_FRAMES * AUDIO_SAMPLES_PER_FRAME 
NUM_CLASSES = 700
NUM_OUTPUT_SAMPLES = 512

WANDB_DEFAULT_PROJECT = "Hugging Face Perceiver IO"

class MultimodalOutputSampler:
  """
  The Hugging Face implementation of perceiver does not allow per-element sampling.
  In other words, we sample indices that are retrieve for every row in the batch.
  There are a couple reasons why this is not ideal

    1) We cannot put sampling in the dataset, because the dataloader will batch the
       sampled indices, which will be different for every element.  This will hurt
       throughput.
    2) From a learning standpoint, it's better to sample different indices for each
       row in the batch.

  So we are implementing a sampler here that is specific to the Hugging Face perceiver,
  but it should not be used elsewhere.

  Also, the Hugging Face implementation decodes audio into patches, i.e. index size 1920
  instead of 16*1920, which means that when we sample data points, we sample rows of
  audio pts of length 16 from a index size of 1920, and then flatten it.  So if the sample
  size is 10, then we would end up with 16*10 = 160 audio output samples.  It's not clear
  from the paper if that was how training was performed (they just say "512 samples" for 
  both image and audio modes).
  """

  def __init__(self, image_size: int, audio_size: int, audio_patch_size: int, num_samples: int):
    self.image_size = image_size
    self.audio_size = audio_size
    self.audio_patch_size = audio_patch_size
    self.num_samples = num_samples

  def __call__(self, x):
    idx = {
        "image": torch.randint(low=0, high=self.image_size, size=(self.num_samples,)),
        "audio": torch.randint(low=0, high=self.audio_size, size=(self.num_samples,)),
        "label": None,
    }
    audio_patch_samples = rearrange(x["audio"], "b (t dt) -> b t dt", dt=self.audio_patch_size)[:, idx["audio"]]
    y = {
        "image": rearrange(x["image"], "b t c h w -> b (t h w) c")[:, idx["image"], :],
        "audio": rearrange(audio_patch_samples, "b t dt -> b (t dt)"),
        "label": x["label"],
    }
    return idx, y

class SyntheticDataset(Dataset):

  def __len__(self):
    return 100000

  def __getitem__(self, idx):
    return {
      "image": torch.randn(NUM_FRAMES, NUM_CHANNELS, IMAGE_SIZE, IMAGE_SIZE),
      "audio": torch.randn(NUM_AUDIO_ELEMENTS),
      "label": torch.nn.functional.one_hot(torch.randint(low=0, high=NUM_CLASSES, size=()), num_classes=NUM_CLASSES).float(),
    }

def train():
  print(f"==> Preparing data.FLAGS.dataset:{FLAGS.dataset},dataset_dir:'{FLAGS.dataset_dir}")
  is_root = xm.is_master_ordinal(local=False)
  if FLAGS.dataset == "synthetic":
    train_dataset = SyntheticDataset()
    valid_dataset = SyntheticDataset()
  elif FLAGS.dataset  == "kinetics-small":
    train_dataset = build_train_dataset(num_workers=3, root=FLAGS.dataset_dir, debug=True)
    valid_dataset = build_val_dataset(num_workers=3, root=FLAGS.dataset_dir, debug=True)
  elif FLAGS.dataset == "kinetics":
    train_dataset = build_train_dataset(num_workers=64, root=FLAGS.dataset_dir, debug=False)
    valid_dataset = build_val_dataset(num_workers=64, root=FLAGS.dataset_dir, debug=False)
  else:
    raise NotImplementedError(f"unhandled dataset type: {FLAGS.dataset}")

  print("Successfully built the dataset")

  train_loader, test_loader = train_utils.create_data_loaders(
    train_dataset,
    valid_dataset,
    xm.get_ordinal(),
    xm.xrt_world_size(),
    FLAGS.batch_size,
    FLAGS.test_batch_size,
    FLAGS.num_workers,
    FLAGS.drop_last)

  torch.manual_seed(42)

  device = xm.xla_device()
  config = train_utils.load_config(FLAGS.config_file_path)
  model = PerceiverForMultimodalAutoencoding(config).to(device)
  multimodal_output_sampler = MultimodalOutputSampler(
    image_size=NUM_FRAMES*IMAGE_SIZE*IMAGE_SIZE,
    audio_size=AUDIO_SAMPLES_PER_FRAME,
    audio_patch_size=model.config.samples_per_patch,
    num_samples=NUM_OUTPUT_SAMPLES
  )
  writer = None
  if xm.is_master_ordinal():
    logger = train_utils.Logger(FLAGS, xm.xrt_world_size())
  optimizer = optim.AdamW(
      model.parameters(),
      lr=FLAGS.lr,
      weight_decay=0.1)

  if is_root:
    throughput = train_utils.Throughput(FLAGS.batch_size, xm.xrt_world_size(), FLAGS.log_steps)
    print('--------TRAINING CONFIG----------')
    print(FLAGS)
    print('---------------------------------')
    parameters = {
            "Model": "deepmind/multimodal-perceiver",
            "Model configuration": str(model.config),
            "World size": xm.xrt_world_size(),
            "Data parallel degree": xm.xrt_world_size(),
            "Batch size": FLAGS.batch_size,
            "Epoch": FLAGS.num_epochs,
            "Max steps": FLAGS.max_steps,
            "Optimizer": str(optimizer),
            "Dataset": FLAGS.dataset,
            "Environment variables": {variable: value for variable, value in os.environ.items() if variable.startswith("NEURON") or variable.startswith("XLA")}
        }
    train_start = time.time()

  def train_loop_fn(loader, epoch, global_step):
    model.train()
    max_grad_norm = 10
    for step, data in enumerate(loader):
      optimizer.zero_grad()
      # sample outputs from the input
      sampled_idx, sampled_y = multimodal_output_sampler(data)

      output = model(data, subsampled_output_points=sampled_idx)
      logits = output if isinstance(output, torch.Tensor) else output.logits

      loss_image = torch.nn.L1Loss()(logits['image'], sampled_y['image'])
      loss_audio = torch.nn.L1Loss()(logits['audio'], sampled_y['audio'])
      loss_label = torch.nn.CrossEntropyLoss()(logits['label'], sampled_y['label'])

      loss_weight_image = 0.03
      loss_weight_audio = 1.0
      loss_weight_label = 0.0001

      loss = loss_weight_image * loss_image + loss_weight_audio * loss_audio + loss_weight_label * loss_label
      loss.backward()
      torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)  # Gradient clipping is not in AdamW anymore
      xm.optimizer_step(optimizer)
      global_step += 1
      if is_root:
        step_throughput = throughput.get_throughput()
        logger.train_throughputs.append(step_throughput)
        if step % FLAGS.log_steps == 0:
          logger.print_training_update(
            device,
            step,
            FLAGS.lr,
            loss.item(),
            step_throughput,
            epoch,
            writer)
      if global_step >= FLAGS.max_steps:
        xm.mark_step()
        break
    return global_step, loss

  def test_loop_fn(loader, epoch):
    total_samples, correct = 0, 0
    model.eval()
    with torch.no_grad():
      for step, data in enumerate(loader):
        # sample outputs from the input
        sampled_idx, sampled_y = multimodal_output_sampler(data)
        
        output = model(data, subsampled_output_points=sampled_idx)
        #output = model(data)
        logits = output if isinstance(output, torch.Tensor) else output.logits
        loss_image = torch.nn.L1Loss()(logits['image'],sampled_y['image'])
        loss_audio = torch.nn.L1Loss()(logits['audio'],sampled_y['audio'])
        loss_label = torch.nn.CrossEntropyLoss()(logits['label'],sampled_y['label'])

        loss_weight_image = 0.03
        loss_weight_audio = 1.0
        loss_weight_label = 0.0001

        loss = loss_weight_image * loss_image + loss_weight_audio * loss_audio + loss_weight_label * loss_label
        if is_root:
          step_throughput = throughput.get_throughput()
          logger.test_throughputs.append(step_throughput)
          if step % FLAGS.log_steps == 0:
            logger.print_test_update(device, step_throughput, None, epoch, step)
        if step >= FLAGS.max_steps:
          xm.mark_step()
          break
    return loss

  train_device_loader = pl.MpDeviceLoader(train_loader, device)
  test_device_loader = pl.MpDeviceLoader(test_loader, device)
  global_step = 0
  for epoch in range(1, FLAGS.num_epochs + 1):
    xm.master_print('Epoch {} train begin {}'.format(epoch, datetime.now()))
    global_step, loss = train_loop_fn(train_device_loader, epoch, global_step)
    xm.master_print('Epoch {} train end {}'.format(epoch, datetime.now()))
    if FLAGS.metrics_debug:
      xm.master_print(met.metrics_report())
    if is_root:
      average_train_throughput = round(sum(logger.train_throughputs)/len(logger.train_throughputs), 4)
      xm.master_print('Average train throughput: {:.4f}'.format(average_train_throughput))
      xm.master_print('Max train throughput: {:.4f}'.format(max(logger.train_throughputs)))
    if global_step >= FLAGS.max_steps:
      break
  
  if is_root:
    time_to_train = time.time() - train_start
  if FLAGS.do_eval:
    if is_root:
      throughput = train_utils.Throughput(FLAGS.batch_size, xm.xrt_world_size(), FLAGS.log_steps)
    eval_loss = test_loop_fn(test_device_loader, epoch)
    xm.master_print('Epoch {} test end {}, Loss={:.2f}'.format(
        epoch, datetime.now(), eval_loss))
    if is_root:
      average_test_throughput = round(sum(logger.test_throughputs)/len(logger.test_throughputs), 4)
      xm.master_print('Average test throughput: {:.4f}'.format(average_test_throughput))
      xm.master_print('Max test throughput: {:.4f}'.format(max(logger.test_throughputs)))
      xm.master_print('Eval Loss: {:.2f}%'.format(eval_loss))
  
  if is_root:
    # record aggregate & final statistics in the metrics file
    additional_data = {
        "Epoch": epoch, "Global step": global_step
    }
    average_train_throughput = round(sum(logger.train_throughputs)/len(logger.train_throughputs), 4)

    metric_data = [
        Metric("TrainLoss", loss.item(), units="", additional=additional_data),
        Metric("TrainRuntime", round(time_to_train/60, 4), units="minutes", additional=additional_data),
        Metric("TrainMaxThroughput", max(logger.train_throughputs), units="seq/s", additional=additional_data)
    ]

    if FLAGS.expected_average_throughput > 0:
        derived_expected_throughput = (0.95*FLAGS.expected_average_throughput)
        metric_data.append(Metric("TrainMeanThroughput", average_train_throughput, units="seq/s", expected=FLAGS.expected_average_throughput, derived=(0.95*FLAGS.expected_average_throughput), additional=additional_data))
        xm.master_print(f" Posting metrics to heartbeat: \n{metric_data}")
        post_metrics(metric_data, parameters=parameters)
        assert(average_train_throughput >= derived_expected_throughput), "Average throughput :{} is below derived expected threshold: {}".format(average_train_throughput, derived_expected_throughput)
    else:
        metric_data.append(Metric("TrainMeanThroughput", average_train_throughput, units="seq/s", additional=additional_data))
        xm.master_print(f" Posting metrics to heartbeat: \n{metric_data}")
        post_metrics(metric_data, parameters=parameters)

def _mp_fn(index, flags):
  global FLAGS
  FLAGS = flags
  torch.set_default_tensor_type('torch.FloatTensor')
  train()
  xm.rendezvous("_mp_fn finished")

if __name__ == '__main__':
  parser = train_utils.build_train_parser()
  args = parser.parse_args(sys.argv[1:])
  
  if os.environ.get("WORLD_SIZE"):
    #dist.init_process_group('xla')
    _mp_fn(0, args)
  else:
    xmp.spawn(_mp_fn, args=(args,))
