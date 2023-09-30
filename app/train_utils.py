import os
import sys
import argparse
from datetime import datetime
import math
import queue
import time
import inspect
import yaml
from torch.utils.data.dataloader import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.utils.tensorboard import SummaryWriter
from transformers import PerceiverConfig
from lightning.pytorch.callbacks import Callback

def load_config(config_file_path: str):
  with open(config_file_path, "r") as f:
    return PerceiverConfig.from_dict(yaml.load(f, yaml.CLoader))

class Throughput:
  def __init__(self, batch_size, world_size, moving_avg_window_size=10):
    self.seqs_per_iteration = batch_size * world_size
    self.moving_avg_window_size = moving_avg_window_size
    self.moving_avg_window = queue.Queue()
    self.window_time = 0
    self.start_time = time.time()

  def get_throughput(self):
    step_time = time.time() - self.start_time
    self.start_time += step_time
    self.window_time += step_time
    self.moving_avg_window.put(step_time)
    window_size = self.moving_avg_window.qsize()
    if window_size > self.moving_avg_window_size:
        self.window_time -= self.moving_avg_window.get()
        window_size -= 1
    throughput = window_size * self.seqs_per_iteration / self.window_time
    return throughput


class ThroughputCallback(Callback):
  def __init__(self, batch_size, world_size):
    self.history: List = []
    self.average_throughput: float = -1
    self.max_throughput: float = -1
    self.throughput: Throughput = None
    self.batch_size = batch_size
    self.world_size = world_size

  def on_train_start(self, trainer, pl_module):
    if trainer.local_rank == 0:
      self.throughput = Throughput(
          batch_size=self.batch_size,
          world_size=self.world_size,
      )

  def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
    if trainer.local_rank == 0:
      step_throughput = self.throughput.get_throughput()
      self.history.append(step_throughput)
      print(f"  Step throughput {step_throughput}, seq/s")

  def on_train_epoch_end(self, trainer, pl_module):
    if trainer.local_rank == 0:
      self.max_throughput = max(self.max_throughput, max(self.history))
      mean_throughput = self.average_throughput
      self.average_throughput = (
        (mean_throughput + sum(self.history) / len(self.history)) * 0.5
        if (mean_throughput > 0)
        else sum(self.history) / len(self.history)
      )
      self.history = []

      # for some reason logs are not showing, so print it
      print(f"  Average throughput {self.average_throughput}, seq/s")
      print(f"  Peak throughput {self.max_throughput}, seq/s")

  def on_train_end(self, trainer, pl_module):
    #if state.is_local_process_zero:
      print("***** Training Throughput stats*****")
      print(f"  Average throughput {self.average_throughput}, seq/s")
      print(f"  Peak throughput {self.max_throughput}, seq/s")

      # for some reason logs are not showing, so print it
      print(f"  Average throughput {self.average_throughput}, seq/s")
      print(f"  Peak throughput {self.max_throughput}, seq/s")


class Logger:
  def __init__(self, args, world_size):
        xla = 'torch_xla' in sys.modules
        self.train_throughputs = []
        self.test_throughputs = []
        self.summary_writer = SummaryWriter(os.path.join(args.logdir,
                                             f"neuron_tblogs_{time.strftime('%m%d%y_%H%M')}"
                                             f"_w{world_size}"
                                             f"_lr{args.lr}"
                                             f"_bs{args.batch_size}"
                                             f"_bf16autocast{args.enable_pt_autocast}"
                                             f"_xla{xla}"))
        self.summary_writer.add_text('script', "```\n" + inspect.getsource(sys.modules[__name__]) + "\n```", 0)

  def print_training_update(self,
                          device,
                          step,
                          lr,
                          loss,
                          throughput,
                          epoch=None,
                          summary_writer=None):
    """Prints the training metrics at a given step.

    Args:
      device (torch.device): The device where these statistics came from.
      step_num (int): Current step number.
      loss (float): Current loss.
      throughput (float): The examples/sec throughput for the current batch.
      epoch (int, optional): The epoch number.
      summary_writer (SummaryWriter, optional): If provided, this method will
        write some of the provided statistics to Tensorboard.
    """
    update_data = [
        'Training', 'Device={}'.format(str(device)),
        'Epoch={}'.format(epoch) if epoch is not None else None,
        'Step={}'.format(step), 'Learning_Rate={}'.format(lr),
        'Loss={:.5f}'.format(loss), 'Throughput={:.5f}'.format(throughput),
        'Time={}'.format(datetime.now())
    ]
    print('|', ' '.join(item for item in update_data if item), flush=True)
    self.write_to_summary(
        summary_writer,
        dict_to_write={
            'Throughput': throughput,
        })

  def print_test_update(self, device, throughput, accuracy, epoch=None, step=None):
    """Prints single-core test metrics.

    Args:
      device: Instance of `torch.device`.
      accuracy: Float.
    """
    update_data = [
        'Test', 'Device={}'.format(str(device)),
        'Step={}'.format(step) if step is not None else None,
        'Epoch={}'.format(epoch) if epoch is not None else None,
        'Throughput={:.5f}'.format(throughput),
        'Accuracy={:.2f}'.format(accuracy) if accuracy is not None else None,
        'Time={}'.format(datetime.now())
    ]
    print('|', ' '.join(item for item in update_data if item), flush=True)

  def write_to_summary(self,
                      global_step=None,
                      dict_to_write={}):
    """Writes scalars to a Tensorboard SummaryWriter.

    Optionally writes XLA perf metrics.

    Args:
      global_step (int, optional): The global step value for these data points.
        If None, global_step will not be set for this datapoint.
      dict_to_write (dict, optional): Dict where key is the scalar name and value
        is the scalar value to be written to Tensorboard.
    """
    if self.summary_writer is None:
      return
    for k, v in dict_to_write.items():
      self.summary_writer.add_scalar(k, v, global_step)


def build_train_parser():
  parser = argparse.ArgumentParser()
  parser.add_argument('--dataset', type=str, default="synthetic", help="dataset name.")
  parser.add_argument('--dataset_dir', type=str, default="/dataset", help="dataset directory.")
  parser.add_argument('--config_file_path', type=str, default="config/main.yaml", help="model config file path.")
  parser.add_argument('--logdir', type=str, default="log_training", help="Training log directory.")
  parser.add_argument('--batch_size', type=int, default=8, help="Batch size per core used in training.")
  parser.add_argument('--num_epochs', type=int, default=2, help="Number of training epochs.")
  parser.add_argument('--num_workers', type=int, default=0, help="Number of worker used in data loader.")
  parser.add_argument('--log_steps', type=int, default=20, help="Number of steps between each other log message.")
  parser.add_argument('--max_steps', type=int, default=28125, help="Number of max training steps.")
  parser.add_argument('--expected_average_throughput', type=int, default=0, help="Expected average training throughput (seq/s).")
  parser.add_argument('--image_dim', type=int, default=224, help="Image dimension after transformation.")
  parser.add_argument('--test_batch_size', type=int, default=8, help="Batch size per core used in testing.")
  parser.add_argument('--lr', type=float, default=0.00005, help="Learning rate used in training.")
  parser.add_argument('--momentum', type=float, default=0.9, help="Momentum used in SGD optimizer")
  parser.add_argument('--target_accuracy', type=float, default=0, help="Target accuracy (%).")
  parser.add_argument('--drop_last', action='store_true', help="Enable deop_last in data loader.")
  parser.add_argument('--metrics_debug', action='store_true', help="Print debug metrics at the end of each epoch.")
  parser.add_argument('--enable_pt_autocast', action='store_true', help="Enable Auto-cast to BF16 in GPU.")
  parser.add_argument('--do_eval', action='store_true', help="Evaluate the model with eval dataset after training.")

  return parser

def create_data_loaders(train_dataset,
                        test_dataset,
                        rank,
                        world_size,
                        train_batch_size,
                        test_batch_size,
                        num_workers,
                        drop_last=False):
  train_sampler, test_sampler = None, None
  if world_size > 1:
    train_sampler = DistributedSampler(
        train_dataset,
        num_replicas=world_size,
        rank=rank,
        shuffle=True)
    test_sampler = DistributedSampler(
        test_dataset,
        num_replicas=world_size,
        rank=rank,
        shuffle=False)

  train_loader = DataLoader(
      train_dataset,
      batch_size=train_batch_size,
      shuffle=False if train_sampler else True,
      sampler=train_sampler,
      drop_last=drop_last,
      num_workers=num_workers,
      pin_memory=True)
  test_loader = DataLoader(
      test_dataset,
      batch_size=test_batch_size,
      shuffle=False,
      sampler=test_sampler,
      drop_last=drop_last,
      num_workers=num_workers,
      pin_memory=True)

  return train_loader, test_loader
