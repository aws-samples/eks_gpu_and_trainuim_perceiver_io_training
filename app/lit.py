from typing import Dict

from einops import rearrange
import lightning.pytorch as pl
import torch
import torch.nn as nn
from transformers import PerceiverForMultimodalAutoencoding
from transformers.models.perceiver.modeling_perceiver import PerceiverClassifierOutput

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


class LitTransformersAutoencodingPerceiverIO(pl.LightningModule):
    def __init__(
          self, 
          model: PerceiverForMultimodalAutoencoding, 
          output_sampler: MultimodalOutputSampler,
          learning_rate: float,
          label_dropout_rate: float = 0.5,
          loss_weight_image: float = 0.03,
          loss_weight_audio: float = 1.0,
          loss_weight_label: float = 0.0001,
        ):
        super().__init__()
        self.model = model
        self.output_sampler = output_sampler
        self.loss_weight_image = loss_weight_image
        self.loss_weight_audio = loss_weight_audio
        self.loss_weight_label = loss_weight_label
        self.learning_rate = learning_rate
        self.label_dropout_rate = label_dropout_rate
        self.label_dropout = nn.Dropout(p=self.label_dropout_rate)

    def forward(self, x, subsampled_output_points):
        return self.model(x, subsampled_output_points=subsampled_output_points)

    def loss(self, y_hat: PerceiverClassifierOutput, y: Dict[str, torch.Tensor]):
        loss_image = torch.nn.L1Loss()(y_hat.logits['image'], y['image'])
        loss_audio = torch.nn.L1Loss()(y_hat.logits['audio'], y['audio'])
        loss_label = torch.nn.CrossEntropyLoss()(y_hat.logits['label'], y['label'])

        loss = self.loss_weight_image*loss_image + self.loss_weight_audio*loss_audio + self.loss_weight_label*loss_label
        return loss

    def training_step(self, batch, batch_idx):
        # sample outputs from the input
        sampled_idx, sampled_y = self.output_sampler(batch)

        # dropout labels input to prevent model from encoding
        # the label directly into the the latent embedding
        batch["label"] = self.label_dropout(batch["label"])

        # encode and reconstruct the sampled indices
        y_hat = self(batch, sampled_idx)

        # compute loss
        loss = self.loss(y_hat, sampled_y)

        self.log('train/loss', loss, prog_bar=True)

        return loss

    def validation_step(self, batch, batch_idx):
        # sample outputs from the input
        sampled_idx, sampled_y = self.output_sampler(batch)

        # encode and reconstruct the sampled indices
        y_hat = self(batch, sampled_idx)

        # compute loss
        loss = self.loss(y_hat, sampled_y)

        self.log('valid/loss', loss, prog_bar=True)

        return loss

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=self.learning_rate, weight_decay=0.1)

