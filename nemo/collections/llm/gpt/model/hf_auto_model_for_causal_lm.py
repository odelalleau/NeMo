# Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import lightning.pytorch as pl
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM

from nemo.collections.common.tokenizers.huggingface.auto_tokenizer import AutoTokenizer
from nemo.collections.llm import fn
from nemo.lightning import io
from nemo.utils import logging


def masked_cross_entropy(logits, targets, mask=None):
    if mask is not None:
        loss = F.cross_entropy(logits, targets, reduction='none')
        return torch.mean(loss[mask == 1])
    else:
        return F.cross_entropy(logits, targets)


def align_labels(logits, labels):
    logits = logits.float()
    n_cls = logits.shape[-1]
    if logits.shape[-2] == labels.shape[-1]:
        logits = logits[..., :-1, :].contiguous()
        labels = labels[..., 1:].contiguous()
    elif logits.shape[-2] == labels.shape[-1] + 1:
        logits = logits[..., :-1, :].contiguous()
    else:
        raise ValueError("Mismatched labels and logits shapes (" + str(labels.shape) + " " + str(logits.shape))
    return logits.view(-1, n_cls), labels.view(-1)


class HFAutoModelForCausalLM(pl.LightningModule, io.IOMixin, fn.FNMixin):
    def __init__(
        self,
        model_name='gpt2',
        load_pretrained_weights=True,
        tokenizer=None,
        loss_fn=masked_cross_entropy,
        model_transform=None,
        model_accelerator=None,
        trust_remote_code=False,
        default_dtype=torch.bfloat16,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.model_name = model_name
        self._tokenizer = None
        self.model = None
        self.loss_fn = loss_fn
        self.load_pretrained_weights = load_pretrained_weights
        self.is_hf_model = True
        self.model_transform = model_transform
        self.model_accelerator = model_accelerator
        self.trust_remote_code = trust_remote_code
        self.default_dtype = default_dtype

    @property
    def tokenizer(self):
        if self._tokenizer is None:
            self._tokenizer = HFAutoModelForCausalLM.configure_tokenizer(self.model_name, self.trust_remote_code)
        return self._tokenizer

    @tokenizer.setter
    def tokenizer(self, value):
        assert self._tokenizer is None
        self._tokenizer = value

    @staticmethod
    def configure_tokenizer(model_name, trust_remote_code=False):
        return AutoTokenizer(model_name, trust_remote_code=trust_remote_code)

    def configure_model(self):
        # create all your layers here
        if self.load_pretrained_weights:
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name, torch_dtype='auto', trust_remote_code=self.trust_remote_code
            )
        else:
            from transformers import AutoConfig

            config = AutoConfig.from_pretrained(self.model_name, trust_remote_code=self.trust_remote_code)
            dtype = getattr(config, 'torch_dtype', self.default_dtype)
            self.model = AutoModelForCausalLM.from_config(
                config, torch_dtype=dtype, trust_remote_code=self.trust_remote_code
            )

        if self.model_accelerator is not None:
            self.model_accelerator(self.model)

        self.model.train()

    def forward(self, batch):
        return self.model(**batch)

    def training_step(self, batch):
        labels = batch.pop('labels').to(self.model.device)
        loss_mask = batch.pop('loss_mask', None)

        outputs = self.forward(batch)

        # Prepare for loss calculation
        logits, labels = align_labels(outputs.logits.float(), labels)
        assert logits.shape[-2] == labels.shape[-1]

        loss = self.loss_fn(logits, labels, loss_mask)
        self.log('train_log', loss, on_step=True, on_epoch=True, prog_bar=True)
        return loss

    @torch.no_grad
    def validation_step(self, batch, batch_idx):
        labels = batch.pop('labels').to(self.model.device)
        loss_mask = batch.pop('loss_mask', None)

        outputs = self.forward(**batch)

        logits, labels = align_labels(outputs.logits.float(), labels)
        assert logits.shape[-2] == labels.shape[-1]
        loss = self.loss_fn(logits, labels, loss_mask)

        self.log('val_loss', loss, on_step=True, on_epoch=True, prog_bar=True)

    def save_pretrained(self, path):
        assert self.model is not None, "Model has to be created first."
        self.model.save_pretrained(path)
        if self._tokenizer is not None:
            self._tokenizer.save_pretrained(path)
        else:
            logging.warning("A tokenizer wasn't created before to save.")
