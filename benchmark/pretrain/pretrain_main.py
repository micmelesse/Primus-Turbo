import os

os.environ["TOKENIZERS_PARALLELISM"] = "false"

import torch
from data.build_dataset import get_dataloaders
from engine.trainer import Trainer
from models.basic_llama import LlamaBasicModel
from transformers import LlamaConfig


# TODO:
class Config:
    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)

    def __getitem__(self, key):
        return getattr(self, key)

    def __setitem__(self, key, value):
        return setattr(self, key, value)

    def to_dict(self):
        return self.__dict__


def build_config(
    model_path: str,
    dataset_name: str,
    batch_size: int = 4,
    context_length: int = 8192,
    max_steps: int = 1000,
    warmup_steps: int = 10,
    lr: float = 3e-4,
    **kwargs,
):
    hf_config = LlamaConfig.from_pretrained(model_path)

    if isinstance(hf_config.torch_dtype, str):
        try:
            torch_dtype = getattr(torch, hf_config.torch_dtype)
        except AttributeError:
            raise ValueError(f"Invalid torch_dtype string: {hf_config.torch_dtype}")
    else:
        torch_dtype = hf_config.torch_dtype

    return Config(
        hf_config=hf_config,
        model_path=model_path,
        dataset_name=dataset_name,
        batch_size=batch_size,
        context_length=context_length,
        max_steps=max_steps,
        warmup_steps=warmup_steps,
        lr=lr,
        torch_dtype=torch_dtype,
        **kwargs,
    )


if __name__ == "__main__":

    # === Cfg ===
    config = build_config(
        model_path="meta-llama/Llama-3.1-8B-Instruct",
        dataset_name="wikitext",  # c4 , wikitext
        batch_size=4,
        context_length=8192,
        max_steps=50,
        warmup_steps=100,
        lr=3e-4,
    )
    # === Model ===
    model = LlamaBasicModel(config.hf_config).cuda()

    # === Data ===
    train_loader, val_loader = get_dataloaders(config)

    # === Trainer ===
    trainer = Trainer(
        config=config,
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
    )
    trainer.train()
