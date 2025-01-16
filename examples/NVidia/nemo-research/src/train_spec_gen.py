from pathlib import Path
from typing import Tuple

import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from nemo.core import NeuralModule
from nemo.core.config import hydra_runner
from nemo.collections.tts.models import FastPitchModel, SpectrogramEnhancerModel

from numpy_to_pl_dataset import NumpyWrapper


fastpitch = FastPitchModel.from_pretrained(model_name="tts_en_fastpitch")
fastpitch.freeze()

class ConvEmbeddingModule(NeuralModule, pl.LightningModule):
    def __init__(
        self,
        embedding_dim: int = 128,
        vocab_size: int = 1,
        conv_channels: int = 128,
        kernel_size: int = 3,
        lr: float = 1e-3,
    ):
        super().__init__()
        # self.save_hyperparameters()

        # Convert int64 tokens to float
        # self.embedding = nn.Embedding(
        #     num_embeddings=vocab_size,
        #     embedding_dim=embedding_dim
        # )

        # 1D Conv: in_channels = embedding_dim, out_channels = conv_channels
        # Conv1d expects input of shape [B, C, L], so we'll transpose after embedding
        self.conv = nn.Conv1d(
            in_channels=embedding_dim,
            out_channels=conv_channels,
            kernel_size=kernel_size,
            padding=(kernel_size // 2)  # so the output length remains N
        )

        # A projection layer back to vocab size
        # This will create logits of shape [B, N, vocab_size] after we transpose back
        self.fc = nn.Linear(conv_channels, vocab_size)

        self.lr = lr

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: int64 tensor of shape [B, N]
        Returns:
            pred: int64 tensor of shape [B, N]
        """
        # x -> [B, N] (token IDs)
        is_single_example = x.shape[0] == 1

        # 1) Embed to [B, N, embedding_dim]
        x_emb = x.to(torch.float)#self.embedding(x)

        # 2) Rearrange to [B, embedding_dim, N] for Conv1D
        # x_emb = x_emb.transpose(1, 2)  # shape -> [B, E, N]

        # 3) Apply Conv1D -> [B, conv_channels, N]
        x_conv = self.conv(x_emb)

        print("X conv shape:", x_conv.shape)

        if is_single_example:
            x_conv = x_conv.unsqueeze(0)

        # 4) Transpose back to [B, N, conv_channels]
        x_conv = x_conv.transpose(1, 2)

        # 5) Map to vocab logits -> [B, N, vocab_size]
        logits = self.fc(x_conv)

        # 6) Convert logits to predicted IDs (argmax)
        pred = torch.argmax(logits, dim=-1)

        return pred

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)


class AdapterE(NeuralModule, pl.LightningModule):
    """
    A simple adapter that takes input embeddings (fixed),
    transforms them, and feeds them into a frozen image generator.
    """
    def __init__(
        self,
        specgram_generator: NeuralModule,
    ):
        super().__init__()
        
        # Save hyperparameters, if desired
        # self.save_hyperparameters(ignore=["generator"])
        
        # 1) Store the generator and freeze it
        self.generator = specgram_generator
        self.generator.freeze()  # user-defined freeze method
        # (Alternatively: self._freeze_module_params(self.generator))

        # 2) Define an adapter (simple linear in this example)
        self.adapter = ConvEmbeddingModule(1)
            #embedding_dim = specgram_generator.cfg.symbols_embedding_dim)

        self.lr = 1e-3

    def forward(self, embeddings: torch.Tensor) -> torch.Tensor:
        """
        Forward pass:
        embeddings -> adapter -> generator -> output_image
        """
        # Adapt embeddings to generator’s expected input shape
        adapted = self.adapter(embeddings)  

        # Pass through the frozen generator
        images = self.generator.generate_spectrogram(
            tokens=adapted, speaker=0, reference_spec=None, reference_spec_lens=None)
        return images

    def training_step(self, batch, batch_idx):
        """
        Example training step:
         - batch might be (embeddings, target_images)
         - We forward to get generated_images
         - Compute loss vs. target_images
         - Return the loss, which backpropagates into the adapter only
        """
        embeddings, target_images = batch  # both float tensors
        
        generated_images = self(embeddings)  # shape depends on generator output

        # Example: simple pixel-wise MSE loss
        loss = F.mse_loss(generated_images, target_images)
        self.log("train_loss", loss)
        return loss

    def configure_optimizers(self):
        """
        Only optimize the adapter parameters.
        The generator is frozen; it will not update.
        """
        return torch.optim.Adam(self.adapter.parameters(), lr=self.lr)

    @staticmethod
    def _freeze_module_params(module: nn.Module):
        """
        Helper to manually freeze any PyTorch module
        (in case .freeze() is not implemented).
        """
        for param in module.parameters():
            param.requires_grad = False

def io_item_summary(item: dict):
    print("Keys:", item.keys())
    for key in item.keys():
        print(f"Key: {key}, shape: {item[key].shape}")

def load_and_preprocess(path_to_prepro: Path, ) -> Tuple[torch.Tensor, torch.Tensor]:

    embedding_size = fastpitch.cfg.symbols_embedding_dim
    print("Embedding size:", embedding_size)

    data = np.load(path_to_prepro, allow_pickle=True).item()

    emb_data = {}

    max_tokens = 0

    for key in data.keys():
        key_data = data[key]
        psg_data = key_data.pop('psg_str')
        emb_psg = fastpitch.parse(psg_data)
        max_tokens = max(max_tokens, emb_psg.shape[1])
        emb_data[key] = {
            "X": emb_psg,
            "y": key_data['spectrogram'][:, :80]
        }
        # io_item_summary(emb_data[key])
    
    # padd each embedding to max_tokens
    for key in emb_data.keys():
        emb_data[key]["X"] = torch.nn.functional.pad(emb_data[key]["X"], (0, max_tokens - emb_data[key]["X"].shape[1]))
    
    torch_X = torch.stack([torch.Tensor(item["X"]) for item in emb_data.values()])
    torch_y = torch.stack([torch.Tensor(item["y"]) for item in emb_data.values()])
    return torch_X, torch_y


if __name__ == '__main__':
    path_to_prepro = Path('./preprocessed_data/stationary_preprocessed_data_WLDM_str.npy')
    emb_X_y = load_and_preprocess(path_to_prepro)

    # TODO: utilize PyTorch Lightning for training
    adapter = AdapterE(specgram_generator=fastpitch)

    trainer = pl.Trainer(precision=16, accelerator="gpu", max_epochs=10, log_every_n_steps=5)

    train_Xy = NumpyWrapper(emb_X_y[0], emb_X_y[1])

    trainer.fit(adapter, emb_X_y)