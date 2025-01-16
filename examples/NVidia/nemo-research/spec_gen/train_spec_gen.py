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
from nemo.collections.common.parts import adapter_modules

from numpy_to_pl_dataset import NumpyWrapper



class ConvEmbeddingModule(NeuralModule, pl.LightningModule):
    def __init__(
        self,
        embedding_dim: int = 1,
        conv_channels: int = 128,
        kernel_size: int = 3,
        lr: float = 1e-3,
    ):
        super().__init__()

        # 1D Conv: in_channels = embedding_dim, out_channels = conv_channels
        # Conv1d expects input of shape [B, C, L] where C is the number of channels
        self.conv = nn.Conv1d(
            in_channels=embedding_dim,
            out_channels=conv_channels,
            kernel_size=kernel_size,
            padding=(kernel_size // 2)  # so the output length remains N
        )

        self.fc = nn.Linear(conv_channels, embedding_dim)

        self.lr = lr

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: int64 tensor of shape [B, N]
        Returns:
            pred: int64 tensor of shape [B, N]
        """
        # x -> [B, N] (token IDs)
        print("X shape:", x.shape)

        x_emb = x.to(torch.float).unsqueeze(1)  # shape -> [B, 1, N]


        # 3) Apply Conv1D -> [B, conv_channels, N]
        x_conv = self.conv(x_emb)

        print("X conv shape:", x_conv.shape)

        # 4) Transpose back to [B, N, conv_channels]
        x_conv = x_conv.transpose(1, 2)

        # 5) Map back to emb space -> [B, N, 1]
        proj = self.fc(x_conv)

        pred = proj.squeeze(2)

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
        
        self.max_embedding_value = specgram_generator.cfg.symbols_embedding_dim
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
        # clamp output to [0, max_embedding_value]
        adapted = torch.clamp(adapted, 0, self.max_embedding_value)
        print("Adapted shape:", adapted.shape)
        adapted = adapted.long() # required by the generator

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

        # print("Embeddings shape:", embeddings.shape)
        # print("Target images shape:", target_images.shape)
        
        generated_images = self(embeddings)  # shape depends on generator output


        ## Now we have turned our embeddings into images
        ## We need to make sure the images are the same size for the MSE loss below
        # print("Generated images shape:", generated_images.shape)
        shapes_2 = [target_images.shape[2], generated_images.shape[2]]
        shapes_1 = [target_images.shape[1], generated_images.shape[1]]

        # pad to match target_images
        generated_images = F.pad(generated_images, (
            0, max(shapes_2) - generated_images.shape[2],
            0, max(shapes_1) - generated_images.shape[1]
        ))

        # pad to match generated_images
        target_images = F.pad(target_images, (
            0, max(shapes_2) - target_images.shape[2],
            0, max(shapes_1) - target_images.shape[1]
        ))

        # Simple pixel-wise MSE loss
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

def load_and_preprocess(path_to_prepro: Path, parser) -> Tuple[torch.Tensor, torch.Tensor]:

    data = np.load(path_to_prepro, allow_pickle=True).item()

    emb_data = {}

    max_tokens = 0
    min_tokens = 10000000

    for key in data.keys():
        key_data = data[key]
        psg_data = key_data.pop('psg_str')
        emb_psg = parser.parse(psg_data).squeeze()
        spectrogram = key_data['spectrogram'].squeeze().transpose(1, 0)
        max_tokens = max(max_tokens, emb_psg.shape[0])
        min_tokens = min(min_tokens, emb_psg.shape[0])
        emb_data[key] = {
            "X": emb_psg,
            "y": spectrogram[:80]
        }
        # io_item_summary(emb_data[key])
    
    print("@@@@ Max tokens:", max_tokens)
    print("@@@@ Min tokens:", min_tokens)
    # padd each embedding to max_tokens
    for key in emb_data.keys():
        emb_data[key]["X"] = torch.nn.functional.pad(emb_data[key]["X"], (0, max_tokens - emb_data[key]["X"].shape[0]))
    
    # embeddings
    torch_X = torch.stack([torch.Tensor(item["X"]) for item in emb_data.values()])

    # spectrograms
    torch_y = torch.stack([torch.Tensor(item["y"]) for item in emb_data.values()])
    print("@@@@ X shape:", torch_X.shape)
    print("@@@@ y shape:", torch_y.shape)
    return torch_X, torch_y

def main(path_to_prepro: Path):
    fastpitch = FastPitchModel.from_pretrained(model_name="tts_en_fastpitch")
    fastpitch.freeze()
    emb_X_y = load_and_preprocess(path_to_prepro, fastpitch)
    adapter = AdapterE(specgram_generator=fastpitch)
    trainer = pl.Trainer(precision=16, accelerator="gpu", max_epochs=10, log_every_n_steps=5)

    train_Xy = NumpyWrapper(emb_X_y[0], emb_X_y[1], batch_size=1)

    # set random seed for torch
    pl.seed_everything(420)

    # train_Xy.setup()

    # tdl = train_Xy.train_dataloader()
    # for batch in tdl:
    #     print(f"@@@@ Batch shape:\n X: {batch[0].shape}\n y: {batch[1].shape}")
    #     adapter.to('cuda:0')
    #     output = adapter(batch[0].to('cuda:0'))
    #     print("@@@@ Output shape:", output.shape)
    #     import matplotlib.pyplot as plt
    #     plt.imshow(output.squeeze().detach().cpu().numpy(),
    #                aspect='auto', origin='lower')
    #     plt.tight_layout(pad=0)
    #     plt.box(False)
    #     plt.savefig("output.png")
    #     plt.close()
    #     # break

    trainer.fit(adapter, train_Xy)


if __name__ == '__main__':
    path_to_prepro = Path('./preprocessed_data/stationary_preprocessed_data_WLDM_str.npy')

    main(path_to_prepro)