import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import IterableDataset, DataLoader
import pytorch_lightning as pl
from pathlib import Path
import glob
from pytorch_lightning.callbacks import ModelCheckpoint

class SAEIterableDataset(IterableDataset):
    def __init__(self, file_paths):
        super().__init__()
        self.file_paths = file_paths

    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is None:  # single-process data loading, return the full iterator
            file_paths_for_worker = self.file_paths
        else:
            num_workers = worker_info.num_workers
            worker_id = worker_info.id
            file_paths_for_worker = [
                path for i, path in enumerate(self.file_paths) if i % num_workers == worker_id
            ]

        for file_path in file_paths_for_worker:
            data = torch.load(file_path)
            for record in data:
                activations = record["activations"]
                for i in range(activations.shape[0]):
                    yield activations[i]


class SAEDataModule(pl.LightningDataModule):
    def __init__(self, data_dir: str, batch_size: int = 32, num_workers: int = 0):
        super().__init__()
        self.data_dir = Path(data_dir)
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.train_file_paths = []

    def setup(self, stage: str | None = None):
        # Find all training shard files
        # Assuming shard files are named like 'layerX_Y_train.pt'
        if stage == "fit" or stage is None:
            self.train_file_paths = sorted(glob.glob(str(self.data_dir / "*_train.pt")))
            if not self.train_file_paths:
                raise FileNotFoundError(f"No training data shards found in {self.data_dir}")
    
    def prepare_data(self):
        pass

    def train_dataloader(self):
        if not self.train_file_paths:
            # Call setup if dataloader is called before setup (e.g. when not using Trainer.fit)
            self.setup(stage="fit")
            
        train_dataset = SAEIterableDataset(self.train_file_paths)
        return DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=torch.cuda.is_available(),
        )


class SAELightningModule(pl.LightningModule):
    def __init__(self, d_model: int, expansion_factor: int = 8, l1_coeff: float = 1e-3, learning_rate: float = 1e-4):
        super().__init__()
        self.save_hyperparameters() # Saves d_model, expansion_factor, l1_coeff, learning_rate
        
        self.d_model = d_model
        self.d_sae = d_model * expansion_factor
        self.l1_coeff = l1_coeff
        self.learning_rate = learning_rate

        self.encoder = nn.Linear(self.d_model, self.d_sae, bias=True)
        nn.init.constant_(self.encoder.bias, -1.0) 
        self.decoder = nn.Linear(self.d_sae, self.d_model, bias=True)

    def forward(self, x):
        sae_hidden = F.relu(self.encoder(x))
        reconstruction = self.decoder(sae_hidden)
        return reconstruction, sae_hidden

    def training_step(self, batch, batch_idx):
        # batch shape: (batch_size, d_model)
        original_activations = batch

        reconstructed_activations, sae_hidden_activations = self(original_activations)
        
        # MSE Reconstruction Loss
        mse_loss = F.mse_loss(reconstructed_activations, original_activations)
        # L1 Sparsity Loss on hidden activations
        l1_loss = torch.norm(sae_hidden_activations, 1, dim=1).mean()
        total_loss = mse_loss + self.l1_coeff * l1_loss
        
        self.log('train_loss', total_loss, prog_bar=True)
        self.log('mse_loss', mse_loss, prog_bar=True)
        self.log('l1_loss', l1_loss, prog_bar=True)
        self.log('sparsity_l1_coeff', self.l1_coeff, prog_bar=True)
        
        # Calculate L0 norm (number of non-zero activations) for monitoring
        l0_norm = (sae_hidden_activations > 1e-6).float().sum(dim=1).mean()
        self.log('l0_norm_hidden', l0_norm, prog_bar=True)

        return total_loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=1000)
        return [optimizer], [scheduler]

if __name__ == '__main__':
    print("Cuda available:", torch.cuda.is_available())
    pl.seed_everything(42, workers=True)

    # Configuration
    ACTIVATIONS_OUTPUT_DIR = "qwen_acts_idx" # From activations.py
    SAE_CHECKPOINT_DIR = "sae_checkpoints"
    Path(SAE_CHECKPOINT_DIR).mkdir(parents=True, exist_ok=True)

    BATCH_SIZE = 4096
    LEARNING_RATE = 1e-4
    L1_COEFF = 3e-4
    MAX_EPOCHS = 10
    EXPANSION_FACTOR = 8
    D_MODEL = 5120
    CKPT_EVERY = 100

    data_module = SAEDataModule(
        data_dir=ACTIVATIONS_OUTPUT_DIR,
        batch_size=BATCH_SIZE,
        num_workers=4 if torch.cuda.is_available() else 0 # Adjust num_workers based on your system
    )

    # 2. Model
    sae_model = SAELightningModule(
        d_model=D_MODEL,
        expansion_factor=EXPANSION_FACTOR,
        l1_coeff=L1_COEFF,
        learning_rate=LEARNING_RATE,
    )
    print(f"SAE Model created with d_model={sae_model.d_model}, d_sae={sae_model.d_sae}")

    checkpoint_callback = ModelCheckpoint(
        dirpath=SAE_CHECKPOINT_DIR,
        filename='sae-epoch{epoch:02d}-loss{train_loss:.4f}',
        monitor='train_loss',
        mode='min',
        save_top_k=5,
        save_last=True,
        auto_insert_metric_name=False,
        every_n_train_steps=CKPT_EVERY,
    )
    
    resume_from_checkpoint = None
    last_ckpt_path = Path(SAE_CHECKPOINT_DIR) / "last.ckpt"
    if last_ckpt_path.exists():
        print(f"Found last checkpoint at {last_ckpt_path}, attempting to resume.")
        resume_from_checkpoint = str(last_ckpt_path)
    else:
        print("No 'last.ckpt' found. Starting training from scratch.")

    # 4. Trainer
    accelerator = "gpu" if torch.cuda.is_available() else "cpu"
    devices = [0] if torch.cuda.is_available() else 1 # Use first GPU if available, or 1 CPU core
    trainer = pl.Trainer(
        accelerator=accelerator,
        devices=devices,
        callbacks=[checkpoint_callback],
        logger=True, # Uses TensorBoardLogger by default, logs to lightning_logs/
        precision='bf16-mixed',
        max_epochs=MAX_EPOCHS,
    )

    print("Starting SAE training...")
    trainer.fit(
        model=sae_model, 
        datamodule=data_module,
        ckpt_path=resume_from_checkpoint # Pass path to resume_from_checkpoint if it exists
    )
    print("Training finished.")

    # Optionally, save the final trained model explicitly
    final_model_path = Path(SAE_CHECKPOINT_DIR) / "sae_final_model.ckpt"
    trainer.save_checkpoint(final_model_path)
    print(f"Final model saved to {final_model_path}")

