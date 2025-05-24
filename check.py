import torch
import json
from pathlib import Path
import sys

# Attempt to add current directory to sys.path to help with local imports
# Run this script from the root of your qwen-rl project or ensure sae.py is discoverable
sys.path.append('.')

# Attempt to import necessary components from your sae.py
from sae import (
    SAEDataModule,
    SAELightningModule,
    ACTS_DIR,
    LAYER,
    BATCH, # Using BATCH from sae.py for consistency in dataloading
    SEED,
    D_MODEL,
    HIDDEN,
    # BETA, # Not strictly needed for this script's SAELightningModule init
    LR,   # For SAELightningModule init
)



def check_normalization_script():
    print("\n--- Normalization Sanity Check Script ---")

    # --- 1. Inspect Stored Statistics (normalization_metadata.json) ---
    print("\n--- 1. Inspecting Stored Statistics (normalization_metadata.json) ---")
    acts_dir_path = Path(ACTS_DIR)
    metadata_file = acts_dir_path / "normalization_metadata.json"

    if not metadata_file.exists():
        print(f"  ERROR: {metadata_file} not found.")
        print("  Please run your main training script (sae.py) at least once with normalization")
        print("  enabled in dm.setup(norm=True) to generate this file.")
        mu_loaded, sigma_loaded = None, None
    else:
        try:
            meta = json.loads(metadata_file.read_text())
            mu_loaded = torch.tensor(meta["mu"], dtype=torch.float32)
            sigma_loaded = torch.tensor(meta["sigma"], dtype=torch.float32)
            d_model_meta = meta.get("d_model")
            files_meta_count = len(meta.get("shard_files", []))
            samples_meta_count = meta.get("count")

            print(f"  Metadata loaded from: {metadata_file}")
            print(f"  d_model in metadata: {d_model_meta} (Expected from sae.py: {D_MODEL})")
            if d_model_meta != D_MODEL:
                print(f"  WARNING: d_model mismatch! Metadata: {d_model_meta}, Script: {D_MODEL}")
            print(f"  Number of shard files used for stats: {files_meta_count}")
            print(f"  Total samples used for stats: {samples_meta_count}")

            print(f"  μ shape: {mu_loaded.shape}, range: ({mu_loaded.min().item():.4f}, {mu_loaded.max().item():.4f})")
            print(f"  σ shape: {sigma_loaded.shape}, range: ({sigma_loaded.min().item():.4f}, {sigma_loaded.max().item():.4f})")
            print(f"  σ mean: {sigma_loaded.mean().item():.4f}")

            if mu_loaded.shape[0] != D_MODEL:
                print(f"  ERROR: μ dimension ({mu_loaded.shape[0]}) does not match D_MODEL ({D_MODEL}).")
            if not (1e-3 < sigma_loaded.min().item() < 10.0): # Relaxed upper bound for sigma min check
                print(f"  WARNING: σ min ({sigma_loaded.min().item():.4f}) is outside a typical healthy range (e.g., 1e-3 to 10.0).")
            if sigma_loaded.max().item() > 20.0:
                print(f"  WARNING: σ max ({sigma_loaded.max().item():.4f}) is very high (>20.0).")
            if sigma_loaded.min().item() < 1e-5: # Stricter check for very small sigma
                 print(f"  CRITICAL WARNING: σ min ({sigma_loaded.min().item():.4f}) is extremely small (<1e-5). Likely to cause division by zero or very large numbers.")

        except Exception as e:
            print(f"  ERROR: Could not load or parse {metadata_file}: {e}")
            mu_loaded, sigma_loaded = None, None

    if SAEDataModule is None or SAELightningModule is None:
        print("\nSkipping DataModule, Model, and Batch checks due to import issues from sae.py.")
        print("Normalization check script finished (partially).")
        return

    # --- 2. Verifying DataModule and Model Initialization ---
    print("\n--- 2. Verifying DataModule and Model Initialization ---")
    dm = None
    model = None
    try:
        dm = SAEDataModule(ACTS_DIR, LAYER, BATCH, SEED, D_MODEL)
        print("  Instantiated SAEDataModule.")
        print("  Calling dm.setup(norm=True)... (This will load/compute stats)")
        dm.setup(stage='fit', norm=True)

        dm_norm_mu = dm.norm_mu
        dm_norm_sigma = dm.norm_sigma

        if dm_norm_mu is None or dm_norm_sigma is None:
            print("  WARNING: dm.norm_mu or dm.norm_sigma is None after dm.setup(norm=True).")
            print("           Model will likely use default no-op normalization if stats aren't passed.")
        else:
            print(f"  DataModule loaded mu.shape: {dm_norm_mu.shape}, sigma.shape: {dm_norm_sigma.shape}")

        # Instantiate model with BETA=0 for checking reconstruction without L1 influence
        model = SAELightningModule(
            d_model=D_MODEL, hidden=HIDDEN, beta=0.0, lr=LR,
            norm_mu=dm_norm_mu, norm_sigma=dm_norm_sigma
        )
        print("  Instantiated SAELightningModule.")
        print(f"  model.apply_normalization: {model.apply_normalization}")
        print(f"  Initial model parameter dtype (from next(model.parameters()).dtype): {next(model.parameters()).dtype}")
        if model.apply_normalization and (dm_norm_mu is None or dm_norm_sigma is None):
             print("  WARNING: model.apply_normalization is True, but it seems no stats were provided from DataModule.")
        elif not model.apply_normalization and (dm_norm_mu is not None and dm_norm_sigma is not None):
             print("  WARNING: model.apply_normalization is False, but DataModule had stats. Check model init.")


    except Exception as e:
        print(f"  ERROR during DataModule/Model initialization: {e}")
        import traceback
        traceback.print_exc()


    # --- 3. Checking a Batch Before/After Normalization ---
    print("\n--- 3. Checking a Batch Before/After Normalization ---")
    raw_batch_for_loss_check = None
    if dm and hasattr(dm, 'file_paths') and dm.file_paths:
        try:
            print("  Getting train_dataloader...")
            dl = dm.train_dataloader()
            print("  Fetching a batch...")
            raw_batch = next(iter(dl))
            raw_batch_for_loss_check = raw_batch.clone() # Save for later recon loss check
            print(f"  Raw batch shape: {raw_batch.shape}, dtype: {raw_batch.dtype}")

            if model and hasattr(model, 'apply_normalization'):
                # Simulate normalization as done in the model
                x_processed = raw_batch.clone()
                if model.apply_normalization:
                    # model.norm_mu and model.norm_sigma are buffers.
                    # Ensure they are on the same device and dtype as batch for the operation.
                    model_mu = model.norm_mu.to(raw_batch.device, dtype=raw_batch.dtype)
                    model_sigma = model.norm_sigma.to(raw_batch.device, dtype=raw_batch.dtype)
                    
                    # Check if model is using actual stats or placeholders
                    if dm_norm_mu is not None and not torch.allclose(model_mu, dm_norm_mu.to(raw_batch.device, dtype=raw_batch.dtype)):
                        print("  INFO: Model's 'norm_mu' buffer differs from DataModule's loaded 'norm_mu'.")
                    elif torch.all(model_mu == 0) and torch.all(model_sigma == 1) and dm_norm_mu is not None :
                         print("  INFO: Model seems to be using placeholder (no-op) normalization buffers, even though DataModule provided stats.")
                    
                    x_processed = (raw_batch - model_mu) / model_sigma
                    print(f"  Stats for raw batch:          mean={raw_batch.mean().item():.4f}, std={raw_batch.std().item():.4f}")
                    print(f"  Stats for x_processed (normed by model logic): mean={x_processed.mean().item():.4f}, std={x_processed.std().item():.4f}")
                    if not (abs(x_processed.mean().item()) < 0.15 and abs(x_processed.std().item() - 1.0) < 0.15): # Relaxed tolerance
                        print("  WARNING: x_processed mean is not close to 0 or std is not close to 1.")
                else:
                    print("  model.apply_normalization is False. x_processed is same as raw_batch.")
                    print(f"  Stats for x_processed (raw): mean={raw_batch.mean().item():.4f}, std={raw_batch.std().item():.4f}")
            else:
                print("  Skipping batch normalization check as model or apply_normalization flag is unavailable.")
        except Exception as e:
            print(f"  ERROR during batch check: {e}")
            import traceback
            traceback.print_exc()
    else:
        print("  Skipping batch check: DataModule not initialized or no files found.")

    # --- 4. Initial Reconstruction Loss (Simulated) ---
    print("\n--- 4. Initial Reconstruction Loss (Simulated, BETA=0) ---")
    if model and raw_batch_for_loss_check is not None:
        try:
            model.eval() # Set to eval mode
            with torch.no_grad():
                # Ensure batch is on the same device as model parameters
                # Note: if model is not explicitly moved to CUDA, it's on CPU.
                # Raw batch from DataLoader might be on CPU.
                input_batch = raw_batch_for_loss_check.to(next(model.parameters()).device)

                # model.forward() applies normalization internally if model.apply_normalization is True
                # The output is x_hat_processed (reconstruction in the potentially normalized space)
                x_hat_processed = model.forward(input_batch)

                # De-normalize if normalization was applied (mimicking training_step)
                x_hat_reconstructed = x_hat_processed
                if model.apply_normalization:
                    mu_de = model.norm_mu.to(x_hat_processed.device, dtype=x_hat_processed.dtype)
                    sigma_de = model.norm_sigma.to(x_hat_processed.device, dtype=x_hat_processed.dtype)
                    x_hat_reconstructed = x_hat_processed * sigma_de + mu_de
                
                initial_recon_loss = ((input_batch - x_hat_reconstructed) ** 2).mean()
                print(f"  Calculated initial (pre-train) reconstruction loss: {initial_recon_loss.item():.4f}")

                expected_range_norm = (1.0, 3.0) # Broader range
                expected_range_no_norm = (2.0, 6.0) # Broader range

                if model.apply_normalization:
                    print(f"  (With normalization, expected loss often ~{expected_range_norm[0]}-{expected_range_norm[1]})")
                    if not (expected_range_norm[0] <= initial_recon_loss.item() <= expected_range_norm[1]):
                         if initial_recon_loss.item() > 100 or torch.isnan(initial_recon_loss) or torch.isinf(initial_recon_loss):
                            print("  CRITICAL WARNING: Initial recon loss is extremely high or NaN/Inf!")
                         elif initial_recon_loss.item() < expected_range_norm[0]:
                            print("  INFO: Initial recon loss is lower than typical, could be fine or indicate weak reconstruction for other reasons.")
                         else: # Higher than expected but not astronomical
                            print("  WARNING: Initial recon loss is higher than typical for normalized inputs.")
                else:
                    print(f"  (Without normalization, expected loss often ~{expected_range_no_norm[0]}-{expected_range_no_norm[1]})")
                    if not (expected_range_no_norm[0] <= initial_recon_loss.item() <= expected_range_no_norm[1]):
                        if initial_recon_loss.item() > 100 or torch.isnan(initial_recon_loss) or torch.isinf(initial_recon_loss):
                             print("  CRITICAL WARNING: Initial recon loss is extremely high or NaN/Inf!")
                        elif initial_recon_loss.item() < expected_range_no_norm[0]:
                            print("  INFO: Initial recon loss is lower than typical.")
                        else:
                            print("  WARNING: Initial recon loss is higher than typical for unnormalized inputs.")
        except Exception as e:
            print(f"  ERROR during initial reconstruction loss check: {e}")
            import traceback
            traceback.print_exc()
    else:
        print("  Skipping initial recon loss check: Model or batch not available.")

    print("\n--- 5. Suggestions if Issues Found ---")
    print("  - Metadata issues (μ, σ ranges, d_model):")
    print(f"    -> Verify ACTS_DIR ('{ACTS_DIR}'), LAYER ({LAYER}), D_MODEL ({D_MODEL}) in sae.py.")
    print(f"    -> Consider deleting '{metadata_file}' and rerunning `sae.py` with `dm.setup(norm=True)` to regenerate stats.")
    print("  - Normalized batch stats not mean≈0, std≈1:")
    print("    -> Strongly indicates issues with loaded/computed μ, σ values or how they are applied.")
    print("    -> Check the `clamp(min=1e-6)` on `final_std_dev` in `normalize.py` if σ is too small.")
    print("  - Very high/NaN/Inf initial recon loss (especially with normalization ON):")
    print("    -> Critical sign. Usually `sigma` in metadata has very small values, causing division by near-zero.")
    print("    -> Carefully check 'σ range' from step 1 and any CRITICAL WARNINGS.")

    print("\nNormalization check script finished.")

if __name__ == "__main__":
    check_normalization_script()