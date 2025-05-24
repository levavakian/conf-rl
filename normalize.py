import json
import torch
import glob
from pathlib import Path
import logging

# Configure logging for better feedback
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

def get_or_compute_normalization_stats(
    acts_dir: str,
    d_model: int,
    file_glob_pattern: str = "layer*_shard*.pt"
) -> tuple[torch.Tensor | None, torch.Tensor | None]:
    """
    Computes or retrieves normalization statistics (mean and std_dev) for activation shards.

    Checks for a 'normalization_metadata.json' in acts_dir. If it exists and is up-to-date
    (based on the list of shard files and d_model), it returns the stored statistics.
    Otherwise, it computes the statistics from all shard files matching the pattern,
    saves them to the metadata file, and returns them.

    Args:
        acts_dir: Directory containing the activation shard files.
        d_model: The expected dimension of the activation vectors.
        file_glob_pattern: Glob pattern to find shard files within acts_dir
                           (e.g., "layer0_shard*.pt", "shard_*.pt").

    Returns:
        A tuple (mu, sigma) of torch.Tensors representing the mean and standard deviation.
        Returns (None, None) if no data is found or stats cannot be computed.
    """
    acts_dir_path = Path(acts_dir)
    if not acts_dir_path.is_dir():
        logging.error(f"Activations directory not found: {acts_dir_path}")
        return None, None
        
    metadata_path = acts_dir_path / "normalization_metadata.json"

    current_shard_full_paths = sorted(list(acts_dir_path.glob(file_glob_pattern)))
    if not current_shard_full_paths:
        logging.warning(f"No shard files found in '{acts_dir_path}' matching pattern '{file_glob_pattern}'.")
        return None, None

    # Use relative paths for storing and comparison in metadata.json
    current_shard_relative_paths_str = sorted([str(p.relative_to(acts_dir_path)) for p in current_shard_full_paths])

    if metadata_path.exists():
        try:
            logging.info(f"Found existing metadata file: {metadata_path}")
            metadata_content = json.loads(metadata_path.read_text())
            stored_shard_files = sorted(metadata_content.get("shard_files", []))
            
            # Compare sets of file paths for robustness against order changes in glob
            if set(current_shard_relative_paths_str) == set(stored_shard_files):
                mu_list = metadata_content.get("mu")
                sigma_list = metadata_content.get("sigma")
                stored_d_model = metadata_content.get("d_model")

                if mu_list is not None and sigma_list is not None and stored_d_model == d_model:
                    logging.info("Metadata is up-to-date. Loading stored statistics.")
                    mu = torch.tensor(mu_list, dtype=torch.float32)
                    sigma = torch.tensor(sigma_list, dtype=torch.float32)
                    
                    if mu.shape == (d_model,) and sigma.shape == (d_model,):
                        return mu, sigma
                    else:
                        logging.warning(
                            f"Stored statistics dimensions mismatch. Expected ({d_model},), "
                            f"got mu: {mu.shape}, sigma: {sigma.shape}. Recomputing."
                        )
                elif stored_d_model != d_model:
                    logging.info(f"d_model in metadata ({stored_d_model}) differs from requested ({d_model}). Recomputing.")
                else:
                    logging.info("Stored statistics are incomplete in metadata. Recomputing.")
            else:
                logging.info("Shard file list has changed. Recomputing statistics.")
                logging.debug(f"Current files count: {len(current_shard_relative_paths_str)}, Stored files count: {len(stored_shard_files)}")
                logging.debug(f"First few current files: {current_shard_relative_paths_str[:3]}, First few stored files: {stored_shard_files[:3]}")

        except json.JSONDecodeError:
            logging.warning(f"Error decoding {metadata_path}. Recomputing statistics.")
        except Exception as e:
            logging.warning(f"Unexpected error reading {metadata_path}: {e}. Recomputing statistics.")

    logging.info(f"Computing normalization statistics from {len(current_shard_full_paths)} shards...")
    count = 0
    mean_acc = torch.zeros(d_model, dtype=torch.float64)  # Use float64 for accumulation
    m2_acc = torch.zeros(d_model, dtype=torch.float64)   # Welford's M2 accumulator

    for shard_full_path in current_shard_full_paths:
        logging.debug(f"Processing shard: {shard_full_path.name}")
        try:
            data = torch.load(shard_full_path, map_location="cpu")
            if "acts" not in data or not isinstance(data["acts"], torch.Tensor):
                logging.warning(f"Skipping shard {shard_full_path.name}: 'acts' key missing or not a Tensor.")
                continue
            
            activations = data["acts"].to(torch.float64)

            if activations.ndim == 1: # Handle case of a single activation vector in a file
                if activations.shape[0] == d_model:
                    activations = activations.unsqueeze(0)
                else:
                    logging.warning(f"Skipping shard {shard_full_path.name}: single activation vector has wrong dimension {activations.shape[0]}, expected {d_model}.")
                    continue
            
            if activations.ndim != 2 or activations.shape[1] != d_model:
                logging.warning(f"Skipping shard {shard_full_path.name}: activations have incorrect shape {activations.shape}, expected (N, {d_model}).")
                continue

            if activations.shape[0] == 0:
                logging.debug(f"Shard {shard_full_path.name} is empty.")
                continue

            # Welford's algorithm update step for each sample in the current shard
            for i in range(activations.shape[0]):
                x_sample = activations[i, :]
                count += 1
                delta = x_sample - mean_acc
                mean_acc += delta / count
                delta2 = x_sample - mean_acc # New delta with updated mean
                m2_acc += delta * delta2
        
        except FileNotFoundError: # Should be caught by acts_dir_path.is_dir() and glob check
            logging.error(f"Shard file not found during processing: {shard_full_path}.")
            continue
        except Exception as e:
            logging.warning(f"Error loading or processing shard {shard_full_path.name}: {e}. Skipping shard.")
            continue

    if count < 2:
        logging.error(f"Not enough data points (found {count}) across all shards to compute variance reliably. Min 2 required.")
        return None, None

    final_mean = mean_acc.to(torch.float32)
    # Use (count - 1) for unbiased sample variance; if count is population, use count.
    # For large N, difference is small. Standard practice for sample variance.
    variance = m2_acc / (count - 1) 
    final_std_dev = torch.sqrt(variance).clamp(min=1e-6).to(torch.float32) # clamp for numerical stability

    logging.info(f"Computed statistics from {count} samples: mean shape {final_mean.shape}, std_dev shape {final_std_dev.shape}")

    metadata_to_save = {
        "shard_files": current_shard_relative_paths_str,
        "d_model": d_model,
        "mu": final_mean.tolist(),
        "sigma": final_std_dev.tolist(),
        "count": count
    }
    try:
        metadata_path.write_text(json.dumps(metadata_to_save, indent=4))
        logging.info(f"Saved new normalization statistics to {metadata_path}")
    except Exception as e:
        logging.error(f"Failed to save metadata to {metadata_path}: {e}")

    return final_mean, final_std_dev

if __name__ == "__main__":
    # Create dummy data for testing
    dummy_acts_dir_name = "dummy_acts_dir_normalize_test"
    dummy_acts_dir = Path(dummy_acts_dir_name)
    dummy_acts_dir.mkdir(exist_ok=True)
    
    D_MODEL_TEST = 3
    N_SHARDS_TEST = 2
    SAMPLES_PER_SHARD_TEST = 50 # Increased samples for more stable stats

    # Clean up old metadata and shards if any
    meta_file_test = dummy_acts_dir / "normalization_metadata.json"
    if meta_file_test.exists():
        meta_file_test.unlink()
    for old_shard in dummy_acts_dir.glob("layer0_shard*.pt"):
        old_shard.unlink()

    # Create dummy shards
    all_test_data_list = []
    for i in range(N_SHARDS_TEST):
        # Make data with somewhat predictable mean and std
        shard_data = torch.randn(SAMPLES_PER_SHARD_TEST, D_MODEL_TEST) * (i + 1.5) + (i * 2.0)
        all_test_data_list.append(shard_data)
        torch.save({"acts": shard_data}, dummy_acts_dir / f"layer0_shard{i}.pt")
    
    combined_test_data = torch.cat(all_test_data_list, dim=0)
    expected_mean = combined_test_data.mean(dim=0)
    expected_std = combined_test_data.std(dim=0, unbiased=True) # unbiased=True matches (count-1)

    print(f"--- First call (should compute) ---")
    mu1, sigma1 = get_or_compute_normalization_stats(str(dummy_acts_dir), D_MODEL_TEST, "layer0_shard*.pt")
    if mu1 is not None:
        print("Mu (1st call):     ", mu1)
        print("Sigma (1st call):  ", sigma1)
        print("Expected Mu:       ", expected_mean)
        print("Expected Sigma:    ", expected_std)
        if torch.allclose(mu1, expected_mean, atol=1e-5) and torch.allclose(sigma1, expected_std, atol=1e-5):
            print("SUCCESS: Computed stats match expected stats.")
        else:
            print("NOTE: Computed stats deviation from torch.std/mean might occur due to online algorithm precision with float32 final.")


    print(f"\n--- Second call (should load from metadata) ---")
    mu2, sigma2 = get_or_compute_normalization_stats(str(dummy_acts_dir), D_MODEL_TEST, "layer0_shard*.pt")
    if mu2 is not None:
        print("Mu (2nd call):     ", mu2)
        print("Sigma (2nd call):  ", sigma2)
        if mu1 is not None and torch.allclose(mu1, mu2) and torch.allclose(sigma1, sigma2):
            print("SUCCESS: Stats from 1st and 2nd call are identical.")
        else:
            print("FAILURE: Stats differ between calls or first call failed.")

    # Test with an added shard
    print(f"\n--- Adding a new shard ---")
    new_shard_data = torch.randn(SAMPLES_PER_SHARD_TEST, D_MODEL_TEST) * 10.0 - 5.0
    all_test_data_list.append(new_shard_data)
    torch.save({"acts": new_shard_data}, dummy_acts_dir / f"layer0_shard{N_SHARDS_TEST}.pt")
    
    combined_test_data_updated = torch.cat(all_test_data_list, dim=0)
    expected_mean_updated = combined_test_data_updated.mean(dim=0)
    expected_std_updated = combined_test_data_updated.std(dim=0, unbiased=True)

    print(f"--- Third call (should recompute) ---")
    mu3, sigma3 = get_or_compute_normalization_stats(str(dummy_acts_dir), D_MODEL_TEST, "layer0_shard*.pt")
    if mu3 is not None:
        print("Mu (3rd call):     ", mu3)
        print("Sigma (3rd call):  ", sigma3)
        print("Expected Mu (new): ", expected_mean_updated)
        print("Expected Sigma (new):", expected_std_updated)
        if mu2 is not None and (not torch.allclose(mu2, mu3, atol=1e-5) or not torch.allclose(sigma2, sigma3, atol=1e-5)):
             print("SUCCESS: Stats recomputed and are different from previous.")
        else:
            print("FAILURE: Stats not recomputed or are unexpectedly the same as previous.")
        if torch.allclose(mu3, expected_mean_updated, atol=1e-5) and torch.allclose(sigma3, expected_std_updated, atol=1e-5):
            print("SUCCESS: Recomputed stats match new expected stats.")
        else:
            print("NOTE: Recomputed stats deviation from torch.std/mean might occur.")
            
    # Test with d_model mismatch
    print(f"\n--- Fourth call with different d_model (should recompute, but will fail if data d_model is fixed) ---")
    # This will recompute because d_model changed, but then fail processing shards if their d_model is D_MODEL_TEST
    # For a more robust test, one might create new data with D_MODEL_TEST+1
    mu4, sigma4 = get_or_compute_normalization_stats(str(dummy_acts_dir), D_MODEL_TEST + 1, "layer0_shard*.pt")
    if mu4 is None and sigma4 is None:
        print(f"SUCCESS: Call with d_model={D_MODEL_TEST+1} returned None, None as expected (data has d_model={D_MODEL_TEST}).")
    else:
        print(f"FAILURE: Call with d_model={D_MODEL_TEST+1} did not behave as expected.")


    # Clean up dummy dir (optional, uncomment to use)
    # import shutil
    # shutil.rmtree(dummy_acts_dir)
    # print(f"\nCleaned up {dummy_acts_dir}")
