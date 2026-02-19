from OS import OSDataset
from OS_CrossAttention_Optuna_Imaging import build_dataloaders_from_ids
import torch

# Add this test function
def test_dataloader():
    """Test the DataLoader with minimal configuration"""
    # Simple config for testing
    test_cfg = {
        "patch_size": (8, 64, 64),
        "emb_dim": 128,
        "lr": 3e-4,
        "wd": 1e-4,
        "depth_b": 2,
        "heads_b": 4,
        "mlp_layers": "single",
        "mask_pad_thresh": 0.8,
        "imaging_aug_deg": 0,
        "batch_size": 2,
        "num_workers": 0,
        "clinical_aug": 0,
    }
    
    # Use your actual data paths
    data_paths = {
        "img": "/projects/prjs1779/Osteosarcoma/exp_data/T2W_FS/v1/input/img",
        "seg": "/projects/prjs1779/Osteosarcoma/exp_data/T2W_FS/v1/input/seg",
        "csv": "/projects/prjs1779/Osteosarcoma/exp_data/T2W_FS/v1/clinical_features_with_Huvos.csv",
    }
    
    # Get all sample IDs
    all_ids = OSDataset.sample_ids(data_paths["seg"])
    print(f"Total samples: {len(all_ids)}")
    
    # Split for testing
    train_ids = all_ids[:4]
    val_ids = all_ids[4:6]
    
    try:
        train_dl, val_dl = build_dataloaders_from_ids(
            data_paths=data_paths,
            cfg=test_cfg,
            largest_tumor=(80, 347, 498),
            train_ids=train_ids,
            val_ids=val_ids,
        )
        
        print("DataLoaders created successfully!")
        
        # Try to get one batch
        print("\nTrying to get one batch from train_dl...")
        batch = next(iter(train_dl))
        print(f"Batch keys: {list(batch.keys())}")
        
        for key, value in batch.items():
            if torch.is_tensor(value):
                print(f"  {key}: shape={value.shape}, dtype={value.dtype}")
        
        return True
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return False

# Run this test before starting Optuna
if __name__ == "__main__":
    success = test_dataloader()
    if success:
        print("\nDataLoader test PASSED!")
    else:
        print("\nDataLoader test FAILED!")