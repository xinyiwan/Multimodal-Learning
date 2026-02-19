import matplotlib.pyplot as plt
import numpy as np
import torch
import einops

def reconstruct_and_visualize_patches(data, sample_id=None, is_batch=False, sample_idx=0, 
                                      patch_size=(8, 64, 64), spatial_layout=(10, 6, 8)):
    """
    Reconstruct patches from flattened format and visualize them spatially
    
    Args:
        data: Sample or batch data
        sample_id: Sample ID for title
        is_batch: Whether data is from DataLoader
        sample_idx: Which sample in batch to visualize
        patch_size: Tuple of (patch_z, patch_y, patch_x)
        spatial_layout: Tuple of (z_patches, y_patches, x_patches)
    """
    # Extract patches and masks
    if is_batch:
        if "data.input.img.tumor3d.patches" in data:
            patches_dict = data["data.input.img.tumor3d.patches"]
            if isinstance(patches_dict, dict):
                patches = patches_dict["patches"][sample_idx]
                patches_mask = patches_dict["patches_mask"][sample_idx]
            else:
                patches = data["data.input.img.tumor3d.patches"][sample_idx]
                patches_mask = data.get("model.embed_mask_b", torch.ones_like(patches))[sample_idx]
        
        # Convert tensors to numpy
        if torch.is_tensor(patches):
            patches = patches.cpu().numpy()
        if torch.is_tensor(patches_mask):
            patches_mask = patches_mask.cpu().numpy()
    else:
        patches = data["data.input.img.tumor3d.patches"]["patches"]
        patches_mask = data["data.input.img.tumor3d.patches"]["patches_mask"]
    
    print(f"Patches shape: {patches.shape}")
    print(f"Patches mask shape: {patches_mask.shape}")
    
    # Reshape patches if they're flattened
    if patches.ndim == 2:  # Flattened: (n_patches, patch_features)
        n_patches, patch_features = patches.shape
        patch_z, patch_y, patch_x = patch_size
        z_patches, y_patches, x_patches = spatial_layout
        
        # Calculate expected features
        expected_features = patch_z * patch_y * patch_x
        if patch_features != expected_features:
            print(f"Warning: Patch features {patch_features} != expected {expected_features}")
        
        # Reshape patches to 3D spatial layout
        patches_reshaped = einops.rearrange(
            patches,
            '(z y x) (pz py px) -> z y x pz py px',
            z=z_patches, y=y_patches, x=x_patches,
            pz=patch_z, py=patch_y, px=patch_x
        )
        
        # Reshape masks if they're flattened too
        if patches_mask.ndim == 1:  # Single value per patch
            patches_mask_reshaped = einops.rearrange(
                patches_mask,
                '(z y x) -> z y x',
                z=z_patches, y=y_patches, x=x_patches
            )
        else:
            patches_mask_reshaped = patches_mask
        
        print(f"Reshaped patches: {patches_reshaped.shape}")
        print(f"Reshaped mask: {patches_mask_reshaped.shape}")
    else:
        # Already in spatial format
        patches_reshaped = patches
        patches_mask_reshaped = patches_mask
    
    # Visualize 3D patch mask
    visualize_3d_patch_mask(patches_reshaped, patches_mask_reshaped, sample_id, patch_size)
    
    # Visualize individual unmasked patches
    visualize_unmasked_patches(patches_reshaped, patches_mask_reshaped, sample_id, patch_size)
    
    # Visualize reconstructed volume slices
    visualize_reconstructed_slices(patches_reshaped, patches_mask_reshaped, sample_id, patch_size)

def visualize_3d_patch_mask(patches, patches_mask, sample_id, patch_size):
    """
    3D visualization of patch mask values
    """
    # Aggregate mask values per patch
    if patches_mask.ndim == 6:  # Full 3D mask per patch
        mask_values = patches_mask.mean(axis=(-3, -2, -1))
    elif patches_mask.ndim == 3:  # Single value per patch
        mask_values = patches_mask
    else:
        mask_values = patches_mask
    
    # Create 3D grid
    z_patches, y_patches, x_patches = mask_values.shape[:3]
    z_coords, y_coords, x_coords = np.meshgrid(
        range(z_patches), 
        range(y_patches), 
        range(x_patches), 
        indexing='ij'
    )
    
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # Flatten for scatter plot
    scatter = ax.scatter(
        x_coords.flatten(), 
        y_coords.flatten(), 
        z_coords.flatten(),
        c=mask_values.flatten(),
        cmap='RdYlGn',  # Red (masked) to Green (unmasked)
        s=50,
        alpha=0.7,
        edgecolors='k',
        vmin=0,
        vmax=1
    )
    
    ax.set_xlabel('X Patch Index')
    ax.set_ylabel('Y Patch Index')
    ax.set_zlabel('Z Patch Index')
    
    title = f'3D Patch Mask - Valid Patches in Green (mask > 0.5)'
    if sample_id:
        title = f'{sample_id}\n' + title
    ax.set_title(title)
    
    plt.colorbar(scatter, ax=ax, label='Mask Value')
    plt.tight_layout()
    plt.show()
    
    # Print statistics
    valid_patches = (mask_values > 0.5).sum()
    total_patches = mask_values.size
    print(f"\nMask Statistics:")
    print(f"  Valid patches (mask > 0.5): {valid_patches}/{total_patches} ({valid_patches/total_patches*100:.1f}%)")
    print(f"  Mask range: [{mask_values.min():.3f}, {mask_values.max():.3f}]")
    print(f"  Mean mask value: {mask_values.mean():.3f}")

def visualize_unmasked_patches(patches, patches_mask, sample_id, patch_size, max_patches=12):
    """
    Visualize individual patches that are not masked
    """
    # Get mask values per patch
    if patches_mask.ndim == 6:
        mask_values = patches_mask.mean(axis=(-3, -2, -1))
    elif patches_mask.ndim == 3:
        mask_values = patches_mask
    else:
        mask_values = patches_mask
    
    # Find unmasked patches
    unmasked_indices = np.argwhere(mask_values > 0.5)
    
    if len(unmasked_indices) == 0:
        print("No unmasked patches found!")
        return
    
    print(f"\nFound {len(unmasked_indices)} unmasked patches")
    
    # Limit number of patches to visualize
    if len(unmasked_indices) > max_patches:
        # Select a diverse set of patches
        step = len(unmasked_indices) // max_patches
        selected_indices = unmasked_indices[::step][:max_patches]
    else:
        selected_indices = unmasked_indices
    
    # Create figure
    n_cols = min(4, len(selected_indices))
    n_rows = (len(selected_indices) + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4*n_cols, 4*n_rows))
    
    if n_rows == 1 and n_cols == 1:
        axes = np.array([axes])
    axes = axes.flatten()
    
    for idx, (ax, (z_idx, y_idx, x_idx)) in enumerate(zip(axes, selected_indices)):
        # Extract patch
        patch = patches[z_idx, y_idx, x_idx]
        mask_val = mask_values[z_idx, y_idx, x_idx]
        
        # For 3D patches, show middle slice
        if patch.ndim == 3:
            patch_slice = patch[patch.shape[0] // 2]
        else:
            patch_slice = patch
        
        ax.imshow(patch_slice, cmap='gray', vmin=-150, vmax=250)
        ax.set_title(f'Patch ({z_idx},{y_idx},{x_idx})\nmask={mask_val:.3f}')
        ax.axis('off')
        
        # Highlight if it's a valid patch
        if mask_val > 0.5:
            for spine in ax.spines.values():
                spine.set_edgecolor('green')
                spine.set_linewidth(3)
    
    # Hide unused axes
    for ax in axes[len(selected_indices):]:
        ax.axis('off')
    
    title = f'Unmasked Patches (mask > 0.5)'
    if sample_id:
        title = f'{sample_id}\n' + title
    fig.suptitle(title, fontsize=14)
    plt.tight_layout()
    plt.show()

def visualize_reconstructed_slices(patches, patches_mask, sample_id, patch_size):
    """
    Reconstruct the volume and visualize slices
    """
    # Reconstruct full volume
    patch_z, patch_y, patch_x = patch_size
    z_patches, y_patches, x_patches = patches.shape[:3]
    
    # Reconstruct volume
    volume = einops.rearrange(
        patches,
        'z y x pz py px -> (z pz) (y py) (x px)',
        pz=patch_z, py=patch_y, px=patch_x
    )
    
    # Create mask volume
    if patches_mask.ndim == 3:
        # Expand mask to match patch size
        mask_volume = einops.repeat(
            patches_mask,
            'z y x -> (z pz) (y py) (x px)',
            pz=patch_z, py=patch_y, px=patch_x
        )
    else:
        mask_volume = einops.rearrange(
            patches_mask,
            'z y x pz py px -> (z pz) (y py) (x px)',
            pz=patch_z, py=patch_y, px=patch_x
        )
    
    print(f"\nReconstructed volume shape: {volume.shape}")
    print(f"Reconstructed mask shape: {mask_volume.shape}")
    
    # Visualize slices
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()
    
    # Select slices to show
    n_slices = volume.shape[0]
    slice_indices = np.linspace(0, n_slices-1, 6, dtype=int)
    
    for ax, slice_idx in zip(axes, slice_indices):
        # Original slice
        img = volume[slice_idx]
        mask_slice = mask_volume[slice_idx]
        
        # Create overlay
        ax.imshow(img, cmap='gray', vmin=-150, vmax=250)
        
        # Overlay mask
        mask_overlay = np.ma.masked_where(mask_slice < 0.5, mask_slice)
        ax.imshow(mask_overlay, cmap='Reds', alpha=0.3, vmin=0, vmax=1)
        
        ax.set_title(f'Slice {slice_idx}\nValid voxels: {(mask_slice > 0.5).sum()/mask_slice.size*100:.1f}%')
        ax.axis('off')
    
    title = f'Reconstructed Volume Slices with Mask Overlay (Red = masked)'
    if sample_id:
        title = f'{sample_id}\n' + title
    fig.suptitle(title, fontsize=14)
    plt.tight_layout()
    plt.show()
    
    # Show histogram of mask values
    plt.figure(figsize=(10, 4))
    
    plt.subplot(1, 2, 1)
    plt.hist(mask_volume.flatten(), bins=50, alpha=0.7, edgecolor='black')
    plt.axvline(x=0.5, color='red', linestyle='--', label='Threshold=0.5')
    plt.xlabel('Mask Value')
    plt.ylabel('Frequency')
    plt.title('Mask Value Distribution')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    mask_binary = (mask_volume > 0.5).astype(int)
    
    # Project through Z axis
    projection = mask_binary.mean(axis=0)
    im = plt.imshow(projection, cmap='YlOrRd', vmin=0, vmax=1)
    plt.colorbar(im, label='Fraction of valid slices')
    plt.title('Valid Voxels Projection (Y-X plane)')
    plt.xlabel('X')
    plt.ylabel('Y')
    
    plt.suptitle(f'{sample_id} - Mask Analysis' if sample_id else 'Mask Analysis')
    plt.tight_layout()
    plt.show()

def visualize_batch_patches(batch, patch_size=(8, 64, 64), spatial_layout=(10, 6, 8)):
    """
    Visualize patches for all samples in a batch
    """
    # Get batch size
    if "data.sample_id" in batch:
        batch_size = len(batch["data.sample_id"])
        sample_ids = batch["data.sample_id"]
    else:
        # Estimate from first tensor
        first_key = next(iter(batch.keys()))
        if torch.is_tensor(batch[first_key]):
            batch_size = batch[first_key].shape[0]
            sample_ids = [f"Sample_{i}" for i in range(batch_size)]
        else:
            batch_size = 1
            sample_ids = ["Sample_0"]
    
    print(f"Visualizing {batch_size} samples in batch")
    
    for i in range(min(batch_size, 3)):  # Limit to first 3 samples
        print(f"\n{'='*60}")
        print(f"Sample {i}: {sample_ids[i] if i < len(sample_ids) else 'Unknown'}")
        print(f"{'='*60}")
        
        reconstruct_and_visualize_patches(
            batch, 
            sample_id=sample_ids[i] if i < len(sample_ids) else f"Sample_{i}",
            is_batch=True,
            sample_idx=i,
            patch_size=patch_size,
            spatial_layout=spatial_layout
        )

def debug_batch_structure(batch):
    """
    Print detailed information about batch structure
    """
    print("=" * 60)
    print("BATCH STRUCTURE DEBUG INFO")
    print("=" * 60)
    
    for key, value in batch.items():
        if torch.is_tensor(value):
            print(f"{key}: Tensor shape {tuple(value.shape)}, dtype={value.dtype}")
        elif isinstance(value, dict):
            print(f"{key}: Dict with keys {list(value.keys())}")
            for subkey, subvalue in value.items():
                if torch.is_tensor(subvalue):
                    print(f"  {subkey}: Tensor shape {tuple(subvalue.shape)}")
                else:
                    print(f"  {subkey}: Type {type(subvalue)}")
        else:
            print(f"{key}: Type {type(value)}")
    
    print("=" * 60)


