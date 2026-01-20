import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import SimpleITK as sitk
import nibabel as nib
from fuse.data.ops.op_base import OpBase
from fuse.utils import NDict
import torch
import torch.nn as nn
import random
from typing import Tuple, Optional, Sequence, List, Dict, Union
from scipy.ndimage import rotate
from einops import rearrange


class OpGISTLoadImage(OpBase):
    """
    Load a NIfTI image file (.nii.gz) from a given directory and convert it to [Z, Y, X] format.
    Stores the image array, original spacing, and original size in sample_dict.

    - key_in: str → filename key (e.g., 'file.nii.gz')
    - key_out: str → key under which the [Z, Y, X] array is stored
    """

    def __init__(self, dir_path: str):
        super().__init__()
        self._dir_path = dir_path

    def __call__(self, sample_dict: NDict, key_in: str, key_out: str) -> NDict:
        img_filename = os.path.join(self._dir_path, sample_dict[key_in])

        if not img_filename.endswith('.nii.gz'):
            raise ValueError(f"Expected a .nii.gz file, but got: {img_filename}")

        # --- Load image in [X, Y, Z] ---
        img = nib.load(img_filename)
        img_np = img.get_fdata()  # shape: [X, Y, Z]

        # --- Transpose to [Z, Y, X] ---
        img_np = np.transpose(img_np, axes=(2, 1, 0))  # shape: [Z, Y, X]

        # --- Extract spacing and size in [Z, Y, X] ---
        spacing_xyz = img.header.get_zooms()  # [X, Y, Z]
        original_spacing = np.array(spacing_xyz[::-1], dtype=np.float32)  # [Z, Y, X]
        original_size = np.array(img_np.shape, dtype=np.int32)            # [Z, Y, X]

        # --- Save into sample_dict ---
        sample_dict[key_out] = img_np.copy()
        sample_dict["data.input.img.original_spacing"] = original_spacing
        sample_dict["data.input.img.original_size"] = original_size

        return sample_dict


class OpCastLabelToFloat(OpBase):
    """
    Casts an integer label (e.g., 0 or 1) to float (0.0 or 1.0).
    Required for binary classification with BCEWithLogitsLoss.
    """

    def __init__(self):
        super().__init__()

    def __call__(self, sample_dict: NDict, key_in: str, key_out: Optional[str] = None) -> NDict:
        if key_out is None:
            key_out = key_in
        label = sample_dict[key_in]
        sample_dict[key_out] = float(label)
        return sample_dict


class GISTDataUtils:
    def __init__(self, dataset, img_dir, seg_dir, modality):
        """
        Initialize the utility class with a dataset and data directories.

        Args:
            dataset (DatasetDefault): The dataset containing CT scan data.
            img_dir (str): Path to the directory containing CT scan images.
            seg_dir (str): Path to the directory containing segmentation masks.
        """
        self.dataset = dataset
        self.img_dir = img_dir
        self.seg_dir = seg_dir
        self.modality = modality
        self.unique_spacings = []

    def show_mid_slice(self, patient_id, ct_key="data.input.img", seg_key="data.input.seg"):
        """
        Display the mid-slice of a CT scan for a given patient ID, rotated by -90 degrees,
        and overlay the segmentation mask.

        Args:
            patient_id (str): The sample ID of the patient.
            ct_key (str): The key corresponding to the CT scan in the dataset dictionary.
            seg_key (str): The key corresponding to the segmentation mask in the dataset dictionary.
        """
        sample = self.dataset.getitem(patient_id)
        ct_scan = sample[ct_key]
        seg_mask = sample[seg_key]

        if hasattr(ct_scan, "numpy"):
            ct_scan = ct_scan.numpy()
        if hasattr(seg_mask, "numpy"):
            seg_mask = seg_mask.numpy()

        mid_slice = ct_scan.shape[2] // 2
        ct_slice = np.rot90(ct_scan[:, :, mid_slice], k=-1)
        seg_slice = np.rot90(seg_mask[:, :, mid_slice], k=-1)

        plt.figure(figsize=(6, 6))
        plt.imshow(ct_slice, cmap="gray")
        plt.imshow(seg_slice, cmap="jet", alpha=0.3)
        plt.show()

    def check_ct_seg_integrity(self):
        """
        Perform integrity checks on the dataset including voxel spacing, orientation,
        shape consistency, and presence of non-empty tumor masks.
        """
        unique_orientations = set()

        for sample in self.dataset:
            ct_img_path = sample["data.input.img_path"]
            seg_img_path = sample["data.input.seg_path"]

            ct_img = sitk.ReadImage(os.path.join(self.img_dir, ct_img_path))
            seg_img = sitk.ReadImage(os.path.join(self.seg_dir, seg_img_path))

            ct_spacing = ct_img.GetSpacing()
            seg_spacing = seg_img.GetSpacing()

            ct_direction = ct_img.GetDirection()
            seg_direction = seg_img.GetDirection()

            self.unique_spacings.append(ct_spacing)
            unique_orientations.add(ct_direction)

            if ct_spacing != seg_spacing:
                print(f"⚠️ Mismatch in voxel spacing for {ct_img_path}: {ct_spacing} vs {seg_spacing}")

            if ct_direction != seg_direction:
                print(f"⚠️ Mismatch in orientation for {ct_img_path}: {ct_direction} vs {seg_direction}")

            if ct_img.GetSize() != seg_img.GetSize():
                print(
                    f"⚠️ Shape mismatch: {ct_img_path} has size {ct_img.GetSize()}, but segmentation is {seg_img.GetSize()}")

            seg_array = sitk.GetArrayFromImage(seg_img)
            if np.sum(seg_array) == 0:
                print(f"⚠️ Empty mask detected: {seg_img_path}")

        print("\n✅ Unique voxel spacings found:", set(self.unique_spacings))
        print("✅ Unique orientations found:", unique_orientations)


    def find_mean_median_spacing(self):
        """
        Calculate the mean and median voxel spacing from all CT scans in the dataset.
        """
        spacings = []

        for sample in self.dataset:
            ct_img_path = sample["data.input.img_path"]
            ct_img = sitk.ReadImage(os.path.join(self.img_dir, ct_img_path))
            spacings.append(ct_img.GetSpacing())

        spacings = np.array(spacings)  # Convert to NumPy array for easy calculations
        mean_spacing = np.mean(spacings, axis=0)
        median_spacing = np.median(spacings, axis=0)

        min_spacing = np.min(spacings, axis=0)
        max_spacing = np.max(spacings, axis=0)

        print(f"Min Spacing:{min_spacing}, Max Spacing:{max_spacing}")
        print(f"Mean Spacing: {mean_spacing}")
        print(f"Median Spacing: {median_spacing}")
    
    def get_max_tumor_size(self) -> Tuple[Tuple[int, int, int], Dict]:
        """
        Simple version with basic plots including quartiles.
        """
        stats_dict = {}
        largest_tumor = (0, 0, 0)
        max_volume = 0
        
        # Collect data
        z_sizes, y_sizes, x_sizes, volumes = [], [], [], []
        
        for i, sample in enumerate(self.dataset):
            try:
                # Try to get the cropped tumor
                cropped_tumour = sample["data.input.img.tumor3d"]
                tumor_shape = cropped_tumour.shape  # (z, y, x)
                volume = tumor_shape[0] * tumor_shape[1] * tumor_shape[2]
                
                # Store data
                z_sizes.append(tumor_shape[0])
                y_sizes.append(tumor_shape[1])
                x_sizes.append(tumor_shape[2])
                volumes.append(volume)
                
                # Update largest tumor
                if volume > max_volume:
                    max_volume = volume
                    largest_tumor = tumor_shape
                    
            except (KeyError, ValueError) as e:
                # Skip samples without tumors or with errors
                print(f"Skipping sample {i}: {str(e)}")
                continue
        
        if not volumes:  # No tumors found at all
            print("No tumors found in any samples!")
            return (0, 0, 0), {}
                
        # Calculate quartiles for each dimension
        z_quartiles = np.percentile(z_sizes, [25, 75])
        y_quartiles = np.percentile(y_sizes, [25, 75])
        x_quartiles = np.percentile(x_sizes, [25, 75])
        vol_quartiles = np.percentile(volumes, [25, 75])
        
        # Simple print output
        print(f"Largest tumor (z,y,x): {largest_tumor}")
        print(f"Largest volume: {max_volume}")
        print(f"\nQuartile Statistics:")
        print(f"Z - Q25: {z_quartiles[0]:.1f}, Q75: {z_quartiles[1]:.1f}")
        print(f"Y - Q25: {y_quartiles[0]:.1f}, Q75: {y_quartiles[1]:.1f}")
        print(f"X - Q25: {x_quartiles[0]:.1f}, Q75: {x_quartiles[1]:.1f}")
        print(f"Volume - Q25: {vol_quartiles[0]:.1f}, Q75: {vol_quartiles[1]:.1f}")
        
        # Create simple plots
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        
        # Plot 1: Z dimension
        axes[0, 0].hist(z_sizes, bins=15, color='skyblue', edgecolor='black', alpha=0.7)
        axes[0, 0].axvline(np.median(z_sizes), color='red', linestyle='--', label=f'Median: {np.median(z_sizes):.1f}')
        axes[0, 0].axvline(z_quartiles[0], color='orange', linestyle=':', label=f'Q25: {z_quartiles[0]:.1f}')
        axes[0, 0].axvline(z_quartiles[1], color='purple', linestyle=':', label=f'Q75: {z_quartiles[1]:.1f}')
        axes[0, 0].set_title('Z Dimension Distribution')
        axes[0, 0].set_xlabel('Size (slices)')
        axes[0, 0].set_ylabel('Count')
        axes[0, 0].legend()
        
        # Plot 2: Y dimension
        axes[0, 1].hist(y_sizes, bins=15, color='lightgreen', edgecolor='black', alpha=0.7)
        axes[0, 1].axvline(np.median(y_sizes), color='red', linestyle='--', label=f'Median: {np.median(y_sizes):.1f}')
        axes[0, 1].axvline(y_quartiles[0], color='orange', linestyle=':', label=f'Q25: {y_quartiles[0]:.1f}')
        axes[0, 1].axvline(y_quartiles[1], color='purple', linestyle=':', label=f'Q75: {y_quartiles[1]:.1f}')
        axes[0, 1].set_title('Y Dimension Distribution')
        axes[0, 1].set_xlabel('Size (pixels)')
        axes[0, 1].set_ylabel('Count')
        axes[0, 1].legend()
        
        # Plot 3: X dimension
        axes[1, 0].hist(x_sizes, bins=15, color='lightcoral', edgecolor='black', alpha=0.7)
        axes[1, 0].axvline(np.median(x_sizes), color='red', linestyle='--', label=f'Median: {np.median(x_sizes):.1f}')
        axes[1, 0].axvline(x_quartiles[0], color='orange', linestyle=':', label=f'Q25: {x_quartiles[0]:.1f}')
        axes[1, 0].axvline(x_quartiles[1], color='purple', linestyle=':', label=f'Q75: {x_quartiles[1]:.1f}')
        axes[1, 0].set_title('X Dimension Distribution')
        axes[1, 0].set_xlabel('Size (pixels)')
        axes[1, 0].set_ylabel('Count')
        axes[1, 0].legend()
        
        # Plot 4: Volume distribution
        axes[1, 1].hist(volumes, bins=15, color='gold', edgecolor='black', alpha=0.7)
        axes[1, 1].axvline(np.median(volumes), color='red', linestyle='--', label=f'Median: {np.median(volumes):.0f}')
        axes[1, 1].axvline(vol_quartiles[0], color='orange', linestyle=':', label=f'Q25: {vol_quartiles[0]:.0f}')
        axes[1, 1].axvline(vol_quartiles[1], color='purple', linestyle=':', label=f'Q75: {vol_quartiles[1]:.0f}')
        axes[1, 1].set_title('Volume Distribution')
        axes[1, 1].set_xlabel('Volume (voxels)')
        axes[1, 1].set_ylabel('Count')
        axes[1, 1].legend()
        
        plt.tight_layout()
        plt.show()
        plt.savefig(f'/projects/prjs1779/Osteosarcoma/ViT_OSdata/largest_tumour_{self.modality}.png')
        
        return largest_tumor, stats_dict
            

    def get_intensity_range(self) -> Tuple[float, float, Dict]:
        """
        Find the minimum and maximum intensity values across all CT images.
        
        Returns:
            Tuple containing:
                - (min_intensity, max_intensity)
                - stats_dict: Dictionary with intensity statistics for each patient
        """
        stats_dict = {}
        global_min = float('inf')
        global_max = float('-inf')
        
        # Lists to collect data for plotting
        min_intensities = []
        max_intensities = []
        mean_intensities = []
        intensity_ranges = []
        
        for i, sample in enumerate(self.dataset):
            # Read CT/MRI image
            ct_img_path = sample["data.input.img_path"]
            ct_img = sitk.ReadImage(os.path.join(self.img_dir, ct_img_path))
            ct_array = sitk.GetArrayFromImage(ct_img)
            
            # Calculate patient-specific statistics
            patient_min = np.min(ct_array)
            patient_max = np.max(ct_array)
            patient_mean = np.mean(ct_array)
            patient_std = np.std(ct_array)
            patient_range = patient_max - patient_min
            
            # Update global range
            global_min = min(global_min, patient_min)
            global_max = max(global_max, patient_max)
            
            # Store for plotting
            min_intensities.append(patient_min)
            max_intensities.append(patient_max)
            mean_intensities.append(patient_mean)
            intensity_ranges.append(patient_range)
            
            # Store patient statistics
            stats_dict[f"patient_{i}"] = {
                "min_intensity": float(patient_min),
                "max_intensity": float(patient_max),
                "mean_intensity": float(patient_mean),
                "std_intensity": float(patient_std),
                "intensity_range": float(patient_range)
            }
        
        # Create simple plots
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        
        # Plot 1: Minimum intensities
        axes[0, 0].hist(min_intensities, bins=15, color='skyblue', edgecolor='black', alpha=0.7)
        axes[0, 0].axvline(np.median(min_intensities), color='red', linestyle='--', 
                        label=f'Median: {np.median(min_intensities):.1f}')
        axes[0, 0].set_title('Minimum Intensity Distribution')
        axes[0, 0].set_xlabel('Intensity')
        axes[0, 0].set_ylabel('Count')
        axes[0, 0].legend()
        
        # Plot 2: Maximum intensities
        axes[0, 1].hist(max_intensities, bins=15, color='lightgreen', edgecolor='black', alpha=0.7)
        axes[0, 1].axvline(np.median(max_intensities), color='red', linestyle='--',
                        label=f'Median: {np.median(max_intensities):.1f}')
        axes[0, 1].set_title('Maximum Intensity Distribution')
        axes[0, 1].set_xlabel('Intensity')
        axes[0, 1].set_ylabel('Count')
        axes[0, 1].legend()
        
        # Plot 3: Mean intensities
        axes[1, 0].hist(mean_intensities, bins=15, color='lightcoral', edgecolor='black', alpha=0.7)
        axes[1, 0].axvline(np.median(mean_intensities), color='red', linestyle='--',
                        label=f'Median: {np.median(mean_intensities):.1f}')
        axes[1, 0].set_title('Mean Intensity Distribution')
        axes[1, 0].set_xlabel('Intensity')
        axes[1, 0].set_ylabel('Count')
        axes[1, 0].legend()
        
        # Plot 4: Intensity ranges
        axes[1, 1].hist(intensity_ranges, bins=15, color='gold', edgecolor='black', alpha=0.7)
        axes[1, 1].axvline(np.median(intensity_ranges), color='red', linestyle='--',
                        label=f'Median: {np.median(intensity_ranges):.1f}')
        axes[1, 1].set_title('Intensity Range Distribution')
        axes[1, 1].set_xlabel('Range (max-min)')
        axes[1, 1].set_ylabel('Count')
        axes[1, 1].legend()
        
        plt.tight_layout()
        plt.show()
        plt.savefig(f'/projects/prjs1779/Osteosarcoma/ViT_OSdata/intensity_tumour_{self.modality}.png')
        
        # Print summary
        print("=" * 60)
        print("INTENSITY RANGE ANALYSIS")
        print("=" * 60)
        print(f"Global intensity range: [{global_min:.1f}, {global_max:.1f}]")
        print(f"Global range width: {global_max - global_min:.1f}")
        
        print(f"\nPer-patient Statistics:")
        print(f"Average intensity range: {np.mean(intensity_ranges):.1f}")
        print(f"Minimum intensity range: {np.min(intensity_ranges):.1f}")
        print(f"Maximum intensity range: {np.max(intensity_ranges):.1f}")
        
        print(f"\nQuartile Statistics:")
        print(f"Min intensity - Q25: {np.percentile(min_intensities, 25):.1f}, "
            f"Median: {np.median(min_intensities):.1f}, "
            f"Q75: {np.percentile(min_intensities, 75):.1f}")
        print(f"Max intensity - Q25: {np.percentile(max_intensities, 25):.1f}, "
            f"Median: {np.median(max_intensities):.1f}, "
            f"Q75: {np.percentile(max_intensities, 75):.1f}")
        print(f"Mean intensity - Q25: {np.percentile(mean_intensities, 25):.1f}, "
            f"Median: {np.median(mean_intensities):.1f}, "
            f"Q75: {np.percentile(mean_intensities, 75):.1f}")
        
        return (float(global_min), float(global_max)), stats_dict


class OpGISTResample(OpBase):
    """
    Resamples CT scans and segmentation masks to a specified spacing (Z, Y, X).
    Assumes all arrays are in [Z, Y, X] format and returns resampled arrays in the same format.
    """

    def __init__(self, target_spacing_z, target_spacing_y, target_spacing_x):
        super().__init__()
        # Store as tuple to be converted later
        self._target_spacing = (target_spacing_z, target_spacing_y, target_spacing_x)

    def __call__(self, sample_dict: NDict) -> NDict:
        # Convert spacing to NumPy array (Z, Y, X)
        target_spacing = np.array(self._target_spacing, dtype=np.float32)

        # --- Check required metadata ---
        if "data.input.img.original_spacing" not in sample_dict or \
           "data.input.img.original_size" not in sample_dict:
            raise ValueError("Missing original spacing or size in metadata.")

        ct_np = sample_dict["data.input.img"]        # shape: [Z, Y, X]
        seg_np = sample_dict["data.input.seg"]       # shape: [Z, Y, X]

        original_spacing = sample_dict["data.input.img.original_spacing"]  # [Z, Y, X]
        original_size = sample_dict["data.input.img.original_size"]        # [Z, Y, X]

        # --- Convert to SimpleITK Images ---
        ct_sitk = sitk.GetImageFromArray(ct_np)
        seg_sitk = sitk.GetImageFromArray(seg_np)

        ct_sitk.SetSpacing(original_spacing[::-1].tolist())  # to [X, Y, Z]
        seg_sitk.SetSpacing(original_spacing[::-1].tolist())

        # --- Compute new size ---
        new_size = np.round(original_size * (original_spacing / target_spacing)).astype(int)

        # --- Resample CT ---
        resampled_ct = self.resample_image(
            ct_sitk,
            target_spacing=target_spacing[::-1],  # to [X, Y, Z]
            new_size=new_size[::-1],
            interpolator=sitk.sitkLinear
        )

        # --- Resample segmentation ---
        resampled_seg = self.resample_image(
            seg_sitk,
            target_spacing=target_spacing[::-1],
            new_size=new_size[::-1],
            interpolator=sitk.sitkNearestNeighbor
        )

        # --- Convert back to NumPy (still [Z, Y, X]) ---
        sample_dict["data.input.img.resampled"] = sitk.GetArrayFromImage(resampled_ct).copy()
        sample_dict["data.input.seg.resampled"] = sitk.GetArrayFromImage(resampled_seg).copy()

        return sample_dict

    def resample_image(self, image, target_spacing, new_size, interpolator):
        resampler = sitk.ResampleImageFilter()
        resampler.SetReferenceImage(image)
        resampler.SetOutputSpacing([float(s) for s in target_spacing])
        resampler.SetSize([int(s) for s in new_size])
        resampler.SetInterpolator(interpolator)
        return resampler.Execute(image)



class OpClipMaskedNoNorm(OpBase):
    """
    Clip CT values within the tumor region only, without min-max normalization.
    - Tumor voxels: clipped to `clip_range` and kept in HU scale
    - Non-tumor voxels: set to `background_val`
    """

    def __init__(
        self,
        # TODO: this range is for CT, need adapt for MRI
        clip_range: Tuple[float, float] = (-150, 250),
        background_val: float = -150.0,
    ):
        super().__init__()
        self.clip_range = clip_range
        self.background_val = background_val

    def __call__(self, sample_dict: NDict, key_in: Tuple[str, str], key_out: str) -> NDict:
        key_img, key_seg = key_in
        ct = sample_dict[key_img]    # [Z, Y, X]
        seg = sample_dict[key_seg]   # [Z, Y, X]

        assert ct.shape == seg.shape, \
            f"CT and segmentation shape mismatch: {ct.shape} vs {seg.shape}"

        # tumor mask
        mask = seg > 0

        # init output as background
        out = np.full_like(ct, fill_value=self.background_val, dtype=np.float32)

        # clip tumor voxels (NO min-max normalization)
        tumor_voxels = ct[mask]
        clipped = np.clip(tumor_voxels, self.clip_range[0], self.clip_range[1])

        out[mask] = clipped.astype(np.float32)

        sample_dict[key_out] = out.copy()
        return sample_dict


class OpTumorCrop(OpBase):
    """
    Crop both CT and segmentation arrays using the 3D bounding box of the tumor.
    Assumes inputs are in [Z, Y, X] format.

    Input:
        - key_in: Tuple[str, str] → (ct_key, seg_key)
        - key_out: Tuple[str, str] → (cropped_ct_key, cropped_seg_key)
    """

    def __init__(self):
        super().__init__()

    def __call__(self, sample_dict: NDict, key_in: Tuple[str, str], key_out: Tuple[str, str]) -> NDict:
        key_ct_in, key_seg_in = key_in
        key_ct_out, key_seg_out = key_out

        ct = sample_dict[key_ct_in]    # [Z, Y, X]
        seg = sample_dict[key_seg_in]  # [Z, Y, X]

        assert ct.shape == seg.shape, \
            f"CT and segmentation shape mismatch: {ct.shape} vs {seg.shape}"

        # Find tumor bounding box (non-zero region)
        nonzero_indices = np.argwhere(seg > 0)
        if nonzero_indices.size == 0:
            raise ValueError(f"No tumor region found in segmentation for sample: {sample_dict.get('data.sample_id', 'UNKNOWN')}")

        z_min, y_min, x_min = nonzero_indices.min(axis=0)
        z_max, y_max, x_max = nonzero_indices.max(axis=0) + 1  # inclusive upper bound

        # Apply crop
        ct_crop = ct[z_min:z_max, y_min:y_max, x_min:x_max].copy()
        seg_crop = seg[z_min:z_max, y_min:y_max, x_min:x_max].copy()

        # Store output
        sample_dict[key_ct_out] = ct_crop
        sample_dict[key_seg_out] = seg_crop

        return sample_dict


class OpTumorRandomRotation(OpBase):
    """
    Applies a random 3D rotation to both CT and segmentation volumes in [Z, Y, X] format.

    - Rotates around randomly chosen axes (e.g., (Z,Y), (Z,X), (Y,X))
    - Uses cubic interpolation for CT, nearest for segmentation
    - Pads missing regions with cval: -1024 for CT, 0 for segmentation
    """

    def __init__(
        self,
        angle_range: Tuple[float, float] = (-10.0, 10.0),
        axes_options: Optional[Sequence[Tuple[int, int]]] = None,
    ):
        super().__init__()
        self.angle_range = angle_range
        self.axes_options = axes_options if axes_options is not None else [(0, 1), (0, 2), (1, 2)]  # (Z, Y), (Z, X), (Y, X)

    def __call__(self, sample_dict: NDict, key_in: Tuple[str, str], key_out: Tuple[str, str]) -> NDict:
        key_ct_in, key_seg_in = key_in
        key_ct_out, key_seg_out = key_out

        ct = sample_dict[key_ct_in]  # [Z, Y, X]
        seg = sample_dict[key_seg_in]  # [Z, Y, X]

        assert ct.shape == seg.shape, \
            f"CT and segmentation shape mismatch: {ct.shape} vs {seg.shape}"

        # Sample random rotation
        angle = random.uniform(*self.angle_range)
        axes = random.choice(self.axes_options)

        # Apply rotation
        ct_rot = rotate(
            ct, angle=angle, axes=axes,
            reshape=True, order=3, mode='constant', cval=-150.0
        )
        seg_rot = rotate(
            seg, angle=angle, axes=axes,
            reshape=True, order=0, mode='constant', cval=0
        )

        sample_dict[key_ct_out] = ct_rot.copy()
        sample_dict[key_seg_out] = seg_rot.copy()

        # Optional debug info
        sample_dict['data.input.img.rotated.info'] = {
            'angle_deg': angle,
            'axes': axes
        }

        return sample_dict


class OpPadOrCropToFixedDivisibleShape(OpBase):
    """
    Pads or center-crops a 3D tumor volume to a fixed shape [Z, Y, X] that is divisible by patch_size.

    - Pads with -1.0 if input is smaller
    - Center-crops if input is larger (e.g., due to rotation)
    """

    def __init__(self, patch_size: Tuple[int, int, int], largest_tumor: Tuple[int, int, int]):
        """
        :param patch_size: patch size (Z, Y, X)
        :param largest_tumor: the largest tumor size (Z, Y, X) from dataset (will be rounded up inside __call__)
        """
        super().__init__()
        self.patch_size = patch_size
        self.largest_tumor = largest_tumor
        self.pad_val = -150.0

    def __call__(
        self,
        sample_dict: NDict,
        key_in: str = 'data.input.img.tumor3d.norm',
        key_out: str = 'data.input.img.tumor3d.fitted'
    ) -> NDict:
        volume = sample_dict[key_in]  # Expected shape: [Z, Y, X]
        assert volume.ndim == 3, f"Expected 3D volume, got shape {volume.shape}"

        # Compute padded/cropped target shape rounded up to nearest divisible by patch size
        target_shape = tuple(
            ((s + p - 1) // p) * p for s, p in zip(self.largest_tumor, self.patch_size)
        )

        fitted = volume
        for dim in range(3):  # Z, Y, X
            input_dim = fitted.shape[dim]
            target_dim = target_shape[dim]

            if input_dim < target_dim:
                # Pad symmetrically
                total_pad = target_dim - input_dim
                pad_before = total_pad // 2
                pad_after = total_pad - pad_before
                pad_config = [(0, 0), (0, 0), (0, 0)]
                pad_config[dim] = (pad_before, pad_after)
                fitted = np.pad(fitted, pad_config, mode='constant', constant_values=self.pad_val)

            elif input_dim > target_dim:
                # Center-crop
                excess = input_dim - target_dim
                start = excess // 2
                end = start + target_dim
                fitted = np.take(fitted, indices=range(start, end), axis=dim)

        assert fitted.shape == target_shape, f"Expected shape {target_shape}, got {fitted.shape}"
        sample_dict[key_out] = fitted.copy()
        return sample_dict


class OpCTPatchifyWithMask(OpBase):
    """
    Converts a 3D tumor volume [Z, Y, X] into non-overlapping patch tokens and creates an attention mask.

    - Input: volume of shape [Z, Y, X], assumed to be padded/cropped to a fixed shape
    - Output:
        - key_out (default: 'data.input.img.tumor3d.patches'): FloatTensor [num_patches, patch_dim]
        - 'model.embed_mask_b': BoolTensor [num_patches], True for valid tokens, False for padded ones

    Parameters:
        patch_size: tuple of ints
        pad_val: float, value used in padding
        mask_pad_threshold: float in [0.0, 1.0] – percentage of patch that can be padding before masking
            e.g. 0.0 = mask if any voxel is padding (strict), 0.3 = mask if ≥30% padding, 1.0 = mask only if fully padding
    """

    def __init__(
        self,
        patch_size: Tuple[int, int, int],
        # TODO: choose the pad_min
        pad_val: float = -150.0,
        mask_pad_threshold: float = 0.0     # 0.3: >=70% tumor -> keep          <70% tumor -> mask
                                            # 0.0: A patch is masked even if it contains just one voxel with the padding value
                                            # 1.0: every patch will be kept, even those that are fully padding.
    ):
        super().__init__()
        self.patch_size = patch_size
        self.pad_val = pad_val
        self.mask_pad_threshold = mask_pad_threshold

    def __call__(
        self,
        sample_dict: NDict,
        key_in: str = 'data.input.img.tumor3d.fitted',
        key_out: str = 'data.input.img.tumor3d.patches'
    ) -> NDict:
        volume = sample_dict[key_in]  # [Z, Y, X]
        assert volume.ndim == 3, f"Expected 3D volume, got shape {volume.shape}"

        # Patchify
        z, y, x = volume.shape
        pz, py, px = self.patch_size
        assert z % pz == 0 and y % py == 0 and x % px == 0, \
            f"Volume shape {volume.shape} not divisible by patch size {self.patch_size}"

        patches = rearrange(volume,
                            '(z pz) (y py) (x px) -> (z y x) (pz py px)',
                            pz=pz, py=py, px=px)
        patches_tensor = torch.tensor(patches, dtype=torch.float32)

        # Create mask: patch is valid if the ratio of non-padding voxels ≥ (1 - threshold)
        total_voxels = patches_tensor.shape[1]
        num_non_pad = (patches_tensor != self.pad_val).sum(dim=1)
        mask = num_non_pad >= (1.0 - self.mask_pad_threshold) * total_voxels

        # Store outputs
        sample_dict[key_out] = patches_tensor  # [num_patches, patch_dim]
        sample_dict['model.embed_mask_b'] = mask  # [num_patches]

        return sample_dict


class OpClinicalPreprocess(OpBase):
    """
    Prepares clinical features for transformer input:
    - Binarizes continuous variables using thresholds
    - Converts categorical strings to integer indices
    - Assigns masked token index to missing values
    - Outputs: a flat vector of all features
    """

    def __init__(
        self,
        thresholds: Dict[str, float],
        categorical_mappings: Dict[str, Dict[str, int]]
    ):
        super().__init__()
        self.thresholds = thresholds
        self.cat_map = categorical_mappings

        # Ordered feature list for embedding extraction
        self.feature_names = list(thresholds.keys()) + list(categorical_mappings.keys())

    def __call__(self, sample_dict, key_in: str, key_out: str):
        raw = sample_dict[key_in]
        vector = []

        for feature in self.feature_names:
            if feature in self.thresholds:  # Binarize continuous
                raw_value = raw.get(feature, None)
                if pd.isna(raw_value):
                    category = 2  # masked
                else:
                    category = 0 if raw_value <= self.thresholds[feature] else 1
            else:  # Encode categorical
                raw_value = raw.get(feature, None)
                
                # transfer int to string for mapping in OS
                if isinstance(raw_value, int):
                    raw_value = str(raw_value)
                    
                num_categories = len(self.cat_map[feature])
                if pd.isna(raw_value):
                    category = num_categories  # masked
                else:
                    category = self.cat_map[feature][raw_value]

            vector.append(category)

        sample_dict[key_out] = torch.tensor(vector, dtype=torch.long)
        return sample_dict


class OpClinicalAugmentation(OpBase):
    """
    Applies augmentation to categorical clinical vectors:
    - Randomly masks categorical values using reserved index
    """

    def __init__(
        self,
        dropout_p: float = 0.1,
        feature_names: List[str] = None,
        mask_token_index: Dict[str, int] = None,
    ):
        """
        :param dropout_p: probability of masking a token
        :param feature_names: list of all categorical features (including binarized continuous ones)
        :param mask_token_index: dict mapping each feature name to its reserved mask token index
        """
        super().__init__()
        self.dropout_p = dropout_p
        self.feature_names = feature_names
        self.mask_token_index = mask_token_index

    def __call__(self, sample_dict: NDict, key_in: str, key_out: Optional[str] = None) -> NDict:
        vec = sample_dict[key_in].clone()
        if key_out is None:
            key_out = key_in

        for i, feature in enumerate(self.feature_names):
            if random.random() < self.dropout_p:
                vec[i] = self.mask_token_index[feature]

        sample_dict[key_out] = vec
        return sample_dict


class OpClinicalMask(OpBase):
    """
    Generates a binary mask for clinical input (xa) based on mask token index.
    `True` where value is not the mask token, `False` where it is.
    """
    def __init__(self, feature_names: List[str], mask_token_index: Dict[str, int]):
        super().__init__()
        self.feature_names = feature_names
        self.mask_token_index = mask_token_index

    def __call__(self, sample_dict: NDict, key_in='data.input.clinical.vector', key_out='model.embed_mask_a') -> NDict:
        #DEBUG
        # print(f"DEBUG OpClinicalMask: key_in = {key_in}")
        # print(f"DEBUG OpClinicalMask: sample_dict keys = {list(sample_dict.keys())}")
        vec = sample_dict[key_in]  # shape: [num_features]

        # print(f"DEBUG OpClinicalMask: vec = {vec}")
        mask = torch.ones_like(vec, dtype=torch.bool)

        for i, feat in enumerate(self.feature_names):
            if vec[i].item() == self.mask_token_index[feat]:
                mask[i] = False

        sample_dict[key_out] = mask # shape: [num_features]
        return sample_dict


class OpClinicalEmbedID(OpBase):
    """
    Generates per-feature embed_ids for TransformerWrapper, using string keys.

    Assumes input vector shape: [num_features] (each index is a token/category).
    Produces: model.embed_ids_a as a dict[str, Tensor] for each feature index as string.
    """

    def __init__(self, num_features: int):
        super().__init__()
        self.num_features = num_features

    def __call__(self, sample_dict, key_in='data.input.clinical.vector', key_out='model.embed_ids_a'):
        vec = sample_dict[key_in]  # shape: [num_features]
        embed_ids = {}

        for i in range(self.num_features):
            feature_key = str(i)
            embed_ids[feature_key] = vec[i].unsqueeze(0)  # shape: [1]

        sample_dict[key_out] = embed_ids
        return sample_dict

