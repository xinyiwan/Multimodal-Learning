from fuse.data.datasets.dataset_default import DatasetDefault
from fuse.data.pipelines.pipeline_default import PipelineDefault
from fuse.data.ops.ops_read import OpReadDataframe
from fuse.utils.ndict import NDict
from fuse.data.ops.op_base import OpReversibleBase
from fuse.data.ops.ops_common import OpKeepKeypaths
from typing import Optional, Sequence, Tuple, List
import pandas as pd
import os
import torch
import matplotlib.pyplot as plt

from GISTDataUtils import (
    OpGISTLoadImage, OpCastLabelToFloat, OpGISTResample, OpTumorCrop,
    OpClipMaskedNoNorm, OpTumorRandomRotation, OpPadOrCropToFixedDivisibleShape,
    OpCTPatchifyWithMask, OpClinicalPreprocess, OpClinicalAugmentation,
    OpClinicalMask, OpClinicalEmbedID, GISTDataUtils
)
from GIST import GISTDataset

class OpGISTSampleIDDecode(OpReversibleBase):
    # derive file paths from sample_id
    def __call__(self, sample_dict: NDict, op_id: Optional[str] = None) -> NDict:
        sid = sample_dict["data.sample_id"]
        sample_dict["data.input.img_path"] = f"{sid}.nii.gz"
        sample_dict["data.input.seg_path"] = f"{sid}.nii.gz"
        return sample_dict

class OSDataset:
    @staticmethod    
    def sample_ids(data_dir_seg: str) -> Sequence[str]:
        
        if not os.path.exists(data_dir_seg):
            raise FileNotFoundError(f"Directory not found: {data_dir_seg}")
        
        return [
            f.split(".")[0]
            for f in os.listdir(data_dir_seg)
            if f.endswith(".nii.gz")
        ]
    
    @staticmethod
    def setup_clinical_preprocessing(df_clinical: pd.DataFrame, sample_ids: List[str]):
        # Debug print to check what we're receiving
        # print(f"sample_ids type: {type(sample_ids)}")
        # print(f"sample_ids first few: {sample_ids[:5] if sample_ids else 'Empty list'}")
        # print(sample_ids)
        
        # Verify it's a list
        if not isinstance(sample_ids, list):
            raise TypeError(f"sample_ids must be a list, got {type(sample_ids)}: {sample_ids}")
        
        # Now filter
        df_filtered = df_clinical[df_clinical['sample_id'].isin(sample_ids)]
        print(f"Filtered from {len(df_clinical)} to {len(df_filtered)} samples")
        
        # TODO: possibly not use this in imaging only model. 
        # But what if there is no threshold for age 
        thresholds = {
            'Age_Start':18,
        }

        # TODO: already have mappings in clinical info
        categorical_mappings = {
            'sex': {
                '0': 0, '1': 1,
            },
            'pres_sympt': {
                '0': 0, '1': 1, '2': 2, '3': 3,
                '4': 4, '5': 5, '6': 6, '7': 7
            },
            'location': {
                '0': 0, '1': 1,
            },
            'diagnosis': {
                '0': 0, '1': 1, '2': 2, '3': 3,
                '4': 4, '5': 5, '6': 6
            },
            'metastasis': {
                '0': 0, '1': 1,
            },
            'tumor_size': {
                '0': 0, '1': 1, '2': 2
            }
        }
        feature_names = [
            "Age_Start", "sex", "pres_sympt", "location", 
            "diagnosis", "metastasis", "tumor_size"
        ]
        mask_token_index = {
            "Age_Start": 2, "sex": 2, "pres_sympt": 8,
            "location": 2, "diagnosis": 7, "metastasis": 2, "tumor_size": 3,
        }
        return thresholds, categorical_mappings, feature_names, mask_token_index

    def static_pipeline(
        data_dir_img: str,
        data_dir_seg: str,
        df_clinical: pd.DataFrame,
        thresholds,
        categorical_mappings
    ) -> PipelineDefault:

        return PipelineDefault("static", [
            (OpGISTSampleIDDecode(), dict()),
            (OpGISTLoadImage(data_dir_img), dict(key_in="data.input.img_path", key_out="data.input.img")),
            (OpGISTLoadImage(data_dir_seg), dict(key_in="data.input.seg_path", key_out="data.input.seg")),
            (OpReadDataframe(
                data=df_clinical,
                columns_to_extract=["sample_id", "Age_Start", 
                                    "sex", "pres_sympt", "location", "diagnosis",
                                    "metastasis", "tumor_size", "Huvosnew"],
                key_column="sample_id",
                key_name="data.sample_id"
            ), dict(prefix="data.input.clinical.raw")),
            (OpClinicalPreprocess(thresholds=thresholds, categorical_mappings=categorical_mappings),
             dict(key_in="data.input.clinical.raw", key_out="data.input.clinical.vector"))
        ])

    def dynamic_pipeline(
        patch_size: Tuple[int, int, int] = (8, 16, 16),
        largest_tumor: Tuple[int, int, int] = (83, 274, 301),
        train: bool = False,
        feature_names=None,
        mask_token_index=None,
        angle_range: Tuple[float, float] = (-10, 10),     # NEW: imaging augmentation range (degrees)
        mask_pad_threshold: float = 0.8,                  # NEW: patch mask threshold
        dropout_p: float = 0.1                            # NEW: clinical augmentation dropout prob
    ) -> PipelineDefault:

        steps = []

        # Resample
        # Note: After axis swap in OpGISTLoadImage, images are in [Z, Y, X] format.
        # OpGISTResample expects (target_spacing_z, target_spacing_y, target_spacing_x)
        steps.append((
            OpGISTResample(
                target_spacing_z=3.0,
                target_spacing_y=1.5,
                target_spacing_x=1.5
            ), dict()
        ))

        # Augmentation (rotation) — only in training
        if train:
            steps.append((
                OpTumorRandomRotation(angle_range=angle_range, axes_options=[(1, 2)]), dict(
                    key_in=("data.input.img.resampled", "data.input.seg.resampled"),
                    key_out=("data.input.img.rotated", "data.input.seg.rotated")
                )
            ))
            crop_input_keys = ("data.input.img.rotated", "data.input.seg.rotated")
        else:
            crop_input_keys = ("data.input.img.resampled", "data.input.seg.resampled")

        # Tumor crop
        steps.append((
            OpTumorCrop(), dict(
                key_in=crop_input_keys,
                key_out=("data.input.img.tumor3d", "data.input.seg.tumor3d")
            )
        ))

        # Normalize
        steps.append((
            OpClipMaskedNoNorm(), dict(
                key_in=("data.input.img.tumor3d", "data.input.seg.tumor3d"),
                key_out="data.input.img.tumor3d.norm"
            )
        ))

        # Pad/Crop to common divisible shape
        steps.append((
            OpPadOrCropToFixedDivisibleShape(
                patch_size=patch_size,
                largest_tumor=largest_tumor,
            ),
            dict(
                key_in='data.input.img.tumor3d.norm',
                key_out='data.input.img.tumor3d.fitted'
            )
        ))

        # Patchify + mask generation (uses configurable threshold)
        steps.append((
            OpCTPatchifyWithMask(
                patch_size=patch_size,
                pad_val=0.0,
                mask_pad_threshold=mask_pad_threshold,  # <-- NEW param
            ),
            dict(
                key_in='data.input.img.tumor3d.fitted',
                key_out='data.input.img.tumor3d.patches'
            )
        ))

        # Clinical augmentation (training only) with configurable dropout
        if train:
            steps.append((
                OpClinicalAugmentation(dropout_p=dropout_p, feature_names=feature_names, mask_token_index=mask_token_index),
                dict(key_in="data.input.clinical.vector")
            ))

        # Clinical mask for TransformerWrapper
        steps.append((
            # TODO: mask the unneccesary clincal features?
            OpClinicalMask(feature_names=feature_names, mask_token_index=mask_token_index),
            dict(key_in="data.input.clinical.vector")
        ))

        # Clinical emb_ids
        # For OS, assuming 7 clinical features as in previous code
        steps.append((
            OpClinicalEmbedID(num_features=7), dict()
        ))

        # Cast label to float
        steps.append((
            OpCastLabelToFloat(), dict(
                key_in="data.input.clinical.raw.Huvosnew",
                key_out="data.input.clinical.raw.Huvosnew.f"
            )
        ))

        keep_keys = [
            "data.sample_id",
            # TODO: comment out Center for now, but need more changes to align with full codes
            # "data.input.clinical.raw.Center",
            # "data.input.img",
            # "data.input.seg",
            # "data.input.img.b_swap",
            # "data.input.seg.b_swap",
            "data.input.img_path",
            "data.input.seg_path",
            "data.input.clinical.vector",
            "model.embed_ids_a",
            "model.embed_mask_a",
            # "data.input.img.tumor3d.fitted",
            # "data.input.img.tumor3d",
            "data.input.img.tumor3d.patches",
            "model.embed_mask_b",
            "data.input.clinical.raw.Huvosnew",
            "data.input.clinical.raw.Huvosnew.f"
        ]
        steps.append((OpKeepKeypaths(), {'keep_keypaths': keep_keys}))

        return PipelineDefault("dynamic", steps)

    def dataset(
        data_dir_img: str,
        data_dir_seg: str,
        clinical_csv_path: str,
        patch_size: Tuple[int, int, int] = (8, 16, 16),
        largest_tumor: Tuple[int, int, int] = (83, 274, 301),
        train: bool = False,
        sample_ids: Optional[Sequence[str]] = None,
        angle_range: Tuple[float, float] = (-10, 10),    # NEW pass-through
        mask_pad_threshold: float = 0.8,                  # NEW pass-through
        dropout_p: float = 0.1                            # NEW pass-through
    ) -> DatasetDefault:
        """
        Build GIST dataset with configurable augmentation & masking knobs.
        """

        df_clinical = pd.read_csv(clinical_csv_path)

        if sample_ids is None:
            # Get sample IDs from the segmentation directory
            sample_ids = OSDataset.sample_ids(data_dir_seg)
        else:
            # Make sure sample_ids is a list, not a string
            if isinstance(sample_ids, str):
                print(f"WARNING: sample_ids is string '{sample_ids}', converting to list")
                sample_ids = [sample_ids]

        thresholds, categorical_mappings, feature_names, mask_token_index = OSDataset.setup_clinical_preprocessing(
            df_clinical, sample_ids)

        static_pipeline = OSDataset.static_pipeline(
            data_dir_img=data_dir_img,
            data_dir_seg=data_dir_seg,
            df_clinical=df_clinical,
            thresholds=thresholds,
            categorical_mappings=categorical_mappings,
        )

        dynamic_pipeline = OSDataset.dynamic_pipeline(
            patch_size=patch_size,
            largest_tumor=largest_tumor,
            train=train,
            feature_names=feature_names,
            mask_token_index=mask_token_index,
            angle_range=angle_range,                # <-- wired in
            mask_pad_threshold=mask_pad_threshold,  # <-- wired in
            dropout_p=dropout_p                     # <-- wired in
        )

        dataset = DatasetDefault(
            sample_ids=sample_ids,
            static_pipeline=static_pipeline,
            dynamic_pipeline=dynamic_pipeline,
        )
        dataset.create()
        return dataset

if __name__ == "__main__":
    
    modalities = ['T1W', 'T1W_FS_C', 'T2W_FS']
    largest_tumor_sizes = [(53, 92, 95), (53, 93, 95), (47, 93, 106)]

    for idx, modality in enumerate(modalities):
        data_dir = f'/projects/prjs1779/Osteosarcoma/exp_data/{modality}/v1/'
        data_dir_img = os.path.join(data_dir, "input", "img")
        data_dir_seg = os.path.join(data_dir, "input", "seg")
        clinical_csv_path = os.path.join(data_dir, "clinical_features_with_Huvos.csv")
        data_paths = {"img": data_dir_img, "seg": data_dir_seg, "csv": clinical_csv_path}

        # Print the largest size for this modality
        print(f"Processing modality: {modality} with largest tumor size: {largest_tumor_sizes[idx]}")

        full_dataset = OSDataset.dataset(
            data_dir_img=data_paths["img"],
            data_dir_seg=data_paths["seg"],
            clinical_csv_path=data_paths["csv"],
            train=False,
            # sample_ids=['OS_000095_01', 'OS_000114_01'],
            sample_ids=None,
            patch_size=(8, 64, 64),
            largest_tumor=largest_tumor_sizes[idx],
            angle_range=(0.0, 0.0),
            mask_pad_threshold=0.8,
            dropout_p=0.0,
        )


        # Create a DataLoader
        dataloader = torch.utils.data.DataLoader(
            full_dataset,
            batch_size=1,  # Adjust batch size as needed
            shuffle=True,  # Shuffle for training, False for validation
            num_workers=0,  # Adjust based on your system
            pin_memory=True  # Useful for GPU training
        )


        # OSUtils = GISTDataUtils(dataset=full_dataset,
        #                         img_dir=data_dir_img,
        #                         seg_dir=data_dir_seg,
        #                         modality=modality)

        # # 1. Get maximum tumor size
        # largest_tumor, tumor_stats = OSUtils.get_max_tumor_size()

        # # 2. Get intensity range
        # (min_intensity, max_intensity), intensity_stats = OSUtils.get_intensity_range()
        # print(f"Global intensity range: [{min_intensity}, {max_intensity}]")

        # 3. visualize the pacthes for each subject
        for batch in dataloader:
            print(f"Processing batch with sample IDs: {batch['data.sample_id']}")
            # Visualize patches
            keys = ['data.input.img.tumor3d.fitted', 'data.input.img.tumor3d', 'data.input.img']
            for key in keys:
                def quick_plot(batch, key):
                    """
                    Quick plot - just show the middle segmentation slice
                    """
                    # get pid
                    pid = batch['data.sample_id'][0]
                    # get category
                    cat = key.split('.')[-1]
                    # makedir 
                    savedir = f'/projects/prjs1779/Osteosarcoma/ViT_OSdata/dataloader/{modality}/{cat}/'
                    os.makedirs(savedir, exist_ok=True)

                    data = batch[key][0]
                    data = data.detach().cpu().numpy()
                    
                    # Get middle slice
                    mid_slice = data[data.shape[0] // 2]
                    
                    # Simple plot
                    plt.figure(figsize=(8, 8))
                    plt.imshow(mid_slice, cmap='gray')
                    plt.savefig(os.path.join(savedir, f'{pid}_mid_slice.png'))
                quick_plot(batch, key=key)