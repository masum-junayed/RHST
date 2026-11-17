"""Patient-grouped spatial sections. No assumed grades/labels are fabricated."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from sklearn.model_selection import GroupKFold, GroupShuffleSplit

REQUIRED = {"sample_id", "patient_id", "cohort", "npz_path", "grade_label"}


def load_manifest(path, label_column="grade_label", patient_column="patient_id", cohort_column="cohort"):
    table = pd.read_csv(path)
    needed = {"sample_id", "npz_path", label_column, patient_column, cohort_column}
    missing = needed.difference(table.columns)
    if missing:
        raise ValueError(f"Manifest missing columns: {sorted(missing)}")
    if table["sample_id"].duplicated().any():
        raise ValueError("Duplicate sample_id in manifest")
    if table[label_column].isna().any():
        raise ValueError("Missing grading labels. Supply documented labels before training.")
    return table.reset_index(drop=True)


def resolve_file(manifest_path, npz_path):
    p = Path(str(npz_path))
    return p if p.is_absolute() else Path(manifest_path).resolve().parent / p


def check_features(manifest, manifest_path):
    shapes = []
    for row in manifest.itertuples(index=False):
        p = resolve_file(manifest_path, getattr(row, "npz_path"))
        if not p.exists():
            raise FileNotFoundError(p)
        with np.load(p, allow_pickle=False) as d:
            for name in ("image_features", "genes", "coords"):
                if name not in d:
                    raise ValueError(f"{p}: missing {name}")
            img, gene, coords = d["image_features"], d["genes"], d["coords"]
            if img.ndim != 2 or gene.ndim != 2 or coords.ndim != 2 or coords.shape[1] != 2:
                raise ValueError(f"{p}: expect image_features N,D; genes N,G; coords N,2")
            if not (len(img) == len(gene) == len(coords)) or len(img) == 0:
                raise ValueError(f"{p}: spots must have equal, nonzero counts")
            shapes.append((img.shape[1], gene.shape[1]))
    if len(set(shapes)) != 1:
        raise ValueError("Feature/gene dimensions differ; align gene IDs/order across sections")
    return shapes[0]


def make_folds(frame, k, val_fraction, patient_col, seed):
    groups = frame[patient_col].astype(str).values
    n_patients = len(np.unique(groups))
    if n_patients < k:
        raise ValueError(f"Need at least {k} independent patients, found {n_patients}")
    cv = GroupKFold(n_splits=k)
    for fold, (trainval, test) in enumerate(cv.split(frame, groups=groups)):
        splitter = GroupShuffleSplit(n_splits=1, test_size=val_fraction, random_state=seed+fold)
        a,b = next(splitter.split(trainval, groups=groups[trainval]))
        train, val = trainval[a], trainval[b]
        sets = [set(groups[x]) for x in (train, val, test)]
        assert not (sets[0]&sets[1] or sets[0]&sets[2] or sets[1]&sets[2])
        yield fold, train, val, test


class SectionDataset(Dataset):
    def __init__(self, frame, manifest_path, class_map, max_spots=None, training=False, seed=42):
        self.frame = frame.reset_index(drop=True)
        self.manifest_path = manifest_path
        self.class_map = class_map
        self.max_spots = max_spots
        self.training = training
        self.seed = seed
        self.epoch = 0

    def set_epoch(self, epoch):
        self.epoch = epoch

    def __len__(self):
        return len(self.frame)

    def __getitem__(self, index):
        row = self.frame.iloc[index]
        path = resolve_file(self.manifest_path, row["npz_path"])
        with np.load(path, allow_pickle=False) as dat:
            image = np.asarray(dat["image_features"], dtype=np.float32)
            genes = np.asarray(dat["genes"], dtype=np.float32)
            coords = np.asarray(dat["coords"], dtype=np.float32)
        if not (np.isfinite(image).all() and np.isfinite(genes).all() and np.isfinite(coords).all()):
            raise ValueError(f"{path} contains non-finite values")
        if np.any(genes < 0):
            raise ValueError(f"{path}: genes must be nonnegative raw expression counts")
        # Library size normalization is per spot; no train/test statistics are estimated.
        genes = np.log1p(1e4 * genes / np.maximum(genes.sum(axis=1, keepdims=True), 1.0)).astype(np.float32)
        n = len(image)
        if self.max_spots is not None and n > self.max_spots:
            rng = np.random.default_rng(self.seed + index * 99991 + (self.epoch if self.training else 0)*1000003)
            # Deterministic TEST subsample, random different TRAIN subset each epoch.
            idx = np.sort(rng.choice(n, size=self.max_spots, replace=False))
            image, genes, coords = image[idx], genes[idx], coords[idx]
        return {
            "images": torch.from_numpy(image), "genes": torch.from_numpy(genes),
            "coords": torch.from_numpy(coords),
            "label": torch.tensor(self.class_map[str(row["grade_label"])], dtype=torch.long),
            "sample_id": str(row["sample_id"]), "patient_id": str(row["patient_id"]),
        }


def collate_sections(batch):
    b = len(batch); n = max(len(x["images"]) for x in batch)
    di = batch[0]["images"].shape[-1]; dg=batch[0]["genes"].shape[-1]
    img = torch.zeros(b,n,di); genes=torch.zeros(b,n,dg); coords=torch.zeros(b,n,2)
    mask=torch.zeros(b,n,dtype=torch.bool)
    for i,x in enumerate(batch):
        m=len(x["images"])
        img[i,:m]=x["images"]; genes[i,:m]=x["genes"]
        coords[i,:m]=x["coords"];mask[i,:m]=True
    return dict(images=img,genes=genes,coords=coords,mask=mask,
                labels=torch.stack([x["label"] for x in batch]),
                sample_ids=[x["sample_id"] for x in batch],
                patient_ids=[x["patient_id"] for x in batch])


def save_split(path, frame, train, val, test):
    path = Path(path)
    out = frame.copy()
    out["split"] = ""
    for ids, name in [(train,"train"),(val,"val"),(test,"test")]:
        out.loc[ids,"split"] = name
    out.to_csv(path,index=False)
