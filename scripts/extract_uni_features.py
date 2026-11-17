#!/usr/bin/env python3
"""Extract frozen UNI embeddings from pre-cropped 224x224 patch paths.

Input CSV columns: sample_id, patient_id, cohort, grade_label, patch_path, x, y,
plus one column for each gene starting with gene_ (raw count values).
Rows for a sample must correspond to the same set/order of genes.
Model access requires accepting UNI terms on Hugging Face and authenticating.
"""
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset,DataLoader

class Patches(Dataset):
    def __init__(self,paths,transform):self.paths=paths;self.transform=transform
    def __len__(self):return len(self.paths)
    def __getitem__(self,i):
        with Image.open(self.paths[i]) as image:
            return self.transform(image.convert('RGB'))

@torch.no_grad()
def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--spots-csv',required=True)
    ap.add_argument('--outdir',default='data/features')
    ap.add_argument('--model',default='hf-hub:MahmoodLab/UNI')
    ap.add_argument('--batch-size',type=int,default=128)
    ap.add_argument('--device',default='auto')
    args=ap.parse_args()
    import timm
    from timm.data import resolve_model_data_config,create_transform
    dev='cuda' if args.device=='auto' and torch.cuda.is_available() else ('cpu' if args.device=='auto' else args.device)
    model=timm.create_model(args.model,pretrained=True,init_values=1e-5,dynamic_img_size=True)
    model.eval().to(dev)
    transform=create_transform(**resolve_model_data_config(model),is_training=False)
    frame=pd.read_csv(args.spots_csv)
    cols=['sample_id','patient_id','cohort','grade_label','patch_path','x','y']
    missing=set(cols)-set(frame.columns)
    if missing:raise ValueError(f'Missing {missing}')
    genecols=[c for c in frame.columns if c.startswith('gene_')]
    if not genecols:raise ValueError('No gene_* columns')
    out=Path(args.outdir);out.mkdir(parents=True,exist_ok=True)
    records=[]
    for sid,group in frame.groupby('sample_id',sort=False):
        for col in ['patient_id','cohort','grade_label']:
            if group[col].nunique()!=1:raise ValueError(f'{sid}: {col} not constant')
        dl=DataLoader(Patches(group.patch_path.tolist(),transform),batch_size=args.batch_size)
        feats=[]
        for batch in dl:
            f=model(batch.to(dev))
            if isinstance(f,(tuple,list)):f=f[0]
            if f.ndim!=2:raise RuntimeError('Expected UNI NxD features; check selected model')
            feats.append(f.cpu().numpy())
        path=out/(str(sid).replace('/','_')+'.npz')
        np.savez_compressed(path,image_features=np.concatenate(feats).astype(np.float32),
                            genes=group[genecols].to_numpy(np.float32),
                            coords=group[['x','y']].to_numpy(np.float32))
        records.append({'sample_id':sid,'patient_id':group.patient_id.iloc[0],
                        'cohort':group.cohort.iloc[0],'grade_label':group.grade_label.iloc[0],
                        'npz_path':str(path.resolve())})
    pd.DataFrame(records).to_csv(out.parent/'manifest.csv',index=False)
    (out.parent/'genes.txt').write_text('\n'.join(genecols)+'\n')
    print('Wrote',len(records),'section feature files and manifest')

if __name__=='__main__':main()
