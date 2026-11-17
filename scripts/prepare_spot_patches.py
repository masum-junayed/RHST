#!/usr/bin/env python3
"""Crop registered H&E patches around ST spot coordinates.

This is a CLAM-compatible preprocessing ENTRYPOINT, NOT an exact implementation
of CLAM's segmentation algorithm. Use CLAM-generated tissue masks/WSI coordinates
when available; filter spots with your validated tissue mask before this step.
CSV required: sample_id,patient_id,cohort,grade_label,slide_path,x,y,gene_*.
Coordinates must be at the image's pixel resolution (not microns/array grid indices).
Reads ordinary RGB images with PIL; for SVS/large WSIs, uses OpenSlide if installed.
"""
import argparse
from pathlib import Path
import pandas as pd
from PIL import Image


def crop(path,x,y,size):
    if str(path).lower().endswith(('.svs','.ndpi','.mrxs','.tiff','.tif')):
        try:
            import openslide
        except ImportError as e:
            raise RuntimeError('Install openslide-python and OS libopenslide for WSI') from e
        s=openslide.OpenSlide(str(path))
        try:
            patch=s.read_region((round(x-size/2),round(y-size/2)),0,(size,size)).convert('RGB')
        finally:s.close()
        return patch
    with Image.open(path) as im:
        return im.convert('RGB').crop((round(x-size/2),round(y-size/2),round(x+size/2),round(y+size/2)))

if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--spots-csv',required=True);p.add_argument('--out-csv',default='data/patch_spots.csv')
    p.add_argument('--patch-dir',default='data/patches');p.add_argument('--source-size',type=int,default=224)
    p.add_argument('--size',type=int,default=224)
    a=p.parse_args()
    frame=pd.read_csv(a.spots_csv)
    for c in ['slide_path','x','y','sample_id']:
        if c not in frame:raise ValueError(f'Missing {c}')
    root=Path(a.patch_dir);root.mkdir(parents=True,exist_ok=True)
    paths=[]
    for i,row in frame.iterrows():
        patch=crop(row.slide_path,float(row.x),float(row.y),a.source_size)
        patch=patch.resize((a.size,a.size),Image.Resampling.BICUBIC)
        path=root/f'{str(row.sample_id).replace("/","_")}_{i:06d}.png'
        patch.save(path);paths.append(str(path.resolve()))
    frame['patch_path']=paths
    Path(a.out_csv).parent.mkdir(parents=True,exist_ok=True)
    frame.to_csv(a.out_csv,index=False)
    print('Created',len(paths),'spot-centered patches')
