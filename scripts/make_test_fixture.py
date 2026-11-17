#!/usr/bin/env python3
"""ONLY for testing the code; synthetic data is not paper evidence."""
from pathlib import Path
import numpy as np
import pandas as pd

if __name__=='__main__':
    root=Path('tests/fixture_data');root.mkdir(parents=True,exist_ok=True)
    rng=np.random.default_rng(42);rows=[]
    for i in range(10):
        n=12+i; grade=i%2
        img=rng.normal(loc=grade*.15,size=(n,32)).astype('float32')
        genes=rng.poisson(lam=2+grade,size=(n,16)).astype('float32')
        coords=rng.uniform(0,512,size=(n,2)).astype('float32')
        np.savez_compressed(root/f's{i}.npz',image_features=img,genes=genes,coords=coords)
        rows.append(dict(sample_id=f's{i}',patient_id=f'p{i}',cohort='HEST-1k',
                         grade_label=grade,npz_path=f's{i}.npz'))
    pd.DataFrame(rows).to_csv(root/'manifest.csv',index=False)
    print('Synthetic test fixture created. DO NOT cite as experimental results.')
