#!/usr/bin/env python3
"""Predict grades and spot relevance on UNLABELED sections."""
import argparse,json
from pathlib import Path
import numpy as np
import torch
from reliability_st.model import ReliabilityST
from reliability_st.util import device_from_arg


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--checkpoint',required=True);p.add_argument('--npz',required=True)
    p.add_argument('--out',default='prediction.npz');p.add_argument('--device',default='auto')
    a=p.parse_args();dev=device_from_arg(a.device)
    ck=torch.load(a.checkpoint,map_location=dev,weights_only=False)
    model=ReliabilityST(**ck['model_args']).to(dev)
    model.load_state_dict(ck['state_dict']);model.eval()
    with np.load(a.npz,allow_pickle=False) as data:
        image=data['image_features'].astype(np.float32)
        gene=data['genes'].astype(np.float32)
        coords=data['coords'].astype(np.float32)
    if len(image)!=len(gene) or len(gene)!=len(coords):raise ValueError('Unequal spot counts')
    gene=np.log1p(1e4*gene/np.maximum(gene.sum(1,keepdims=True),1))
    with torch.no_grad():
        out=model(torch.tensor(image[None],device=dev),torch.tensor(gene[None],device=dev),
                  torch.ones((1,len(image)),device=dev,dtype=torch.bool),deterministic=True)
    probs=out['probs'][0].cpu().numpy()
    inv={int(v):k for k,v in ck['class_map'].items()}
    label=inv[int(probs.argmax())]
    np.savez_compressed(a.out,coords=coords,relevance=out['relevance'][0].cpu().numpy(),
                        probs=probs,uncertainty=out['uncertainty'].cpu().numpy())
    print(json.dumps({'predicted_grade':label,'probabilities':probs.tolist(),
                      'sample_latent_uncertainty':float(out['uncertainty'][0]),'output':a.out},indent=2))

if __name__=='__main__':main()
