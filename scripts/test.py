#!/usr/bin/env python3
"""Evaluate a saved checkpoint on a held-out manifest; never tune on these labels."""
import argparse,json
from pathlib import Path
import torch
from reliability_st.data import load_manifest, check_features,SectionDataset,collate_sections
from reliability_st.model import ReliabilityST
from reliability_st.engine import predict
from reliability_st.util import device_from_arg,load_config
from torch.utils.data import DataLoader

if __name__=='__main__':
    a=argparse.ArgumentParser()
    a.add_argument('--checkpoint',required=True);a.add_argument('--manifest',required=True)
    a.add_argument('--config',default='configs/default.yaml');a.add_argument('--device',default='auto')
    a.add_argument('--output',default='heldout_predictions.json');args=a.parse_args()
    cfg=load_config(args.config);dev=device_from_arg(args.device)
    frame=load_manifest(args.manifest)
    ckpt=torch.load(args.checkpoint,map_location=dev,weights_only=False)
    check_features(frame,args.manifest)
    ds=SectionDataset(frame,args.manifest,ckpt['class_map'],cfg['max_spots'])
    dl=DataLoader(ds,batch_size=cfg['training']['sample_batch_size'],collate_fn=collate_sections)
    model=ReliabilityST(**ckpt['model_args']).to(dev)
    model.load_state_dict(ckpt['state_dict'])
    metrics,preds=predict(model,dl,dev)
    Path(args.output).write_text(json.dumps({'metrics':metrics,'predictions':preds},indent=2))
    print(metrics)
