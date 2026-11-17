#!/usr/bin/env python3
import argparse
import numpy as np
import matplotlib.pyplot as plt

if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--prediction',required=True);p.add_argument('--out',default='tumor_relevance.png')
    p.add_argument('--background',help='Optional registered H&E PNG/JPG (same coordinates)')
    args=p.parse_args()
    with np.load(args.prediction) as x: coords=x['coords'];scores=x['relevance']
    fig,ax=plt.subplots(figsize=(8,7))
    if args.background:ax.imshow(plt.imread(args.background))
    s=ax.scatter(coords[:,0],coords[:,1],c=scores,cmap='coolwarm',vmin=0,vmax=1,s=30,alpha=.9)
    ax.invert_yaxis() if not args.background else None
    fig.colorbar(s,ax=ax,label='Predicted tumor relevance (not calibrated probability)')
    ax.set_xlabel('x coordinate');ax.set_ylabel('y coordinate')
    fig.tight_layout();fig.savefig(args.out,dpi=250)
