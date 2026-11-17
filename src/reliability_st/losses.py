import torch
import torch.nn.functional as F


def spatial_consistency(relevance,coords,image_features,mask,neighbors=4,
                        spatial_sigma=1.0,morphology_temperature=1.0):
    # kNN neighborhood in normalized physical/image coordinate space;
    # similarity of pretrained UNI embeddings modulates smoothness.
    vals=[]
    for b in range(coords.shape[0]):
        n=int(mask[b].sum())
        if n<=1: continue
        pos=coords[b,:n]
        # scale coordinate axes to comparable range; sigma=1 denotes normalized units
        scale=pos.std(0,unbiased=False).clamp_min(1e-6)
        p=(pos-pos.mean(0))/scale
        dist=torch.cdist(p,p)
        dist.fill_diagonal_(float('inf'))
        k=min(neighbors,n-1)
        values, ids=dist.topk(k=k,largest=False)
        features=F.normalize(image_features[b,:n],dim=-1,eps=1e-6)
        sim=(features[:,None,:]*features[ids]).sum(-1).clamp(-1,1)
        weight=torch.exp(-(values**2)/(2*spatial_sigma**2))
        weight=weight*torch.exp((sim-1.0)/morphology_temperature)
        dr=(relevance[b,:n,None]-relevance[b,ids])**2
        vals.append((weight*dr).mean())
    return torch.stack(vals).mean() if vals else relevance.sum()*0


def objective(out,batch,cfg):
    t=cfg['training']
    ce=F.cross_entropy(out['logits'],batch['labels'])
    rec=F.mse_loss(out['reconstructed'],out['pooled'])
    kl=-0.5*(1+out['logvar']-out['mu'].square()-out['logvar'].exp()).sum(-1).mean()
    sp=spatial_consistency(out['relevance'],batch['coords'],batch['images'],batch['mask'],
                          t['neighbors'],t['spatial_sigma'],t['morphology_temperature'])
    # Disable VAE objectives for the w/o uncertainty ablation, consistently with model flags.
    if not cfg.get('model', {}).get('use_vae', True):
        rec=rec*0
        kl=kl*0
    total=ce+t['lambda_rec']*rec+t['beta']*kl+t['lambda_sp']*sp
    return total, {'loss':total.detach(),'grade':ce.detach(),'rec':rec.detach(),'kl':kl.detach(),'spatial':sp.detach()}
