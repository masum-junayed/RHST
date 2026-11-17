import numpy as np
import torch
from reliability_st.model import ReliabilityST
from reliability_st.losses import objective
from reliability_st.metrics import compute_metrics


def test_forward_backward():
    model=ReliabilityST(image_dim=32,gene_dim=16,n_classes=3,d_model=32,n_heads=4,
                        latent_dim=12,gene_hidden=20,fusion_hidden=40,refine_hidden=16,classifier_hidden=20)
    im=torch.randn(2,9,32);g=torch.rand(2,9,16)
    mask=torch.tensor([[1]*9,[1]*5+[0]*4],dtype=torch.bool)
    coords=torch.randn(2,9,2)
    out=model(im,g,mask)
    assert out['logits'].shape==(2,3)
    assert out['relevance'].shape==(2,9)
    assert torch.all(out['relevance'][1,5:]==0)
    batch={'images':im,'coords':coords,'labels':torch.tensor([0,2]),'mask':mask}
    cfg={'training':dict(lambda_rec=1,beta=.01,lambda_sp=.1,neighbors=3,spatial_sigma=1.,morphology_temperature=1.)}
    loss,_=objective(out,batch,cfg)
    loss.backward()
    assert torch.isfinite(loss)
    assert model.gene_encoder.net[0].weight.grad is not None


def test_metrics():
    x=compute_metrics([0,1,0,1],np.array([[.9,.1],[.1,.9],[.8,.2],[.3,.7]]))
    assert x['qwk'] == 1
