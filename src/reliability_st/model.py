"""Frozen UNI embeddings -> gene MLP -> cross-attention -> single VAE -> refinement -> grading."""
import torch
from torch import nn
import torch.nn.functional as F


class MLP(nn.Module):
    def __init__(self, inp, hidden, out, dropout=0.1):
        super().__init__()
        self.net=nn.Sequential(nn.Linear(inp, hidden),nn.GELU(),nn.Dropout(dropout),nn.Linear(hidden,out))
    def forward(self,x):
        return self.net(x)


class ReliabilityST(nn.Module):
    def __init__(self, image_dim, gene_dim, n_classes, d_model=512, n_heads=8,
                 latent_dim=256, dropout=0.1, gene_hidden=256,
                 fusion_hidden=512, refine_hidden=128, classifier_hidden=256, use_gene=True, use_cross_attention=True,
                 use_vae=True, use_refinement=True, use_tumor_pooling=True):
        super().__init__()
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"
        self.use_gene=use_gene
        self.use_cross_attention=use_cross_attention
        self.use_vae=use_vae
        self.use_refinement=use_refinement
        self.use_tumor_pooling=use_tumor_pooling
        self.img_proj = nn.Linear(image_dim,d_model)
        self.gene_encoder = MLP(gene_dim,gene_hidden,d_model,dropout)
        self.i_to_g = nn.MultiheadAttention(d_model,n_heads,dropout=dropout,batch_first=True)
        self.g_to_i = nn.MultiheadAttention(d_model,n_heads,dropout=dropout,batch_first=True)
        self.norm_i=nn.LayerNorm(d_model);self.norm_g=nn.LayerNorm(d_model)
        self.fusion = MLP(2*d_model,fusion_hidden,d_model,dropout)
        # One shared VAE on pooled multimodal features.
        self.vae_encoder = MLP(d_model,d_model,latent_dim,dropout)
        self.mu=nn.Linear(latent_dim,latent_dim)
        self.logvar=nn.Linear(latent_dim,latent_dim)
        self.decoder=MLP(latent_dim,d_model,d_model,dropout)
        self.loc_init=nn.Linear(d_model,1)
        self.refine=MLP(d_model+latent_dim+1,refine_hidden,1,dropout)
        self.classifier=MLP(d_model,classifier_hidden,n_classes,dropout)

    def forward(self, image_features, genes, mask, deterministic=False):
        # mask True for real spots, False for padded positions
        img = self.img_proj(image_features)
        gen = self.gene_encoder(genes)
        key_padding_mask = ~mask
        if self.use_gene and self.use_cross_attention:
            img_ctx,_=self.i_to_g(img,gen,gen,key_padding_mask=key_padding_mask,need_weights=False)
            gene_ctx,_=self.g_to_i(gen,img,img,key_padding_mask=key_padding_mask,need_weights=False)
        else:
            img_ctx=torch.zeros_like(img)
            gene_ctx=torch.zeros_like(gen)
        fi=self.norm_i(img+img_ctx)
        fg=self.norm_g(gen+gene_ctx) if self.use_gene else torch.zeros_like(fi)
        fused=self.fusion(torch.cat([fi,fg],dim=-1))
        fused=fused*mask.unsqueeze(-1)
        pooled=fused.sum(1)/mask.sum(1,keepdim=True).clamp_min(1)
        e=self.vae_encoder(pooled)
        mu=self.mu(e);logvar=self.logvar(e).clamp(-12.0,8.0)
        z = mu if deterministic else mu + torch.exp(0.5*logvar)*torch.randn_like(mu)
        if not self.use_vae:
            z = torch.zeros_like(z)
        reconstructed=self.decoder(z)
        u=logvar.exp().mean(dim=-1,keepdim=True)  # B,1 latent dispersion (not pixel uncertainty)
        if not self.use_vae:
            u = torch.zeros_like(u)
        logits_init=self.loc_init(fused).squeeze(-1)
        ctx=torch.cat([fused, z[:,None,:].expand(-1,fused.shape[1],-1),
                       torch.log1p(u)[:,None,:].expand(-1,fused.shape[1],-1)],dim=-1)
        logits_ref=logits_init + self.refine(ctx).squeeze(-1) if self.use_refinement else logits_init
        relevance=torch.sigmoid(logits_ref)*mask
        alpha = relevance / relevance.sum(1,keepdim=True).clamp_min(1e-8)
        if not self.use_tumor_pooling:
            alpha = mask.float()/mask.sum(1,keepdim=True).clamp_min(1)
        tumor=(alpha.unsqueeze(-1)*fused).sum(1)
        grading_logits=self.classifier(tumor)
        return {"logits":grading_logits,"probs":grading_logits.softmax(-1),
                "relevance":relevance,"initial_relevance":torch.sigmoid(logits_init)*mask,
                "pooling_weights":alpha,"fused":fused,"pooled":pooled,
                "reconstructed":reconstructed,"mu":mu,"logvar":logvar,
                "uncertainty":u.squeeze(-1)}
