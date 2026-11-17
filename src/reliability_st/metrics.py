import numpy as np
from sklearn.metrics import cohen_kappa_score, f1_score, roc_auc_score, log_loss


def brier_score(probs,y):
    target=np.eye(probs.shape[1])[np.asarray(y,dtype=int)]
    return float(np.mean(np.sum((probs-target)**2,axis=1)))


def compute_metrics(y,probs):
    y=np.asarray(y,dtype=int);probs=np.asarray(probs,dtype=float)
    probs=np.maximum(probs,1e-12)
    probs=probs/probs.sum(axis=1,keepdims=True)
    preds=probs.argmax(-1)
    n=probs.shape[1]
    # Kappa is mathematically undefined if both prediction and truth use only one class.
    kappa = float('nan') if (len(set(y))==1 and len(set(preds))==1) else float(cohen_kappa_score(y,preds,labels=list(range(n)),weights='quadratic'))
    out={
        'qwk':kappa,
        'macro_f1':float(f1_score(y,preds,labels=list(range(n)),average='macro',zero_division=0)),
        'nll':float(log_loss(y,probs,labels=list(range(n)))),
        'brier':brier_score(probs,y),
    }
    try:
        if len(set(y)) < 2: raise ValueError('Only one test class; AUC undefined')
        if n == 2:
            out['auc']=float(roc_auc_score(y,probs[:,1]))
        else:
            # multiclass AUC undefined if one or more classes absent in fold.
            if len(set(y)) != n: raise ValueError('Not all classes represented')
            out['auc']=float(roc_auc_score(y,probs,multi_class='ovr',average='macro',labels=np.arange(n)))
    except ValueError:
        out['auc']=float('nan')
    return out
