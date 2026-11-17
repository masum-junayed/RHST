import json
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from .data import SectionDataset, collate_sections, load_manifest, make_folds, check_features,save_split
from .model import ReliabilityST
from .losses import objective
from .metrics import compute_metrics
from .util import set_seed, to_device,makedirs


def loader(frame, cfg, class_map,training):
    ds=SectionDataset(frame,cfg['manifest'],class_map,cfg.get('max_spots'),training,cfg['seed'])
    return DataLoader(ds,batch_size=cfg['training']['sample_batch_size'],shuffle=training,
                      num_workers=cfg.get('num_workers',0),collate_fn=collate_sections)

@torch.no_grad()
def predict(model,dl,device):
    model.eval(); ys=[]; probs=[]; rows=[]
    for batch in dl:
        batch=to_device(batch,device)
        out=model(batch['images'],batch['genes'],batch['mask'],deterministic=True)
        y=batch['labels'].cpu().numpy(); p=out['probs'].cpu().numpy()
        ys.extend(y.tolist());probs.extend(p.tolist())
        for j,sid in enumerate(batch['sample_ids']):
            rows.append(dict(sample_id=sid,patient_id=batch['patient_ids'][j],
                             true_label=int(y[j]),pred_label=int(p[j].argmax()),
                             probabilities=p[j].tolist()))
    return compute_metrics(ys,np.array(probs)),rows


def run_cross_validation(cfg,device):
    set_seed(cfg['seed'])
    frame=load_manifest(cfg['manifest'],cfg['label_column'],cfg['patient_column'],cfg['cohort_column'])
    frame=frame.loc[frame[cfg['cohort_column']].astype(str)==cfg['primary_cohort']].copy().reset_index(drop=True)
    if not len(frame):raise ValueError('No rows match primary_cohort')
    frame=frame.rename(columns={cfg['label_column']:'grade_label',cfg['patient_column']:'patient_id',cfg['cohort_column']:'cohort'})
    img_dim,gene_dim=check_features(frame,cfg['manifest'])
    # Label vocabulary determined by declared labels of this cohort; no test model fitting.
    classes=sorted(frame['grade_label'].astype(str).unique().tolist())
    if len(classes)<2:raise ValueError('At least two distinct labels required')
    class_map={c:i for i,c in enumerate(classes)}
    root=Path(cfg['output_dir']);makedirs(root)
    (root/'classes.json').write_text(json.dumps(class_map,indent=2))
    (root/'config_used.json').write_text(json.dumps(cfg,indent=2))
    folds=[]
    for fold,train_idx,val_idx,test_idx in make_folds(frame,cfg['n_folds'],cfg['val_fraction'],'patient_id',cfg['seed']):
        folder=root/f'fold_{fold}';makedirs(folder)
        save_split(folder/'splits.csv',frame,train_idx,val_idx,test_idx)
        dl_train=loader(frame.iloc[train_idx],cfg,class_map,True)
        dl_val=loader(frame.iloc[val_idx],cfg,class_map,False)
        dl_test=loader(frame.iloc[test_idx],cfg,class_map,False)
        model=ReliabilityST(image_dim=img_dim,gene_dim=gene_dim,n_classes=len(classes),**cfg['model']).to(device)
        t=cfg['training']
        opt=torch.optim.Adam(model.parameters(),lr=t['lr'],weight_decay=t['weight_decay'])
        scheduler=torch.optim.lr_scheduler.StepLR(opt,step_size=t['step_size'],gamma=t['gamma'])
        best=-float('inf'); patience=0
        for epoch in range(t['epochs']):
            dl_train.dataset.set_epoch(epoch);model.train();running=[]
            for batch in dl_train:
                batch=to_device(batch,device)
                opt.zero_grad(set_to_none=True)
                out=model(batch['images'],batch['genes'],batch['mask'])
                loss,logs=objective(out,batch,cfg)
                if not torch.isfinite(loss):raise FloatingPointError('Nonfinite loss')
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(),t['grad_clip'])
                opt.step();running.append(float(loss.detach().cpu()))
            scheduler.step()
            metrics,_=predict(model,dl_val,device)
            key=metrics[t['val_metric']]
            if np.isnan(key):
                # On tiny validation sets QWK/AUC can be undefined; fall back to NLL.
                key=-metrics['nll']
            print(f'fold={fold} epoch={epoch+1} train_loss={np.mean(running):.4f} val={metrics}')
            if key>best:
                best=key;patience=0
                torch.save({'state_dict':model.state_dict(),'model_args':{'image_dim':img_dim,
                     'gene_dim':gene_dim,'n_classes':len(classes),**cfg['model']},
                     'class_map':class_map,'epoch':epoch,'validation':metrics},folder/'best.pt')
            else:
                patience+=1
                if patience>=t['patience']:break
        if not (folder/'best.pt').exists():
            raise RuntimeError('No valid validation checkpoint. Check fold class counts.')
        ckpt=torch.load(folder/'best.pt',map_location=device,weights_only=False)
        model.load_state_dict(ckpt['state_dict'])
        test_metrics,rows=predict(model,dl_test,device)
        (folder/'test_metrics.json').write_text(json.dumps(test_metrics,indent=2))
        (folder/'test_predictions.json').write_text(json.dumps(rows,indent=2))
        folds.append(test_metrics)
        print('TEST',fold,test_metrics)
    names=folds[0].keys()
    def summarize(key):
        vals=np.asarray([f[key] for f in folds],dtype=float)
        valid=vals[np.isfinite(vals)]
        return {'mean':float(valid.mean()) if len(valid) else float('nan'),
                'std':float(valid.std(ddof=1)) if len(valid)>1 else float('nan'),
                'valid_folds':int(len(valid))}
    summary={k:summarize(k) for k in names}
    (root/'cv_summary.json').write_text(json.dumps({'folds':folds,'summary':summary},indent=2))
    return summary
