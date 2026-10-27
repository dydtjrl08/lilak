"""Train the native-pad finding model from separate observation/truth ROOT files.

This is supervised simulation training, never a truth-dependent inference path.
No performance from an earlier pilot is inherited. The input files and all split
source keys are recorded. Every sibling of a source event stays in one split.

ROOT event TTrees: observation event positive_waveform[393216], file_index,
entry, source_group. Separate truth event truth_components[K*393216] and the
same three keys (optional truth_background[393216]). Field names configurable
in the manifest; the full waveform stays in ROOT rather than a required NPZ.
"""
import argparse
import hashlib
import json
from itertools import permutations
from pathlib import Path
import random
from time import perf_counter
import numpy as np


def split_source_groups(source_groups, seed=20261027, validation_fraction=.15, test_fraction=.15):
    if not 0<validation_fraction<1 or not 0<test_fraction<1 or validation_fraction+test_fraction>=1:
        raise ValueError('Need nonzero train, validation and test fractions')
    keys=[str(x) for x in source_groups]
    unique=sorted(set(keys))
    if len(unique)<3:raise ValueError('Need at least three independent source groups')
    rng=np.random.default_rng(seed);order=np.asarray(unique,dtype=object)[rng.permutation(len(unique))]
    nv=max(1,int(round(len(unique)*validation_fraction)))
    nt=max(1,int(round(len(unique)*test_fraction)))
    if nv+nt>=len(unique):nv=nt=1
    group_split={str(x):('validation' if i<nv else 'test' if i<nv+nt else 'train') for i,x in enumerate(order)}
    return [group_split[x] for x in keys],group_split


def pit_charge_loss(probabilities,target_fractions,charge,active_threshold=0.,quiet_weight=.03):
    """Eventwise min over alpha-slot permutations; background is never permuted.

    Charge-weighted soft CE plus quiet-bin background CE. This is not a theta
    MSE; voxel probabilities are arbitrary component slots. No test-set tuning.
    """
    import torch
    p=probabilities
    if p.shape!=target_fractions.shape or charge.shape!=(p.shape[0],*p.shape[2:]):
        raise ValueError('PIT dimensions mismatch')
    if not torch.isfinite(p).all() or not torch.isfinite(target_fractions).all() or not torch.isfinite(charge).all():
        raise ValueError('Nonfinite PIT input')
    if (target_fractions<0).any() or (charge<0).any():raise ValueError('Negative truth fractions or charge')
    if not torch.allclose(target_fractions.sum(1),torch.ones_like(charge),atol=2e-5):
        raise ValueError('Truth fractions must form a simplex')
    k=p.shape[1]-1
    if k>6:raise ValueError('Exact PIT is intended for K<=6')
    dims=tuple(range(1,charge.ndim))
    active=charge>active_threshold
    weight=charge*active
    denom=weight.sum(dims).clamp_min(1e-12)
    quiet=(~active).to(charge.dtype)
    quiet_denom=quiet.sum(dims).clamp_min(1.)
    logp=p.clamp_min(1e-8).log()
    costs=[]
    for perm in permutations(range(k)):
        ce=-(target_fractions*logp[:,list(perm)+[k]]).sum(1)
        costs.append((ce*weight).sum(dims)/denom+quiet_weight*(ce*quiet).sum(dims)/quiet_denom)
    return torch.stack(costs,dim=1).min(1).values.mean()


def _read_scalar_keys(tree):
    for key in ('file_index','entry','source_group'):
        if key not in tree:raise ValueError('ROOT training input missing '+key)
    return {name:np.asarray(tree[name].array(library='np')) for name in ('file_index','entry','source_group')}


def _root_array(tree,name,index,shape):
    if name not in tree:raise ValueError('ROOT input missing waveform branch '+name)
    value=tree[name].array(entry_start=index,entry_stop=index+1,library='np')[0]
    a=np.asarray(value,dtype=np.float32)
    if a.size!=int(np.prod(shape)):raise ValueError('ROOT waveform geometry mismatch for '+name)
    if name+'__shape' in tree:
        embedded=tuple(int(x) for x in tree[name+'__shape'].array(entry_start=index,entry_stop=index+1,library='np')[0])
        if embedded!=tuple(shape):raise ValueError('Embedded stage waveform shape mismatch for '+name)
    return a.reshape(shape)


class RootFindingDataset:
    """Lazy event-by-event ROOT reads; no full corpus load into RAM."""
    def __init__(self,manifest,k=3,tb_bin=4):
        import uproot
        self.manifest=dict(manifest);self.k=int(k);self.tb_bin=int(tb_bin)
        obs_path=Path(manifest['observations_root']).expanduser().resolve(strict=True)
        truth_path=Path(manifest['truth_root']).expanduser().resolve(strict=True)
        if obs_path==truth_path:raise ValueError('Observation and truth ROOT must be separate files')
        self.paths=(obs_path,truth_path);self._handles=None
        self.source_groups_verified=False
        with uproot.open(obs_path) as f,uproot.open(truth_path) as tf:
            tree_name=manifest.get('tree','events' if 'manifest' in f else 'event')
            truth_tree_name=manifest.get('truth_tree','events' if 'manifest' in tf else 'event')
            if 'manifest' in f:
                from .stage_io import manifest as stage_manifest
                observation_meta=stage_manifest(obs_path)
                if observation_meta.get('role')!='observations':
                    raise ValueError('Training observations require observation-role ROOT, never truth/evaluation role')
                self.source_groups_verified=bool(observation_meta.get('source_group_partition_verified',False))
                declared_uuid=observation_meta.get('source_dataset_uuid')
                if declared_uuid and declared_uuid!=manifest.get('source_dataset_uuid'):
                    raise ValueError('Manifest source_dataset_uuid differs from observation-stage provenance')
            else:self.source_groups_verified=bool(manifest.get('source_groups_verified',False))
            if 'manifest' in tf:
                from .stage_io import manifest as stage_manifest
                if stage_manifest(truth_path).get('role')!='truth':raise ValueError('Truth stage must explicitly declare role=truth')
            obs=f[tree_name];truth=tf[truth_tree_name]
            if 'RNTuple' in obs.classname or 'RNTuple' in truth.classname:
                raise ValueError('Use ROOT TTrees compatible with server ROOT6.24, not RNTuple')
            a,b=_read_scalar_keys(obs),_read_scalar_keys(truth)
            mapping={}
            for i,(fi,entry) in enumerate(zip(b['file_index'],b['entry'])):
                key=int(fi),int(entry)
                if key in mapping:raise ValueError('Duplicate truth event key')
                mapping[key]=i
            self.truth_indices=[];self.event_keys=[];self.groups=[]
            source_uuid=str(manifest.get('source_dataset_uuid',''))
            if not source_uuid:raise ValueError('Manifest needs source_dataset_uuid to avoid group collisions across original files')
            seen=set()
            for fi,entry,group in zip(a['file_index'],a['entry'],a['source_group']):
                key=int(fi),int(entry)
                if key in seen:raise ValueError('Duplicate observation event key')
                seen.add(key)
                if key not in mapping:raise ValueError('Missing truth for observation key '+str(key))
                ti=mapping[key]
                if int(b['source_group'][ti])!=int(group):raise ValueError('Observation/truth source-group mismatch')
                self.truth_indices.append(ti);self.event_keys.append(key)
                self.groups.append(source_uuid+':'+str(int(group)))
        self.tree_name=tree_name;self.truth_tree_name=truth_tree_name
    def __len__(self):return len(self.event_keys)
    def __getstate__(self):
        state=dict(self.__dict__);state['_handles']=None;return state
    def close(self):
        if self._handles:
            for h in self._handles:h.close()
            self._handles=None
    def __getitem__(self,index):
        import torch,uproot
        from .algorithms.waveform_models import features_numpy
        if self._handles is None:self._handles=(uproot.open(self.paths[0]),uproot.open(self.paths[1]))
        ot=self._handles[0][self.tree_name]
        tt=self._handles[1][self.truth_tree_name]
        q=_root_array(ot,self.manifest.get('waveform_branch','positive_waveform'),index,(12,64,512))
        comp=_root_array(tt,self.manifest.get('truth_branch','truth_components'),self.truth_indices[index],(self.k,12,64,512))
        if not np.isfinite(q).all() or not np.isfinite(comp).all() or np.any(q<0) or np.any(comp<0):
            raise ValueError('Training requires finite nonnegative observed/truth waveforms')
        if 'truth_background' in tt:
            bg=_root_array(tt,'truth_background',self.truth_indices[index],(12,64,512))
        else:bg=np.maximum(q-comp.sum(0),0.)
        if not np.isfinite(bg).all() or np.any(bg<0):
            raise ValueError('Truth background must be finite nonnegative')
        total=comp.sum(0)+bg
        mismatch=np.abs(total-q).sum()/max(float(q.sum()),1e-12)
        if mismatch>float(self.manifest.get('max_truth_sum_relative_error',.02)):
            raise ValueError('Truth waveforms do not close to observed charge; audit the tag exporter/pedestal')
        all_comp=np.concatenate((comp,bg[None])).reshape(self.k+1,12,64,512//self.tb_bin,self.tb_bin).sum(-1)
        charge=q.reshape(12,64,512//self.tb_bin,self.tb_bin).sum(-1)
        target=np.divide(all_comp,all_comp.sum(0,keepdims=True),out=np.zeros_like(all_comp),where=all_comp.sum(0,keepdims=True)>0)
        target[-1,all_comp.sum(0)==0]=1.
        return (torch.from_numpy(features_numpy(q,self.tb_bin)),torch.from_numpy(target.astype(np.float32)),torch.from_numpy(charge))


def train_finding(manifest,output_dir,config=None):
    """Real optimization with source-group holdout, validation best checkpoint.

    Test loss is evaluated once after model selection, with all keys logged.
    Training loss improvement alone is never called a physics improvement.
    """
    import torch
    from torch.utils.data import DataLoader,Subset
    from .algorithms.waveform_models import NativeWaveformResUNet
    cfg=dict(config or {});seed=int(cfg.get('seed',20261027))
    random.seed(seed);np.random.seed(seed);torch.manual_seed(seed)
    output=Path(output_dir).expanduser().resolve()
    if output.exists():raise FileExistsError('Use a NEW training output directory: '+str(output))
    output.mkdir(parents=True)
    dataset=RootFindingDataset(manifest,int(cfg.get('k',3)),int(cfg.get('tb_bin',4)))
    if not dataset.source_groups_verified and not cfg.get('allow_unverified_source_groups',False):
        raise ValueError('Original source-group provenance is unverified. Supply a verified export; an exploratory old-data pilot requires explicit allow_unverified_source_groups=true and cannot establish source-independent performance.')
    labels,groups=split_source_groups(dataset.groups,seed,float(cfg.get('validation_fraction',.15)),float(cfg.get('test_fraction',.15)))
    indices={s:[i for i,label in enumerate(labels) if label==s] for s in ('train','validation','test')}
    model=NativeWaveformResUNet(dataset.k,dataset.tb_bin,int(cfg.get('width',16)),float(cfg.get('dropout',.05)))
    device=torch.device(cfg.get('device','cuda' if torch.cuda.is_available() else 'cpu'));model.to(device)
    batch=int(cfg.get('batch_size',1));workers=int(cfg.get('workers',0))
    loaders={s:DataLoader(Subset(dataset,idx),batch_size=batch,shuffle=s=='train',num_workers=workers,
                          generator=torch.Generator().manual_seed(seed)) for s,idx in indices.items()}
    optimizer=torch.optim.AdamW(model.parameters(),lr=float(cfg.get('learning_rate',3e-4)),weight_decay=float(cfg.get('weight_decay',1e-4)))
    scheduler=torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,mode='min',factor=.5,patience=int(cfg.get('lr_patience',3)))
    epochs=int(cfg.get('epochs',50));patience=int(cfg.get('early_stop_patience',10))
    if epochs<1:raise ValueError('epochs must be positive')
    split_manifest={s:[dict(file_index=dataset.event_keys[i][0],entry=dataset.event_keys[i][1],source_group=dataset.groups[i]) for i in idx] for s,idx in indices.items()}
    provenance=dict(seed=seed,input_manifest=manifest,split_source_group_count={s:len(set(dataset.groups[i] for i in idx)) for s,idx in indices.items()},
                    split_event_count={s:len(idx) for s,idx in indices.items()},test_used_for_model_selection=False,
                    original_source_partition_verified=dataset.source_groups_verified,
                    physics_accuracy_evaluated=False,training_input_sha256=hashlib.sha256(json.dumps(manifest,sort_keys=True).encode()).hexdigest())
    (output/'split_manifest.json').write_text(json.dumps(split_manifest,indent=2))
    (output/'training_config.json').write_text(json.dumps(dict(config=cfg,provenance=provenance),indent=2))
    history=[];best=float('inf');stale=0;best_state=None;best_epoch=None
    def evaluate(loader):
        model.eval();total=0.;count=0
        with torch.no_grad():
            for x,target,charge in loader:
                x,target,charge=(v.to(device) for v in (x,target,charge))
                loss=pit_charge_loss(model(x),target,charge)
                total+=float(loss)*len(x);count+=len(x)
        return total/max(1,count)
    try:
        for epoch in range(epochs):
            model.train();train_sum=0.;train_count=0;epoch_start=perf_counter()
            for x,target,charge in loaders['train']:
                x,target,charge=(v.to(device) for v in (x,target,charge))
                optimizer.zero_grad();loss=pit_charge_loss(model(x),target,charge)
                if not torch.isfinite(loss):raise RuntimeError('Nonfinite finding training loss')
                loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),float(cfg.get('gradient_clip',5.)))
                optimizer.step();train_sum+=float(loss.detach())*len(x);train_count+=len(x)
                if int(cfg.get('progress_every',10))>0 and train_count%int(cfg.get('progress_every',10))==0:
                    print(json.dumps(dict(epoch=epoch+1,train_events=train_count,train_total=len(indices['train']),
                                          running_train_loss=train_sum/train_count,elapsed_seconds=perf_counter()-epoch_start)),flush=True)
            val=evaluate(loaders['validation']);scheduler.step(val)
            history.append(dict(epoch=epoch+1,train_loss=train_sum/max(1,train_count),validation_loss=val,
                                learning_rate=float(optimizer.param_groups[0]['lr']),epoch_seconds=perf_counter()-epoch_start))
            print(json.dumps(history[-1]),flush=True)
            if val<best:
                best=val;best_epoch=epoch+1;stale=0;best_state={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
                # During optimization this is explicitly incomplete/unusable by default inference.
                torch.save(dict(contract=model.contract(),model=best_state,training_completed=False,
                                training_provenance=provenance,epoch=epoch+1),output/'best_incomplete.pt')
            else:stale+=1
            (output/'history.json').write_text(json.dumps(history,indent=2))
            if stale>=patience:break
        if best_state is None:raise RuntimeError('No finite validation checkpoint')
        model.load_state_dict(best_state);test=evaluate(loaders['test'])
        provenance.update(completed_epochs=len(history),best_epoch=best_epoch,best_validation_loss=best,test_pit_loss=test,
                          test_evaluation_count=1,physics_accuracy_evaluated=False)
        checkpoint=output/'finding_best.pt'
        torch.save(dict(contract=model.contract(),model=best_state,training_completed=True,
                        training_provenance=provenance),checkpoint)
        (output/'result.json').write_text(json.dumps(dict(model_path=str(checkpoint),provenance=provenance),indent=2))
        return dict(model_path=str(checkpoint),history=history,provenance=provenance)
    finally:dataset.close()


def separation_metrics(predicted_fractions,truth_fractions,charge,overlap_charge_threshold=40.,min_visible_layers=2):
    """Permutation-aligned soft charge attribution metrics, background retained.

    L1 may exceed one (it is <=2 for exact simplexes). Recovery is a conditional
    visible-track criterion, never the detector geometric acceptance.
    """
    pred=np.asarray(predicted_fractions,dtype=float);truth=np.asarray(truth_fractions,dtype=float)
    charge=np.asarray(charge,dtype=float)
    if pred.shape!=truth.shape or pred.shape[1:]!=charge.shape or pred.ndim!=4:
        raise ValueError('Separation metric dimensions mismatch')
    if not np.isfinite(pred).all() or not np.isfinite(truth).all() or not np.isfinite(charge).all():
        raise ValueError('Nonfinite separation metric inputs')
    if np.any(pred<0) or np.any(truth<0) or np.any(charge<0):raise ValueError('Negative metric inputs')
    if not np.allclose(pred.sum(0),1.,atol=2e-5) or not np.allclose(truth.sum(0),1.,atol=2e-5):
        raise ValueError('Metrics require true simplexes, background last')
    k=pred.shape[0]-1;denom=max(float(charge.sum()),1e-12)
    best=min(permutations(range(k)),key=lambda perm:float((np.abs(pred[list(perm)+[k]]-truth)*charge[None]).sum()))
    aligned=pred[list(best)+[k]]
    error=np.abs(aligned-truth)*charge[None]
    overlap=(np.sum(truth[:k]>.1,axis=0)>=2)&(charge>overlap_charge_threshold)
    overlap_mae=float(np.abs(aligned[:k,overlap]-truth[:k,overlap]).mean()) if overlap.any() else float('nan')
    trueq=truth[:k]*charge[None];recoq=aligned[:k]*charge[None]
    intersection=np.minimum(trueq,recoq).sum((1,2,3))
    precision=intersection/np.maximum(recoq.sum((1,2,3)),1e-12)
    recall=intersection/np.maximum(trueq.sum((1,2,3)),1e-12)
    true_layers=(trueq.sum((2,3))>overlap_charge_threshold).sum(1)
    reco_layers=(recoq.sum((2,3))>overlap_charge_threshold).sum(1)
    visible=true_layers>=min_visible_layers
    recovered=visible&(precision>=.5)&(recall>=.5)&(reco_layers>=min_visible_layers)
    return dict(charge_l1=float(error.sum()/denom),absolute_charge_error=float(error.sum()),positive_charge=denom,
                overlap_fraction_mae=overlap_mae,overlap_voxels=int(overlap.sum()),slot_permutation=list(best),
                precision=precision,recall=recall,true_visible=visible,recovered=recovered,
                true_layers=true_layers,reco_layers=reco_layers,aligned_fractions=aligned.astype(np.float32))


def evaluate_finding_comparison(manifest,split_manifest,checkpoint,output_root,output_json,config=None):
    """Evaluate frozen methods on the SAME predeclared heldout source groups.

    All inference calls get observed waveforms alone. Truth is used afterwards
    only in this evaluation process, whose ROOT role=evaluation prevents downstream
    reuse as a reconstruction observation stage. No network optimization or
    hyperparameter retuning; per-event hypothesis fitting is observation-only.
    """
    from .stage_io import StageWriter,read_stage,manifest as stage_manifest,sha256,jsonable
    from .algorithms.finding import find_event
    from .evaluation import _reference_index
    cfg=dict(config or {});split=json.loads(Path(split_manifest).read_text())
    selected={(int(r['file_index']),int(r['entry'])) for r in split['test']}
    if not selected:raise ValueError('No predeclared test events')
    if any(selected&{(int(r['file_index']),int(r['entry'])) for r in split[name]} for name in ('train','validation')):
        raise ValueError('Event leakage across frozen split file')
    groups={name:{r['source_group'] for r in rows} for name,rows in split.items()}
    if groups['test']&groups['train'] or groups['test']&groups['validation']:
        raise ValueError('Source group leakage across frozen split file')
    obs_meta=stage_manifest(manifest['observations_root']);truth_meta=stage_manifest(manifest['truth_root'])
    if obs_meta.get('role')=='truth' or truth_meta.get('role')!='truth':raise ValueError('Observation/truth roles reversed')
    labels=_reference_index(manifest['truth_root'])
    if not selected<=set(labels):raise ValueError('Missing frozen-test truth events')
    methods=list(cfg.get('methods',['cnn','hough','ransac','tls','cnn_hough','waveform_hypotheses']))
    output_json=Path(output_json)
    if output_json.exists():raise FileExistsError(output_json)
    per_event=[];method_results={m:[] for m in methods}
    metainfo=dict(role='evaluation',evaluation_only=True,methods=methods,source_dataset_uuid=manifest['source_dataset_uuid'],
        checkpoint=str(Path(checkpoint).resolve()),checkpoint_sha256=sha256(checkpoint),split_manifest_sha256=sha256(split_manifest),
        original_source_partition_verified=bool(obs_meta.get('source_group_partition_verified',False)),
        evaluation_scope='Small old-MC heldout charge-attribution benchmark; not new DDE or physical 3D/energy/vertex accuracy',
        metric_definitions=dict(charge_l1='sum abs(predicted observed attributed charge - MC-tag attributed charge), including background / observed positive sum; may exceed 1',
             overlap_fraction_mae='alpha-slot mean absolute fractions on true >=2 fractions >0.1, coarse charge >40 ADC*TB',
             recovered='visible >=2 layers above40ADC*TB, soft precision/recall>=.5 and recovered>=2layers; conditional criterion'),
        physics_accuracy_evaluated=False,config=cfg)
    with StageWriter(output_root,'finding_evaluation',metainfo) as writer:
        for record,diagnostic in read_stage(manifest['observations_root']):
            key=int(record['file_index']),int(record['entry'])
            if key not in selected:continue
            q=record['positive_waveform'];position,source_group=labels[key]
            if int(record['source_group'])!=source_group:raise ValueError('Evaluation group mismatch')
            label,_=next(read_stage(manifest['truth_root'],allow_truth=True,entry_indices=[position]))
            if (int(label['file_index']),int(label['entry']))!=key or int(label['source_group'])!=source_group:
                raise ValueError('Reference row changed after scalar indexing')
            k=int(label['truth_components'].shape[0]);tb=int(cfg.get('tb_bin',truth_meta.get('tb_bin',4)))
            charge=q.reshape(12,64,512//tb,tb).sum(-1)
            truth=np.concatenate((label['truth_components'],label['truth_background'][None])).reshape(k+1,12,64,512//tb,tb).sum(-1)
            truth=np.divide(truth,charge[None],out=np.zeros_like(truth),where=charge[None]>0)
            truth[-1,charge==0]=1.
            probabilities=[];scores=[];diags={};timings=[]
            for method in methods:
                inference_cfg=dict(cfg.get('finding',{}),k=k,tb_bin=tb,method=method)
                # All hyperparameters are fixed before looking up this event's
                # target. Per-event waveform fitting uses observations only.
                if method in ('cnn','cnn_hough'):inference_cfg['checkpoint']=str(checkpoint)
                found=find_event(q,inference_cfg)
                metric=separation_metrics(found['fractions'],truth,charge)
                probabilities.append(metric.pop('aligned_fractions'));scores.append(metric)
                diags[method]=found['diagnostics'];timings.append(found['diagnostics']['seconds'])
                method_results[method].append(metric)
            arrays=dict(file_index=np.int64(key[0]),entry=np.int64(key[1]),source_group=record['source_group'],
                positive_waveform=q,reference_fractions=truth.astype(np.float32),
                predicted_fractions=np.asarray(probabilities,dtype=np.float32),
                charge_l1=np.asarray([m['charge_l1'] for m in scores]),
                overlap_fraction_mae=np.asarray([m['overlap_fraction_mae'] for m in scores]),
                recovered_slots=np.asarray([m['recovered'] for m in scores],dtype=np.uint8),
                visible_slots=np.asarray([m['true_visible'] for m in scores],dtype=np.uint8),
                precision=np.asarray([m['precision'] for m in scores]),recall=np.asarray([m['recall'] for m in scores]),
                method_seconds=np.asarray(timings))
            writer.write(arrays,dict(method_order=methods,inference_diagnostics=diags))
            per_event.append(dict(file_index=key[0],entry=key[1],metrics={m:jsonable(s) for m,s in zip(methods,scores)}))
            print('Frozen test comparison',key,{m:round(s['charge_l1'],5) for m,s in zip(methods,scores)},flush=True)
        summary={}
        for method,values in method_results.items():
            if len(values)!=len(selected):raise ValueError('Evaluation did not cover every frozen test event')
            overlap=np.asarray([x['overlap_fraction_mae'] for x in values]);finite=overlap[np.isfinite(overlap)]
            summary[method]=dict(N_events=len(values),mean_event_charge_l1=float(np.mean([x['charge_l1'] for x in values])),
                pooled_charge_l1=float(sum(x['absolute_charge_error'] for x in values)/sum(x['positive_charge'] for x in values)),
                mean_event_overlap_fraction_mae=float(finite.mean()) if len(finite) else None,
                overlap_events=len(finite),recovered_visible_slots=int(sum(x['recovered'].sum() for x in values)),
                true_visible_slots=int(sum(x['true_visible'].sum() for x in values)))
        writer.metrics({m+'_mean_event_charge_l1':v['mean_event_charge_l1'] for m,v in summary.items()})
        for method,values in method_results.items():
            writer.histogram(method+'_charge_l1',[x['charge_l1'] for x in values],bins=30,range=(0,2))
            writer.histogram(method+'_overlap_fraction_mae',[x['overlap_fraction_mae'] for x in values],bins=30,range=(0,1))
    result=dict(metadata=metainfo,summary=summary,per_event=per_event)
    with output_json.open('x') as stream:json.dump(jsonable(result),stream,indent=2,allow_nan=False)
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest',required=True,help='JSON with observations_root, truth_root, source_dataset_uuid')
    parser.add_argument('--output-dir',required=True)
    parser.add_argument('--config',help='Optional training JSON; explicit GPU/device/epochs/K/TB bin')
    args=parser.parse_args()
    result=train_finding(json.loads(Path(args.manifest).read_text()),args.output_dir,
                         json.loads(Path(args.config).read_text()) if args.config else None)
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
