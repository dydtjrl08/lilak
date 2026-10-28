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
    k=pred.shape[0]-1;denom=float(charge.sum())
    if not 1<=k<=6 or not np.isfinite(denom):raise ValueError('Need K in [1,6] and finite observed charge sum')
    has_charge=denom>0.
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
    if not has_charge:
        precision[:]=np.nan;recall[:]=np.nan
    return dict(charge_l1=float(error.sum()/denom) if has_charge else float('nan'),
                charge_l1_defined=has_charge,event_has_positive_charge=has_charge,zero_charge=not has_charge,
                true_visible_count=int(visible.sum()),nonzero_low_visible=bool(has_charge and visible.sum()<k),
                absolute_charge_error=float(error.sum()),positive_charge=denom,
                overlap_fraction_mae=overlap_mae,overlap_voxels=int(overlap.sum()),slot_permutation=list(best),
                precision=precision,recall=recall,true_visible=visible,recovered=recovered,
                true_layers=true_layers,reco_layers=reco_layers,aligned_fractions=aligned.astype(np.float32))


def _finding_coverage_summary(values, expected_k):
    """Fixed-cohort counts plus explicitly conditional charge metrics.

    A missing prediction is not replaced by a perfect or background-only score.
    The pooled denominator includes only positive-charge events with a defined
    prediction; its charge/event coverage relative to ALL frozen cases is kept.
    """
    count=len(values)
    active=[v for v in values if v['positive_charge']>0]
    scored=[v for v in active if v.get('charge_l1_defined',np.isfinite(v['charge_l1'])) and np.isfinite(v['charge_l1'])]
    overlap=[v['overlap_fraction_mae'] for v in values if np.isfinite(v['overlap_fraction_mae'])]
    numerator=float(sum(v['absolute_charge_error'] for v in scored))
    denominator=float(sum(v['positive_charge'] for v in scored))
    total_charge=float(sum(v['positive_charge'] for v in values))
    nvisible=[int(v.get('true_visible_count',np.count_nonzero(v['true_visible']))) for v in values]
    visible_counts={str(k):sum(n==k for n in nvisible) for k in range(expected_k+1)}
    visible_slots=int(sum(np.count_nonzero(v['true_visible']) for v in values))
    recovered_slots=int(sum(np.count_nonzero(v['recovered']) for v in values))
    failure=sum(bool(v.get('method_failed',False)) for v in values)
    ambiguous=sum(bool(v.get('ambiguity_assessment_known',False)) and bool(v.get('method_ambiguous',True)) for v in values)
    unknown=sum(not bool(v.get('ambiguity_assessment_known',False)) for v in values)
    declared=sum(bool(v.get('method_declaration_success',False)) for v in values)
    active_mean=float(np.mean([v['charge_l1'] for v in scored])) if scored else None
    angular=np.concatenate([np.asarray(v.get('initial_minus_component_centroid_ols_theta_error_deg',[]),float) for v in values]) if values else np.empty(0)
    angular=angular[np.isfinite(angular)]
    pair_visible=sum(int(np.count_nonzero(v.get('pair_reference_visible',[]))) for v in values)
    pair_recovered=sum(int(np.count_nonzero(v.get('pair_recovered',[]))) for v in values)
    strata=dict(zero_charge=sum(v['positive_charge']==0 for v in values),
        positive_charge_zero_visible=sum(v['positive_charge']>0 and n==0 for v,n in zip(values,nvisible)),
        positive_charge_partially_visible=sum(v['positive_charge']>0 and 0<n<expected_k for v,n in zip(values,nvisible)),
        positive_charge_all_visible=sum(v['positive_charge']>0 and n==expected_k for v,n in zip(values,nvisible)))
    return dict(N_events=count,fixed_cohort_events=count,positive_charge_events=len(active),
        active_events=len(active),zero_charge_events=count-len(active),
        nonzero_low_visible_events=sum(v['positive_charge']>0 and n<expected_k for v,n in zip(values,nvisible)),
        visibility_count_events=visible_counts,exclusive_charge_visibility_strata=strata,
        mean_active_event_charge_l1=active_mean,mean_event_charge_l1=active_mean,
        mean_event_charge_l1_scope='compatibility alias for positive-charge events with defined predictions; zero/failed scores excluded',
        scored_active_events=len(scored),unscored_active_events=len(active)-len(scored),
        active_score_event_coverage=len(scored)/len(active) if active else None,
        pooled_charge_l1=numerator/denominator if denominator>0 else None,
        pooled_absolute_charge_error=numerator,pooled_scored_positive_charge=denominator,
        all_cohort_positive_charge=total_charge,
        pooled_charge_coverage=denominator/total_charge if total_charge>0 else None,
        mean_event_overlap_fraction_mae=float(np.mean(overlap)) if overlap else None,overlap_events=len(overlap),
        recovered_visible_slots=recovered_slots,true_visible_slots=visible_slots,
        recovered_visible_slot_fraction=recovered_slots/visible_slots if visible_slots else None,
        method_failed_events=int(failure),ambiguous_events=int(ambiguous),ambiguity_unknown_events=int(unknown),
        method_declaration_success_events=int(declared),
        method_declaration_success_fraction_all_cohort=declared/count if count else None,
        all_truth_slots_recovered_events=sum(bool(np.all(v['recovered'])) and v['positive_charge']>0 for v in values),
        fixed_all_reference_slots=count*expected_k,
        component_centroid_ols_valid_slots=sum(int(np.count_nonzero(v.get('component_centroid_ols_valid',[]))) for v in values),
        paired_initial_theta_error_slots=len(angular),
        initial_theta_error_slot_coverage=len(angular)/(count*expected_k) if count and expected_k else None,
        conditional_initial_minus_component_centroid_ols_theta_bias_deg=float(angular.mean()) if len(angular) else None,
        conditional_initial_minus_component_centroid_ols_theta_std_deg=float(angular.std(ddof=0)) if len(angular) else None,
        conditional_initial_minus_component_centroid_ols_theta_rmse_deg=float(np.sqrt(np.mean(angular**2))) if len(angular) else None,
        angular_diagnostic_scope='whole found-component centroid OLS, peak>40ADC on >=3 observed layers, then evaluation charge-PIT alignment; initial-vs-observed transport difference included; projected u-v only, no physical 3D drift claim',
        angular_std_definition='population RMS about conditional mean; descriptive, not SEM or calibrated precision',
        fixed_all_reference_pairs=count*expected_k*(expected_k-1)//2,
        reference_visible_pairs=pair_visible,recovered_reference_visible_pairs=pair_recovered,
        recovered_reference_visible_pair_fraction=pair_recovered/pair_visible if pair_visible else None,
        positive_charge_definition='sum of observed positive ADC > 0; not a visibility or physical acceptance claim',
        declaration_scope='non-ambiguous supported method declaration, not an independently validated reconstruction probability')


def _finding_observation_record(record):
    # Explicit allowlist: provenance/MC tags/truth/Si alpha energies and upstream
    # diagnostics cannot become sequential-finding inputs through this evaluator.
    names=('positive_waveform','tb_ns','points_uvw_mm','point_covariance_mm2',
           'point_charge','point_tb','point_layer','point_pad','physical_3d',
           'baseline','noise_sigma')
    return {name:record[name] for name in names if name in record}


def _evaluation_fractions(found, k, shape):
    fractions=np.asarray(found['fractions'],dtype=float)
    if fractions.ndim!=4 or fractions.shape[1:]!=shape or not 0<=len(fractions)-1<=k:
        raise ValueError('Finding returned unsupported slot count or coarse geometry')
    if len(fractions)==k+1:return fractions
    # A competing smaller-K hypothesis gets zero dummy alpha slots, while its
    # background stays last. This does not force the method to find all K tracks.
    padded=np.zeros((k+1,*shape));padded[:len(fractions)-1]=fractions[:-1];padded[-1]=fractions[-1]
    return padded


def _annotate_finding_score(metric,diagnostics,prediction_valid):
    known=isinstance(diagnostics.get('ambiguous'),(bool,np.bool_))
    ambiguous=bool(diagnostics.get('ambiguous',True))
    status=str(diagnostics.get('status','missing_assessment'))
    failed=(not prediction_valid or bool(diagnostics.get('optimization_incomplete',False))
        or diagnostics.get('optimizer_success') is False or status.startswith('invalid_') or status.endswith('_failed'))
    supported=status in ('hypothesis_supported_NOT_physics_validated',
                        'sequential_observation_hypothesis_supported_NOT_physics_validated')
    metric.update(prediction_valid=bool(prediction_valid),method_failed=bool(failed),
        method_ambiguous=ambiguous,ambiguity_assessment_known=known,method_status=status,
        method_declaration_success=bool(metric['positive_charge']>0 and prediction_valid and not failed and known and not ambiguous and supported))
    if not prediction_valid:
        metric.update(charge_l1=float('nan'),charge_l1_defined=False,absolute_charge_error=0.,
                      overlap_fraction_mae=float('nan'),precision=np.full_like(metric['precision'],np.nan),
                      recall=np.full_like(metric['recall'],np.nan),recovered=np.zeros_like(metric['recovered']),
                      reco_layers=np.zeros_like(metric['reco_layers']),slot_permutation=[-1]*len(metric['recovered']))
    return metric


def _reference_projected_angles(label,k):
    """Evaluation-only initial projected angles; no physical-z reconstruction."""
    angles=np.full(k,np.nan);valid=np.zeros(k,bool)
    pairs=np.asarray([(a,b) for a in range(k) for b in range(a+1,k)],dtype=np.int64).reshape(-1,2)
    try:directions=np.asarray(label.get('truth_direction_uvw'),dtype=float)
    except (TypeError,ValueError):directions=np.empty((0,3))
    if directions.shape==(k,3):
        valid=np.isfinite(directions).all(1)&(np.linalg.norm(directions[:,:2],axis=1)>0)
        angles[valid]=np.degrees(np.arctan2(directions[valid,0],directions[valid,1]))
    pair_delta=np.full(len(pairs),np.nan)
    for j,(a,b) in enumerate(pairs):
        if valid[a] and valid[b]:pair_delta[j]=abs((angles[a]-angles[b]+180.)%360.-180.)
    return dict(initial_theta_pad_deg=angles,valid=valid,pair_indices=pairs,pair_delta_theta_pad_deg=pair_delta,
                status='ok' if np.all(valid) else 'missing_or_invalid_initial_direction_reference')


def _component_centroid_ols_angles(found,k,permutation,prediction_valid):
    """Observed components -> centroids -> OLS, then evaluation slot alignment.

    Fixed BEFORE optimization: component layer peak >40 ADC, >=3 such layers.
    Every positive pad/TB contribution enters its layer centroid. No MC-selected
    hits, timing conversion, refit, vertex/120-degree constraint or truth angle
    enters this diagnostic. The charge-PIT permutation is evaluation-only.
    """
    from .preprocessing import PAD_U_MM,LAYER_V_MM
    angles=np.full(k,np.nan);layers=np.zeros(k,dtype=np.int64)
    result=dict(theta_pad_deg=angles,valid=np.zeros(k,bool),active_layers=layers,
                status='prediction_unavailable',min_observed_layers=3,component_layer_peak_adc_strict=40.)
    if not prediction_valid:return result
    if not isinstance(found,dict) or 'components' not in found:
        result['status']='missing_full_observed_components';return result
    try:components=np.asarray(found['components'],dtype=float)
    except (TypeError,ValueError):
        result['status']='invalid_full_observed_components';return result
    if (components.ndim!=4 or components.shape[1:]!=(12,64,512) or
            not 0<=len(components)<=k or not np.isfinite(components).all() or np.any(components<0)):
        result['status']='invalid_full_observed_components';return result
    perm=np.asarray(permutation,dtype=int)
    if perm.shape!=(k,) or sorted(perm.tolist())!=list(range(k)):
        result['status']='invalid_evaluation_charge_permutation';return result
    for slot,component in enumerate(components):
        pad_charge=component.sum(axis=2)
        charge=pad_charge.sum(axis=1)
        active=(component.max(axis=(1,2))>40.)&(charge>0)
        layers[slot]=int(active.sum())
        if layers[slot]<3:continue
        u=(pad_charge[active]@PAD_U_MM)/charge[active]
        v=LAYER_V_MM[active];center=float(v.mean());dv=v-center
        denominator=float(dv@dv)
        if denominator<=0:continue
        slope=float(dv@(u-u.mean())/denominator)
        angles[slot]=np.degrees(np.arctan(slope))
    result.update(theta_pad_deg=angles[perm].copy(),active_layers=layers[perm].copy(),
                  valid=np.isfinite(angles[perm]),status='ok' if np.isfinite(angles).any() else 'insufficient_observed_component_layers')
    return result


def _comparison_output_paths(manifest,split_manifest,checkpoint,output_root,output_json,methods,observation_outputs):
    """Preflight paths before opening ROOT inputs or starting any writer."""
    from collections.abc import Mapping
    def resolved(value):return Path(value).expanduser().resolve()
    inputs=[resolved(manifest['observations_root']),resolved(manifest['truth_root']),
            resolved(split_manifest),resolved(checkpoint)]
    if len(set(inputs))!=len(inputs):raise ValueError('Comparison input roles must have distinct paths')
    root,json_path=resolved(output_root),resolved(output_json)
    outputs={}
    if observation_outputs is not None:
        if not isinstance(observation_outputs,Mapping) or set(observation_outputs)!=set(methods):
            raise ValueError('observation_outputs must match the complete configured method set')
        outputs={method:resolved(observation_outputs[method]) for method in methods}
    targets=[root,json_path]+list(outputs.values())
    if len(set(targets))!=len(targets) or set(targets)&set(inputs):
        raise ValueError('Comparison/handoff output paths collide with input or output roles')
    for path in targets:
        if path.exists() or path.is_symlink():raise FileExistsError(path)
    return root,json_path,outputs


def evaluate_finding_comparison(manifest,split_manifest,checkpoint,output_root,output_json,config=None,*,observation_outputs=None):
    """Evaluate every frozen event, including zero-charge and low-visible cases.

    Inference is observation-only. Truth is used only after prediction, in this
    evaluation process. Failed method calls remain in fixed-cohort coverage with
    undefined scores; no optimizer/model/dataset-cache behavior is changed here.
    observation_outputs optionally publishes independent observation-only finding
    ROOT files from these SAME predictions; no second inference or truth alignment.
    """
    from contextlib import ExitStack
    from .stage_io import StageWriter,read_stage,manifest as stage_manifest,sha256,jsonable
    from .algorithms.finding import find_event
    from .evaluation import _reference_index
    cfg=dict(config or {})
    methods=list(cfg.get('methods',['cnn','hough','ransac','tls','cnn_hough','waveform_hypotheses']))
    if not methods or len(set(methods))!=len(methods):raise ValueError('Methods must be nonempty and unique')
    output_root,output_json,handoff_paths=_comparison_output_paths(manifest,split_manifest,checkpoint,
        output_root,output_json,methods,observation_outputs)
    split=json.loads(Path(split_manifest).read_text())
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
    per_event=[];method_results={m:[] for m in methods}
    metainfo=dict(role='evaluation',evaluation_only=True,methods=methods,source_dataset_uuid=manifest['source_dataset_uuid'],
        checkpoint=str(Path(checkpoint).resolve()),checkpoint_sha256=sha256(checkpoint),split_manifest_sha256=sha256(split_manifest),
        fixed_cohort_events=len(selected),original_source_partition_verified=bool(obs_meta.get('source_group_partition_verified',False)),
        evaluation_scope='Frozen old-MC heldout charge-attribution benchmark; not new DDE or physical 3D/energy/vertex accuracy',
        metric_definitions=dict(charge_l1='sum absolute attributed-charge error including background / actual observed positive sum; undefined at zero charge; may exceed 1',
             mean_active_event_charge_l1='conditional on positive observed charge AND defined prediction; score event/charge coverage reported separately',
             pooled_charge_l1='sum finite scored error numerator / scored positive observed charge; None if denominator zero; failures retained in full-cohort counts',
             observed_positive_charge='actual sum, including exactly zero; never a clamped denominator',
             visibility_count_events='ALL frozen events, including zero charge; visible slots require >=2 layers each >40 ADC*TB',
             overlap_fraction_mae='alpha-slot mean absolute fractions on true >=2 fractions >0.1, coarse charge >40 ADC*TB',
             recovered='visible >=2 layers above40ADC*TB, soft precision/recall>=.5 and recovered>=2layers; conditional criterion',
             method_declaration_success='positive charge plus non-ambiguous supported method declaration; not physical reconstruction accuracy'),
        angular_diagnostic=dict(method='observed component layer-centroid OLS u(v)',min_observed_layers=3,strict_component_layer_peak_adc=40.,
            target='separate truth initial theta_pad = atan2(direction_u,direction_v)',
            alignment='same evaluation charge-PIT permutation; zero-padded absent tracks remain NaN',
            residual='wrapped true_initial - component_centroid_OLS degrees; contains initial/local transport difference',
            physical_3d=False,truth_used_for_hit_selection=False,truth_available_only_in_evaluation_output=True),
        physics_accuracy_evaluated=False,config=cfg)
    metainfo['observation_outputs']={method:str(path) for method,path in handoff_paths.items()}
    maximum_k=None
    with ExitStack() as contexts:
        writer=contexts.enter_context(StageWriter(output_root,'finding_evaluation',metainfo))
        handoff=None
        if handoff_paths:
            from .finding_handoff import FindingHandoff
            handoff=contexts.enter_context(FindingHandoff(handoff_paths,obs_meta,config=cfg,
                checkpoint_sha256=metainfo['checkpoint_sha256'],
                split_manifest_sha256=metainfo['split_manifest_sha256']))
        for record,diagnostic in read_stage(manifest['observations_root']):
            key=int(record['file_index']),int(record['entry'])
            if key not in selected:continue
            q=record['positive_waveform'];position,source_group=labels[key]
            if int(record['source_group'])!=source_group:raise ValueError('Evaluation group mismatch')
            label,_=next(read_stage(manifest['truth_root'],allow_truth=True,entry_indices=[position]))
            if (int(label['file_index']),int(label['entry']))!=key or int(label['source_group'])!=source_group:
                raise ValueError('Reference row changed after scalar indexing')
            k=int(label['truth_components'].shape[0]);tb=int(cfg.get('tb_bin',truth_meta.get('tb_bin',4)))
            if maximum_k is None:maximum_k=k
            if k!=maximum_k:raise ValueError('Fixed-cohort finding benchmark requires the same declared maximum K per event')
            charge=q.reshape(12,64,512//tb,tb).sum(-1)
            truth=np.concatenate((label['truth_components'],label['truth_background'][None])).reshape(k+1,12,64,512//tb,tb).sum(-1)
            truth=np.divide(truth,charge[None],out=np.zeros_like(truth),where=charge[None]>0)
            truth[-1,charge==0]=1.
            # Reference scoring is deferred until after a fresh observation
            # handoff, so no truth-PIT/visibility/angle annotation can precede it.
            reference_metric=None;reference_angles=None
            probabilities=[];scores=[];diags={};timings=[]
            for method in methods:
                inference_cfg=dict(cfg.get('finding',{}),k=k,tb_bin=tb,method=method)
                if method in ('cnn','cnn_hough'):inference_cfg['checkpoint']=str(checkpoint)
                started=perf_counter();predicted_slot_count=-1;evaluation_padding_count=0;found=None;inference_error=None
                try:
                    if method=='sequential_vertex':
                        from .algorithms.sequential_finding import find_event as sequential_find_event
                        found=sequential_find_event(_finding_observation_record(record),inference_cfg)
                    else:found=find_event(q,inference_cfg)
                except Exception as error:
                    inference_error=error
                if handoff is not None:
                    # Fresh native slots are persisted BEFORE truth-based PIT,
                    # reference angles, visibility scores, or evaluation padding.
                    # Schema/closure/real I/O errors deliberately propagate and
                    # abort all still-temporary output contexts.
                    if inference_error is None:
                        handoff.write(method,record,found=found,observation_diagnostic=diagnostic)
                    else:
                        handoff.write(method,record,error=inference_error,observation_diagnostic=diagnostic)
                if reference_metric is None:
                    # Invalid references are data-integrity failures, not a
                    # method failure; ExitStack aborts every temporary output.
                    reference_metric=separation_metrics(truth,truth,charge)
                    reference_angles=_reference_projected_angles(label,k)
                try:
                    if inference_error is not None:raise inference_error
                    raw_fractions=np.asarray(found['fractions'])
                    if raw_fractions.ndim==4:predicted_slot_count=int(len(raw_fractions)-1)
                    fractions=_evaluation_fractions(found,k,charge.shape)
                    evaluation_padding_count=k-predicted_slot_count
                    metric=separation_metrics(fractions,truth,charge)
                    method_diag=dict(found.get('diagnostics',{}));prediction_valid=True
                    aligned=metric.pop('aligned_fractions')
                except Exception as error:
                    # Keep the frozen event; NaN prediction and undefined score
                    # cannot masquerade as zero error. No fallback method runs.
                    method_diag=dict(status='evaluation_method_failed',ambiguous=True,
                        error_type=type(error).__name__,error_message=str(error),seconds=perf_counter()-started)
                    metric={name:(value.copy() if isinstance(value,np.ndarray) else value)
                            for name,value in reference_metric.items() if name!='aligned_fractions'}
                    aligned=np.full((k+1,*charge.shape),np.nan,dtype=np.float32);prediction_valid=False
                method_diag['evaluation_slot_alignment']=dict(predicted_slot_count=predicted_slot_count,
                    reference_maximum_k=k,zero_padded_alpha_slots=evaluation_padding_count,
                    scope='evaluation matching only; padded zero slots are not found tracks; background unchanged')
                metric.update(predicted_slot_count=predicted_slot_count,evaluation_padding_count=evaluation_padding_count)
                metric=_annotate_finding_score(metric,method_diag,prediction_valid)
                angular=_component_centroid_ols_angles(found,k,metric['slot_permutation'],prediction_valid)
                angle_error=np.full(k,np.nan)
                angle_valid=angular['valid']&reference_angles['valid']
                angle_error[angle_valid]=(reference_angles['initial_theta_pad_deg'][angle_valid]-angular['theta_pad_deg'][angle_valid]+180.)%360.-180.
                pair_visible=np.asarray([bool(metric['true_visible'][a] and metric['true_visible'][b]) for a,b in reference_angles['pair_indices']],bool)
                pair_recovered=np.asarray([bool(pair_visible[j] and metric['recovered'][a] and metric['recovered'][b]) for j,(a,b) in enumerate(reference_angles['pair_indices'])],bool)
                metric.update(predicted_component_centroid_ols_theta_pad_deg=angular['theta_pad_deg'],
                    component_centroid_ols_valid=angular['valid'],component_centroid_ols_active_layers=angular['active_layers'],
                    initial_minus_component_centroid_ols_theta_error_deg=angle_error,
                    pair_reference_visible=pair_visible,pair_recovered=pair_recovered)
                method_diag['component_centroid_ols']=dict(status=angular['status'],min_observed_layers=3,strict_layer_peak_adc=40.,
                    reference_status=reference_angles['status'],initial_local_transport_difference_included=True,physical_3d=False,
                    truth_used_for_hit_selection=False,alignment='evaluation charge-PIT permutation only')
                probabilities.append(aligned);scores.append(metric)
                diags[method]=method_diag;timings.append(float(method_diag.get('seconds',perf_counter()-started)))
                method_results[method].append(metric)
            visible_count=int(scores[0]['true_visible_count']);positive_charge=float(reference_metric['positive_charge'])
            arrays=dict(file_index=np.int64(key[0]),entry=np.int64(key[1]),source_group=record['source_group'],
                positive_waveform=q,reference_fractions=truth.astype(np.float32),
                predicted_fractions=np.asarray(probabilities,dtype=np.float32),positive_charge=np.float64(positive_charge),
                observed_zero_charge=np.uint8(positive_charge==0),reference_visible_count=np.int64(visible_count),
                nonzero_low_visible=np.uint8(positive_charge>0 and visible_count<k),
                reference_initial_theta_pad_deg=reference_angles['initial_theta_pad_deg'],
                reference_initial_theta_pad_valid=reference_angles['valid'].astype(np.uint8),
                reference_pair_indices=reference_angles['pair_indices'],
                reference_pair_delta_theta_pad_deg=reference_angles['pair_delta_theta_pad_deg'],
                predicted_component_centroid_ols_theta_pad_deg=np.asarray([m['predicted_component_centroid_ols_theta_pad_deg'] for m in scores]),
                initial_minus_component_centroid_ols_theta_error_deg=np.asarray([m['initial_minus_component_centroid_ols_theta_error_deg'] for m in scores]),
                component_centroid_ols_valid=np.asarray([m['component_centroid_ols_valid'] for m in scores],dtype=np.uint8),
                component_centroid_ols_active_layers=np.asarray([m['component_centroid_ols_active_layers'] for m in scores],dtype=np.int64),
                pair_reference_visible=np.asarray([m['pair_reference_visible'] for m in scores],dtype=np.uint8),
                pair_recovered=np.asarray([m['pair_recovered'] for m in scores],dtype=np.uint8),
                charge_l1=np.asarray([m['charge_l1'] for m in scores]),
                charge_l1_defined=np.asarray([m['charge_l1_defined'] for m in scores],dtype=np.uint8),
                prediction_valid=np.asarray([m['prediction_valid'] for m in scores],dtype=np.uint8),
                predicted_slot_count=np.asarray([m['predicted_slot_count'] for m in scores],dtype=np.int64),
                evaluation_padding_count=np.asarray([m['evaluation_padding_count'] for m in scores],dtype=np.int64),
                method_failed=np.asarray([m['method_failed'] for m in scores],dtype=np.uint8),
                method_ambiguous=np.asarray([m['method_ambiguous'] for m in scores],dtype=np.uint8),
                ambiguity_assessment_known=np.asarray([m['ambiguity_assessment_known'] for m in scores],dtype=np.uint8),
                method_declaration_success=np.asarray([m['method_declaration_success'] for m in scores],dtype=np.uint8),
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
            summary[method]=_finding_coverage_summary(values,maximum_k)
        root_metrics={}
        for method,values in summary.items():
            for name in ('fixed_cohort_events','positive_charge_events','zero_charge_events','nonzero_low_visible_events',
                         'scored_active_events','unscored_active_events','mean_active_event_charge_l1','pooled_charge_l1',
                         'pooled_charge_coverage','method_failed_events','ambiguous_events','ambiguity_unknown_events',
                         'method_declaration_success_fraction_all_cohort','fixed_all_reference_slots',
                         'component_centroid_ols_valid_slots','paired_initial_theta_error_slots','initial_theta_error_slot_coverage',
                         'conditional_initial_minus_component_centroid_ols_theta_bias_deg',
                         'conditional_initial_minus_component_centroid_ols_theta_std_deg',
                         'conditional_initial_minus_component_centroid_ols_theta_rmse_deg',
                         'fixed_all_reference_pairs','reference_visible_pairs','recovered_reference_visible_pairs','recovered_reference_visible_pair_fraction'):
                root_metrics[method+'_'+name]=np.nan if values[name] is None else values[name]
            for n,count in values['visibility_count_events'].items():root_metrics[method+'_all_cohort_visible_'+n+'_events']=count
        writer.metrics(root_metrics)
        for method,values in method_results.items():
            writer.histogram(method+'_positive_charge_defined_event_charge_l1',
                [v['charge_l1'] for v in values if v['positive_charge']>0 and v['charge_l1_defined']],bins=30,range=(0,2))
            writer.histogram(method+'_defined_overlap_fraction_mae',[v['overlap_fraction_mae'] for v in values],bins=30,range=(0,1))
            writer.histogram(method+'_conditional_initial_minus_component_centroid_ols_theta_error_deg',
                np.concatenate([v['initial_minus_component_centroid_ols_theta_error_deg'] for v in values]),bins=60)
            writer.histogram(method+'_all_cohort_reference_visible_count',[v['true_visible_count'] for v in values],
                bins=maximum_k+1,range=(-.5,maximum_k+.5))
            writer.histogram(method+'_all_cohort_zero_charge',[v['zero_charge'] for v in values],bins=2,range=(-.5,1.5))
            writer.histogram(method+'_all_cohort_method_failed',[v['method_failed'] for v in values],bins=2,range=(-.5,1.5))
            writer.histogram(method+'_all_cohort_method_declaration_success',[v['method_declaration_success'] for v in values],bins=2,range=(-.5,1.5))
    result=dict(metadata=metainfo,summary=summary,per_event=per_event)
    output_json.parent.mkdir(parents=True,exist_ok=True)
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
