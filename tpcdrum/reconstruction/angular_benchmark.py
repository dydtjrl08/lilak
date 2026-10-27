"""Fixed-cohort, truth-labelled angular-separation benchmark (evaluation only).

No reconstruction/model is called. A single before-stage charge permutation is
reused after refitting. Curves always retain the original eligible denominator;
conditional resolutions, candidate coverage and paired common-valid comparisons
are reported separately. Delta theta is a LAB u-v projection, never a COM cut.
"""
from pathlib import Path
from itertools import combinations
import json
import math
import numpy as np

from .stage_io import StageWriter,read_stage,manifest,sha256,jsonable
from .evaluation import _reference_index,_fractions

STRATA=('observed_drift_gap_tb','observed_shared_pad_fraction',
        'observed_visible_layers_min','observed_charge_imbalance')
ERROR_FIELDS=('theta_2d','theta_3d_accepted','theta_3d_candidate',
              'vertex_2d','vertex_3d_accepted','vertex_3d_candidate')


def configuration(config=None):
    c=dict(config or {})
    c.setdefault('angle_bins_deg',[0.,.5,1.,2.,3.,5.,10.,20.,180.])
    c.setdefault('pair_mode','minimum_pair')
    c.setdefault('purity_min',.5);c.setdefault('recall_min',.5)
    c.setdefault('merge_recall_min',.5);c.setdefault('split_recall_min',.2)
    c.setdefault('split_purity_min',.5)
    c.setdefault('layer_charge_threshold',40.);c.setdefault('minimum_visible_layers',2)
    c.setdefault('tb_bin',4);c.setdefault('wilson_z',1.959963984540054)
    c.setdefault('prefer_joint_same_model_theta',True)
    if c['pair_mode'] not in ('all_pairs','minimum_pair'):
        raise ValueError('pair_mode must be all_pairs or minimum_pair')
    for key in ('purity_min','recall_min','merge_recall_min','split_recall_min','split_purity_min'):
        if not 0<float(c[key])<=1:raise ValueError('Threshold must be in (0,1]: '+key)
    for name,edges in [('angle_bins_deg',c['angle_bins_deg'])]+list(c.get('strata_bins',{}).items()):
        a=np.asarray(edges,float)
        if a.ndim!=1 or len(a)<2 or not np.isfinite(a).all() or np.any(np.diff(a)<=0):
            raise ValueError('Finite increasing bin edges required: '+name)
    if (not 1<=int(c['minimum_visible_layers'])<=12 or
            not np.isfinite(float(c['layer_charge_threshold'])) or float(c['layer_charge_threshold'])<0):
        raise ValueError('Invalid visibility definition')
    if not np.isfinite(float(c['wilson_z'])) or float(c['wilson_z'])<=0:raise ValueError('Wilson z must be positive finite')
    if int(c['tb_bin'])<1 or 512%int(c['tb_bin']):raise ValueError('tb_bin must divide 512')
    if any(name not in STRATA for name in c.get('strata_bins',{})):
        raise ValueError('Unknown observed stratification field')
    return c


def wilson(success,total,z=1.959963984540054):
    """Marginal binomial Wilson interval; empty denominator is undefined."""
    success,total=int(success),int(total)
    if total<0 or not 0<=success<=total:raise ValueError('Invalid binomial counts')
    if not np.isfinite(z) or z<=0:raise ValueError('Wilson z must be positive finite')
    if total==0:return np.nan,np.nan,np.nan
    p=success/total;denom=1.+z*z/total
    center=(p+z*z/(2*total))/denom
    half=z*math.sqrt(p*(1-p)/total+z*z/(4*total*total))/denom
    return p,0. if success==0 else max(0.,center-half),1. if success==total else min(1.,center+half)


def _wrap(degrees):return (np.asarray(degrees,float)+180.)%360.-180.


def truth_pairs(reference,config=None):
    c=configuration(config)
    directions=np.asarray(reference['truth_direction_uvw'],float)
    if directions.ndim!=2 or directions.shape[1]!=3 or not np.isfinite(directions).all():
        raise ValueError('Need finite initial truth directions for LAB angle axis')
    if np.any(np.linalg.norm(directions[:,:2],axis=1)<=0):raise ValueError('Undefined projected truth angle')
    theta=np.degrees(np.arctan2(directions[:,0],directions[:,1]))
    pairs=np.asarray(list(combinations(range(len(theta)),2)),dtype=np.int64).reshape(-1,2)
    if not len(pairs):raise ValueError('Angular pair benchmark needs >=2 truth alphas')
    delta=np.abs(_wrap(theta[pairs[:,0]]-theta[pairs[:,1]]))
    if c['pair_mode']=='minimum_pair':
        index=int(np.argmin(delta));pairs=pairs[index:index+1];delta=delta[index:index+1]
    return pairs,delta,theta


def _attribution(record,reference,c,matching=None):
    """Global matching plus explicit many-to-one/one-to-many diagnostics."""
    target=np.asarray(reference['truth_components'],float)
    kt=len(target)
    if 'components' not in record:
        # A previous compact evaluation ROOT can be summarized, but cannot
        # reconstruct cross-slot merge/split information it did not store.
        if not {'matched_prediction_index','soft_precision','soft_recall'}<=set(record):
            raise ValueError('Method needs finding components or explicit compact paired-evaluation matching/purity/recall')
        match=np.asarray(record['matched_prediction_index'],int)
        if matching is not None and not np.array_equal(match,matching):
            raise ValueError('Compact after evaluation uses a different alpha permutation')
        if match.shape!=(kt,):raise ValueError('Invalid compact matching')
        precision=np.asarray(record.get('soft_precision',np.zeros(kt)),float)[:kt]
        recall=np.asarray(record.get('soft_recall',np.zeros(kt)),float)[:kt]
        visible=(target.sum((2,3))>c['layer_charge_threshold']).sum(1)>=c['minimum_visible_layers']
        good=visible&(match>=0)&(precision>=c['purity_min'])&(recall>=c['recall_min'])
        return dict(matching=match,precision=precision,recall=recall,visible=visible,good=good,
            merge=np.zeros((kt,kt),bool),split=np.zeros(kt,bool),structural_metrics_known=False,
            components=None,charge_l1=float(record.get('charge_l1',np.nan)))
    pred=np.asarray(record['components'],float)
    if pred.ndim!=4 or target.ndim!=4 or pred.shape[1:]!=(12,64,512) or target.shape[1:]!=pred.shape[1:]:
        raise ValueError('Attribution requires embedded native component waveforms')
    if not np.isfinite(pred).all() or np.any(pred<0) or not np.isfinite(target).all() or np.any(target<0):
        raise ValueError('Attribution waveforms must be finite and nonnegative')
    kp=len(pred)
    q=np.asarray(record.get('positive_waveform'),float)
    bg=np.asarray(record.get('background_waveform',record.get('background')),float)
    truth_bg=np.asarray(reference.get('truth_background'),float)
    if q.shape!=target.shape[1:] or bg.shape!=q.shape or truth_bg.shape!=q.shape:
        raise ValueError('Missing same-event positive waveform/background')
    if not np.isfinite(q).all() or not np.isfinite(bg).all() or np.any(q<0) or np.any(bg<0):
        raise ValueError('Invalid observed waveform/background')
    scale=max(1.,float(q.max()))
    if not np.allclose(pred.sum(0)+bg,q,atol=2e-6*scale,rtol=2e-6) or not np.allclose(target.sum(0)+truth_bg,q,atol=2e-6*scale,rtol=2e-6):
        raise ValueError('Observation/prediction/reference charge closure differs')
    if matching is None:
        if kp:
            base=_fractions(record,reference,int(c['tb_bin']))
            matching=base['matched_prediction_index'];charge_l1=base['charge_l1']
        else:
            matching=np.full(kt,-1)
            charge_l1=float((target.sum()+np.abs(bg-truth_bg).sum())/max(float(q.sum()),1e-12))
    else:
        matching=np.asarray(matching,int)
        if matching.shape!=(kt,):raise ValueError('Invalid fixed matching shape')
        # A removed slot remains a failed match; never rematch the surviving
        # after slots in a way that hides a lost track.
        matching=np.where(matching<kp,matching,-1)
        charge_l1=np.nan
    match=np.asarray(matching,int)
    if len(set(int(i) for i in match if i>=0))!=np.count_nonzero(match>=0):
        raise ValueError('Global alpha matching must be injective')
    true_total=target.sum((1,2,3));pred_total=pred.sum((1,2,3))
    tb=int(c['tb_bin'])
    coarse_pred=pred.reshape(kp,12,64,512//tb,tb).sum(-1)
    coarse_target=target.reshape(kt,12,64,512//tb,tb).sum(-1)
    intersection=np.array([[np.minimum(p,t).sum() for t in coarse_target] for p in coarse_pred]).reshape(kp,kt)
    by_recall=intersection/np.maximum(true_total[None],1e-12)
    by_precision=intersection/np.maximum(pred_total[:,None],1e-12)
    true_layers=(target.sum((2,3))>c['layer_charge_threshold']).sum(1)
    pred_layers=(pred.sum((2,3))>c['layer_charge_threshold']).sum(1)
    visible=true_layers>=c['minimum_visible_layers']
    precision=np.zeros(kt);recall=np.zeros(kt);good=np.zeros(kt,bool)
    for j,i in enumerate(match):
        if i<0:continue
        precision[j]=by_precision[i,j];recall[j]=by_recall[i,j]
        good[j]=visible[j] and precision[j]>=c['purity_min'] and recall[j]>=c['recall_min'] and pred_layers[i]>=c['minimum_visible_layers']
    merge=np.zeros((kt,kt),bool)
    active=pred_layers>=c['minimum_visible_layers']
    for a,b in combinations(range(kt),2):
        # Perfect one-to-one reconstructions with identical response shapes are
        # not called merged merely because their waveform supports coincide.
        recovered=good[a] and good[b] and match[a]!=match[b]
        merge[a,b]=merge[b,a]=not recovered and np.any(active&(by_recall[:,a]>=c['merge_recall_min'])&(by_recall[:,b]>=c['merge_recall_min']))
    used={int(match[j]):j for j in range(kt) if good[j]}
    split=np.zeros(kt,bool)
    for j in range(kt):
        pieces=[i for i in range(kp) if active[i] and (i not in used or used[i]==j) and
                by_recall[i,j]>=c['split_recall_min'] and by_precision[i,j]>=c['split_purity_min']]
        split[j]=len(pieces)>=2
    return dict(matching=match,precision=precision,recall=recall,visible=visible,good=good,
        merge=merge,split=split,structural_metrics_known=True,components=pred,charge_l1=charge_l1)


def _aligned(values,matching,width=None):
    result=np.full((len(matching),) if width is None else (len(matching),width),np.nan)
    a=np.asarray(values,float)
    if a.ndim!=(1 if width is None else 2) or (width is not None and a.shape[1]!=width):return result
    for j,i in enumerate(matching):
        if 0<=i<len(a):result[j]=a[i]
    return result


def _method_confidence(record,diagnostic):
    """Event-level method declaration: -1 unknown, 0 not ambiguous, 1 ambiguous.

    This is not a truth-derived recovery label or calibrated confidence level.
    If a scalar branch and diagnostics disagree, ambiguity wins conservatively.
    """
    supplied=[]
    if 'finding_ambiguous' in record:supplied.append(record['finding_ambiguous'])
    finding=diagnostic.get('finding',{})
    if isinstance(finding,dict) and 'ambiguous' in finding:supplied.append(finding['ambiguous'])
    known=[]
    for value in supplied:
        array=np.asarray(value)
        if array.size!=1 or array.dtype.kind not in 'biuf':continue
        flag=float(array.reshape(-1)[0])
        if np.isfinite(flag) and flag in (0.,1.):known.append(int(flag))
    return max(known) if known else -1


def _errors(record,diagnostic,reference,matching,c):
    kt=len(matching);truth_direction=np.asarray(reference['truth_direction_uvw'],float)
    theta=np.degrees(np.arctan2(truth_direction[:,0],truth_direction[:,1]))
    values={name:np.full(kt if name.startswith('theta') else 1,np.nan) for name in ERROR_FIELDS}
    physical=diagnostic.get('fitting',{}).get('physical_3d') is True or bool(record.get('physical_3d',False))
    direction=_aligned(record.get('track_directions_uvw',[]),matching,3)
    if physical:
        finite=np.isfinite(direction).all(1)&(np.linalg.norm(direction[:,:2],axis=1)>0)
        residual=_wrap(np.degrees(np.arctan2(direction[:,0],direction[:,1]))-theta)
        values['theta_3d_candidate'][finite]=residual[finite]
        ok=_aligned(np.asarray(record.get('fit_ok',[]),float),matching)
        valid=finite&(ok==1);values['theta_3d_accepted'][valid]=residual[valid]
    if 'theta_uv_ols_true_minus_fitted_deg' in record:
        values['theta_2d']=-np.asarray(record['theta_uv_ols_true_minus_fitted_deg'],float)[:kt]
    elif 'fit_theta_pad_2d_ols_deg' in record:
        values['theta_2d']=_wrap(_aligned(record['fit_theta_pad_2d_ols_deg'],matching)-theta)
    if 'theta_uv_true_minus_fitted_deg' in record and 'direction_valid' in record:
        compact=-np.asarray(record['theta_uv_true_minus_fitted_deg'],float)[:kt]
        valid=np.asarray(record['direction_valid'],bool)[:kt]
        if compact.shape==(kt,) and valid.shape==(kt,):
            values['theta_3d_accepted']=np.where(valid,compact,np.nan)
    target=np.asarray(reference.get('truth_vertex_uvw_mm',[]),float)
    if target.shape==(3,) and np.isfinite(target).all():
        point=np.asarray(record.get('vertex_uvw_mm',[]),float)
        if physical and point.shape==(3,) and np.isfinite(point).all() and int(record.get('vertex_dimension',3))==3:
            distance=float(np.linalg.norm(point-target));values['vertex_3d_candidate'][0]=distance
            if bool(record.get('vertex_valid',False)):values['vertex_3d_accepted'][0]=distance
        candidate=record.get('vertex_candidate_uvw_mm',diagnostic.get('vertex',{}).get('candidate_vertex_mm'))
        if physical and candidate is not None:
            candidate=np.asarray(candidate,float)
            if candidate.shape==(3,) and np.isfinite(candidate).all():values['vertex_3d_candidate'][0]=np.linalg.norm(candidate-target)
        p=np.asarray(record.get('vertex_projected_uv_mm',[]),float)
        if p.shape==(2,) and np.isfinite(p).all() and bool(record.get('vertex_projected_valid',False)):
            values['vertex_2d'][0]=np.linalg.norm(p-target[:2])
    for name,value,flag in [('vertex_3d_accepted','vertex_error_norm_mm','vertex_error_valid'),
                            ('vertex_3d_candidate','vertex_candidate_error_norm_mm','vertex_candidate_error_valid')]:
        if bool(record.get(flag,False)) and np.isfinite(record.get(value,np.nan)):
            values[name][0]=float(record[value])
    return values


def _strata(score,pairs,c):
    result=np.full((len(pairs),len(STRATA)),np.nan)
    comp=score['components'];match=score['matching']
    if comp is None:return result
    for r,(a,b) in enumerate(pairs):
        i,j=match[a],match[b]
        if min(i,j)<0:continue
        one,two=comp[i],comp[j];q1,q2=one.sum(),two.sum()
        if min(q1,q2)<=0:continue
        time=np.arange(512)
        gap=abs(float(one.sum((0,1))@time/q1)-float(two.sum((0,1))@time/q2))
        pads1,pads2=one.sum(2),two.sum(2)
        overlap=float(np.minimum(pads1,pads2).sum()/min(q1,q2))
        layers=min(int(np.count_nonzero(one.sum((1,2))>c['layer_charge_threshold'])),int(np.count_nonzero(two.sum((1,2))>c['layer_charge_threshold'])))
        result[r]=gap,overlap,layers,max(q1,q2)/(q1+q2)
    return result


def event_result(before,before_diagnostic,reference,after=None,after_diagnostic=None,config=None):
    """Compact one-method pair results; no waveform/model retained in output."""
    c=configuration(config);pairs,delta,theta=truth_pairs(reference,c)
    b=_attribution(before,reference,c)
    a=_attribution(after,reference,c,b['matching']) if after is not None else b
    be=_errors(before,before_diagnostic,reference,b['matching'],c)
    ae=_errors(after,after_diagnostic or {},reference,b['matching'],c) if after is not None else {n:np.full_like(v,np.nan) for n,v in be.items()}
    same=(after if after is not None else before)
    same_model=False
    same_vertex_comparison=False
    if c['prefer_joint_same_model_theta'] and 'vertex_free_theta_uv_deg' in same and 'vertex_joint_theta_uv_deg' in same:
        free=_aligned(same['vertex_free_theta_uv_deg'],b['matching'])
        joint=_aligned(same['vertex_joint_theta_uv_deg'],b['matching'])
        valid=_aligned(np.asarray(same.get('vertex_joint_comparison_valid',[]),float),b['matching'])==1
        be['theta_2d']=_wrap(free-theta)
        ae['theta_2d']=np.where(valid,_wrap(joint-theta),np.nan)
        same_model=True
        if after is None:ae['vertex_2d']=_errors(before,before_diagnostic,reference,b['matching'],c)['vertex_2d']
    if 'vertex_free_projected_uv_mm' in same:
        target=np.asarray(reference.get('truth_vertex_uvw_mm',[]),float)
        free=np.asarray(same['vertex_free_projected_uv_mm'],float)
        be['vertex_2d']=np.full(1,np.nan)
        if (target.shape==(3,) and np.isfinite(target).all() and free.shape==(2,) and
                np.isfinite(free).all() and bool(same.get('vertex_free_projected_valid',False))):
            be['vertex_2d'][0]=np.linalg.norm(free-target[:2])
        ae['vertex_2d']=_errors(same,after_diagnostic or before_diagnostic,reference,b['matching'],c)['vertex_2d']
        same_vertex_comparison=True
    out=dict(pair_indices=pairs,delta_theta_xy_deg=delta,matching=b['matching'],
             pair_visible=np.all(b['visible'][pairs],axis=1).astype(np.uint8),
             observed_strata=_strata(b,pairs,c),
             phase_available=np.array([1,after is not None or same_model or same_vertex_comparison],np.uint8))
    for phase,score,errors in [('before',b,be),('after',a,ae)]:
        available=bool(out['phase_available'][0 if phase=='before' else 1])
        confidence_record=before if phase=='before' else (after if after is not None else before)
        confidence_diag=before_diagnostic if phase=='before' else (after_diagnostic if after is not None else before_diagnostic)
        ambiguous=_method_confidence(confidence_record,confidence_diag or {}) if available else -1
        out['pair_recovered_'+phase]=np.array([bool(score['good'][i] and score['good'][j] and score['matching'][i]!=score['matching'][j]) for i,j in pairs],np.uint8)
        out['pair_method_ambiguous_'+phase]=np.full(len(pairs),ambiguous,dtype=np.int64)
        out['pair_method_confidence_assessed_'+phase]=np.full(len(pairs),ambiguous>=0,dtype=np.uint8)
        out['pair_accepted_recovered_'+phase]=(out['pair_recovered_'+phase]&(ambiguous==0)&available).astype(np.uint8)
        out['pair_merge_'+phase]=np.array([score['merge'][i,j] for i,j in pairs],np.uint8)
        out['pair_fake_split_'+phase]=np.array([score['split'][i] or score['split'][j] for i,j in pairs],np.uint8)
        out['structural_known_'+phase]=np.full(len(pairs),score['structural_metrics_known'],np.uint8)
        out['pair_purity_'+phase]=score['precision'][pairs]
        out['pair_recall_'+phase]=score['recall'][pairs]
        for name,value in errors.items():
            out[name+'_error_'+phase]=value[pairs] if name.startswith('theta') else np.repeat(value,len(pairs))
    return out,dict(matching='before-stage global charge matching reused after fit',same_model_projected_theta_comparison=same_model,
        projected_vertex_comparison='free local-track vertex-region point versus joint common vertex' if same_vertex_comparison else 'separately supplied stage vertices',
        residual_sign='reconstructed minus initial MC theta; vertex fields are Euclidean distance',
        theta_3d_label='u-v theta of calibrated physical 3D tangent, not 3D opening error',
        observed_strata='fixed baseline method observed assigned waveforms; evaluation matching labels these estimates; not MC timing/charge truth',
        perfect_overlap_identifiability='charge matching and purity do not independently establish a unique decomposition')


class _Stats:
    def __init__(self):self.n=0;self.mean=0.;self.m2=0.;self.squared=0.
    def add(self,values):
        for x in np.asarray(values,float).reshape(-1):
            if not np.isfinite(x):continue
            self.n+=1;d=x-self.mean;self.mean+=d/self.n;self.m2+=d*(x-self.mean);self.squared+=x*x
    def result(self,prefix):
        return {prefix+'_n':self.n,prefix+'_bias':self.mean if self.n else np.nan,
                prefix+'_std':math.sqrt(max(0.,self.m2/(self.n-1))) if self.n>1 else np.nan,
                prefix+'_rmse':math.sqrt(self.squared/self.n) if self.n else np.nan}


class AngularAccumulator:
    """Only bin moments/counts are retained; no corpus waveforms or hit arrays."""
    def __init__(self,methods,config=None):
        self.methods=list(methods);self.config=configuration(config);self.bins={};self.events=0
        self.outside_angle_range_pairs=0
        self.groups=[('all',None,None)]
        for name,edges in self.config.get('strata_bins',{}).items():
            self.groups.extend((name,j,(edges[j],edges[j+1])) for j in range(len(edges)-1))
            self.groups.append((name,-1,None))
        for m in range(len(self.methods)):
            for group_index,_ in enumerate(self.groups):
                for i in range(len(self.config['angle_bins_deg'])-1):
                    self.bins[m,group_index,i]=dict(all_pairs=0,eligible_pairs=0,counts={},stats={})
    def add(self,row):
        row={name:np.asarray(value) for name,value in row.items()}
        # JSON serializes undefined numeric results as null. Restore them as
        # NaN rather than accidentally counting null as a zero-error result.
        row={name:(np.asarray(value,dtype=float) if value.dtype.kind=='O' else value)
             for name,value in row.items()}
        self.events+=1;edges=np.asarray(self.config['angle_bins_deg'])
        for p,angle in enumerate(row['delta_theta_xy_deg']):
            if not np.isfinite(angle):continue
            binid=int(np.searchsorted(edges,angle,side='right')-1)
            if angle==edges[-1]:binid=len(edges)-2
            if not 0<=binid<len(edges)-1:self.outside_angle_range_pairs+=1;continue
            groups=[0]
            for g,(name,index,bounds) in enumerate(self.groups[1:],1):
                value=float(row['observed_strata'][p,STRATA.index(name)])
                all_edges=self.config['strata_bins'][name]
                if index==-1 and (not np.isfinite(value) or value<all_edges[0] or value>all_edges[-1]):groups.append(g)
                elif bounds is not None and np.isfinite(value):
                    upper=value<=bounds[1] if bounds[1]==self.config['strata_bins'][name][-1] else value<bounds[1]
                    if bounds[0]<=value and upper:groups.append(g)
            eligible=bool(row['pair_visible'][p])
            for m in range(len(self.methods)):
                for g in groups:
                    b=self.bins[m,g,binid];b['all_pairs']+=1
                    if not eligible:continue
                    b['eligible_pairs']+=1
                    for phase in ('before','after'):
                        available=bool(row.get('phase_available',np.ones((len(self.methods),2)))[m,0 if phase=='before' else 1])
                        for flag in ('pair_recovered','pair_merge','pair_fake_split',
                                     'pair_accepted_recovered','pair_method_ambiguous','pair_method_confidence_assessed'):
                            assessed=bool(row.get('pair_method_confidence_assessed_'+phase,np.zeros_like(row['pair_recovered_'+phase]))[m,p])
                            if flag=='pair_method_ambiguous':known=available and assessed
                            elif flag in ('pair_recovered','pair_accepted_recovered','pair_method_confidence_assessed'):known=available
                            else:known=available and bool(row['structural_known_'+phase][m,p])
                            if known:
                                key=flag+'_'+phase
                                b['counts'][key]=b['counts'].get(key,0)+int(row.get(key,np.zeros_like(row['pair_recovered_'+phase]))[m,p])
                                b['counts'][key+'_denom']=b['counts'].get(key+'_denom',0)+1
                        if not available:continue
                        for name in ('pair_purity','pair_recall'):
                            b['stats'].setdefault(name+'_'+phase,_Stats()).add(row[name+'_'+phase][m,p])
                        for name in ERROR_FIELDS:
                            b['stats'].setdefault(name+'_'+phase,_Stats()).add(row[name+'_error_'+phase][m,p])
                    for name in ERROR_FIELDS:
                        before=np.atleast_1d(row[name+'_error_before'][m,p]);after=np.atleast_1d(row[name+'_error_after'][m,p])
                        mask=np.isfinite(before)&np.isfinite(after)
                        for label,value in [('common_before',before[mask]),('common_after',after[mask]),
                                            ('paired_abs_error_change',np.abs(after[mask])-np.abs(before[mask]))]:
                            b['stats'].setdefault(name+'_'+label,_Stats()).add(value)
    def curves(self):
        rows=[];edges=self.config['angle_bins_deg']
        for (m,g,i),b in self.bins.items():
            group,index,bounds=self.groups[g]
            row=dict(method_index=m,stratum_group_index=g,angle_bin=i,angle_low_deg=edges[i],angle_high_deg=edges[i+1],
                     all_truth_pairs=b['all_pairs'],eligible_visible_pairs=b['eligible_pairs'])
            for phase in ('before','after'):
                available=b['counts'].get('pair_recovered_'+phase+'_denom',0)
                row['phase_available_'+phase+'_pair_count']=available
                row['phase_available_'+phase+'_fraction']=available/b['eligible_pairs'] if b['eligible_pairs'] else np.nan
                for flag in ('pair_recovered','pair_merge','pair_fake_split','pair_accepted_recovered',
                             'pair_method_ambiguous','pair_method_confidence_assessed'):
                    key=flag+'_'+phase;s=b['counts'].get(key,0);n=b['counts'].get(key+'_denom',0)
                    rate,low,high=wilson(s,n,float(self.config['wilson_z']))
                    row.update({key+'_count':s,key+'_denominator':n,key+'_fraction':rate,key+'_wilson_low':low,key+'_wilson_high':high})
                for name in ('pair_purity','pair_recall')+ERROR_FIELDS:
                    stat=b['stats'].get(name+'_'+phase,_Stats());moments=stat.result(name+'_'+phase)
                    if name not in ERROR_FIELDS:
                        moments[name+'_'+phase+'_mean']=moments.pop(name+'_'+phase+'_bias')
                        moments.pop(name+'_'+phase+'_rmse')
                    elif name.startswith('vertex'):
                        moments[name+'_'+phase+'_mean_distance_mm']=moments.pop(name+'_'+phase+'_bias')
                    row.update(moments)
                    if name in ERROR_FIELDS:
                        denom=b['eligible_pairs']*(2 if name.startswith('theta') else 1)
                        row[name+'_'+phase+'_coverage']=stat.n/denom if denom and available else np.nan
            for name in ERROR_FIELDS:
                for label in ('common_before','common_after','paired_abs_error_change'):
                    row.update(b['stats'].get(name+'_'+label,_Stats()).result(name+'_'+label))
                before=row[name+'_before_coverage'];after=row[name+'_after_coverage']
                row[name+'_after_coverage_decreased']=int(np.isfinite(before) and np.isfinite(after) and after<before)
            rows.append(row)
        return rows


def _method_inputs(methods):
    if not methods or len(set(methods))!=len(methods):raise ValueError('Need unique nonempty method names')
    result={}
    for name,value in methods.items():
        if isinstance(value,(str,Path)):value={'before':value}
        if not isinstance(value,dict) or 'before' not in value:raise ValueError('Each method needs a before ROOT')
        result[str(name)]={phase:str(Path(value[phase]).expanduser().resolve(strict=True)) for phase in ('before','after') if value.get(phase) is not None}
    return result


def _save_curves(writer,curves):
    keys=list(curves[0]);integer={k for k,v in curves[0].items() if isinstance(v,(int,np.integer))}
    tree=writer.file.mktree('angular_curves',{k:'int64' if k in integer else 'float64' for k in keys})
    tree.extend({k:np.asarray([r[k] for r in curves],dtype=np.int64 if k in integer else np.float64) for k in keys})
    # Native TH1 count views accompany the full curve TTree for the ROOT viewer.
    # Rates and undefined empty-bin values remain in angular_curves, not replaced
    # by visually misleading zero-efficiency histogram bins.
    global_rows=[r for r in curves if r['stratum_group_index']==0]
    for method in sorted({r['method_index'] for r in global_rows}):
        selected=sorted((r for r in global_rows if r['method_index']==method),key=lambda r:r['angle_bin'])
        edges=np.asarray([r['angle_low_deg'] for r in selected]+[selected[-1]['angle_high_deg']],float)
        for key in ('eligible_visible_pairs','pair_recovered_before_count','pair_accepted_recovered_before_count',
                    'pair_method_confidence_assessed_before_count','pair_method_ambiguous_before_count',
                    'pair_merge_before_count','pair_fake_split_before_count'):
            writer.file['performance/method_%d_%s'%(method,key)]=(np.asarray([r[key] for r in selected],float),edges)


def benchmark(methods,truth_root,output_root,output_json=None,config=None,figure_base=None):
    """Stream same-cohort stage ROOTs; save compact standalone evaluation ROOT.

    methods={name:path} or {name:{'before':path,'after':path}}. config.event_keys
    fixes a subset; otherwise every truth event is required in every method.
    Extra method rows are not silently used. Missing requested events are errors,
    not a success-only intersection. No full corpus is loaded into memory.
    """
    c=configuration(config);paths=_method_inputs(methods);names=list(paths)
    if output_json is not None and Path(output_json).exists():raise FileExistsError(output_json)
    tm=manifest(truth_root)
    if tm.get('role')!='truth':raise ValueError('Reference must be evaluation-only truth ROOT')
    if not tm.get('source_group_partition_verified',False) and not c.get('allow_unverified_source_groups',False):
        raise ValueError('Truth source groups are unverified')
    truth_index=_reference_index(truth_root)
    requested=[tuple(map(int,k)) for k in c['event_keys']] if c.get('event_keys') is not None else list(truth_index)
    if len(set(requested))!=len(requested):raise ValueError('Fixed cohort contains duplicate event identities')
    cohort=sorted(requested)
    if not cohort or not set(cohort)<=set(truth_index):raise ValueError('Empty or missing fixed truth cohort')
    namespace=tm.get('source_dataset_uuid')
    if not namespace:raise ValueError('Missing truth source namespace')
    indices={};source_records={}
    for name,phases in paths.items():
        indices[name]={}
        for phase,path in phases.items():
            meta=manifest(path)
            if meta.get('role') not in ('observations','evaluation') or meta.get('source_dataset_uuid')!=namespace:
                raise ValueError('Method/reference role or source namespace mismatch')
            if not meta.get('source_group_partition_verified',False) and not c.get('allow_unverified_source_groups',False):
                raise ValueError('Unverified source groups cannot establish fixed-cohort independence')
            index=_reference_index(path)
            if not set(cohort)<=set(index):raise ValueError('Method lacks fixed cohort events: '+name+'/'+phase)
            indices[name][phase]=index
            source_records[name+'/'+phase]=dict(path=path,manifest=meta,sha256=sha256(path))
    metadata=dict(role='evaluation',evaluation_only=True,source_dataset_uuid=namespace,
        source_group_partition_verified=bool(tm.get('source_group_partition_verified',False)),
        methods=names,method_inputs=source_records,truth_root=str(Path(truth_root).resolve()),truth_sha256=sha256(truth_root),
        config=c,pair_mode=c['pair_mode'],cohort_events=len(cohort),no_120_degree_constraint=True,
        thresholds_predeclared=bool(c.get('thresholds_frozen',False)),
        axis='abs(wrapped initial-MC theta_u_v difference) in LAB degrees; evaluation axis only',
        recovery='both truth tracks visible, injectively matched to distinct slots, frozen purity/recall/layer cuts',
        accepted_recovery='conditional MC recovery AND an explicit method finding_ambiguous=false declaration; unknown/not-assessed never counts as method-accepted',
        method_confidence='event-level method declaration, not calibrated confidence or MC truth; ambiguous branch/diagnostic conflicts conservatively count ambiguous',
        method_ambiguity_codes={'unknown_not_assessed':-1,'explicit_not_ambiguous':0,'ambiguous':1},
        ambiguity_curves='assessed fraction uses all phase-available eligible pairs; ambiguous fraction is conditional on an assessed declaration; accepted recovery retains the full eligible denominator',
        purity_recall='min(predicted,reference) attributed charge after the same configured TB aggregation used for global matching; missing slots contribute zero',
        merge='unrecovered pair with one visible predicted slot covering both truth charges above merge_recall_min',
        fake_split='one true alpha has >=2 significant predicted pieces after excluding slots successfully matched to other true alphas',
        structural_metrics='conditional attributed-charge diagnostics, not proof of information-theoretic identifiability',
        intervals='marginal Wilson binomial intervals; all_pairs repeats correlated pairs per event and source siblings may also correlate; this is not an independent-pair significance test',
        resolutions='conditional finite-result bias/sample-std/RMSE plus fixed-denominator coverage and paired common-valid changes; never claim improvement from success-only resolution',
        cohort_unique_source_groups=len({truth_index[key][1] for key in cohort}),
        residual_sign='reco minus truth for theta; Euclidean distance for vertex',
        theta_reference_caveat='first-observed fitted tangent versus initial MC production theta includes propagation/reference-plane differences; not pure detector resolution',
        strata='observed assigned-waveform estimates frozen from the first method before-stage; matching is evaluation-only',
        physics_accuracy_evaluated=False)
    accumulator=AngularAccumulator(names,c)
    metadata['stratum_groups']=accumulator.groups
    with StageWriter(output_root,'angular_benchmark',metadata) as writer:
        for key in cohort:
            position,group=truth_index[key]
            reference,_=next(read_stage(truth_root,allow_truth=True,entry_indices=[position]))
            method_values=[];method_notes=[]
            for name in names:
                phase_rows={}
                for phase,path in paths[name].items():
                    index,actual_group=indices[name][phase][key]
                    if actual_group!=group:raise ValueError('Source group changed between paired stage/reference')
                    phase_rows[phase]=next(read_stage(path,allow_truth=True,entry_indices=[index]))
                before,bd=phase_rows['before'];after,ad=phase_rows.get('after',(None,None))
                values,notes=event_result(before,bd,reference,after,ad,c)
                method_values.append(values);method_notes.append(notes)
            first=method_values[0]
            arrays=dict(file_index=np.int64(key[0]),entry=np.int64(key[1]),source_group=np.int64(group),
                pair_indices=first['pair_indices'],delta_theta_xy_deg=first['delta_theta_xy_deg'],
                pair_visible=first['pair_visible'],observed_strata=first['observed_strata'])
            for name in first:
                if name in ('pair_indices','delta_theta_xy_deg','pair_visible','observed_strata'):continue
                arrays[name]=np.stack([v[name] for v in method_values])
            if any(not np.array_equal(v['pair_visible'],first['pair_visible']) for v in method_values):
                raise ValueError('Truth-visible denominator differs across method representations')
            accumulator.add(arrays);writer.write(arrays,dict(methods=method_notes,truth_for_evaluation_only=True))
        curves=accumulator.curves();_save_curves(writer,curves)
        writer.metrics({'cohort_events':accumulator.events,'methods':len(names),
                        'outside_angle_range_pairs':accumulator.outside_angle_range_pairs,'physics_accuracy_evaluated':0.})
    summary=dict(metadata=metadata,curves=curves,cohort_events=accumulator.events)
    if output_json is not None:
        with Path(output_json).open('x') as stream:json.dump(jsonable(summary),stream,indent=2,allow_nan=False)
    if figure_base is not None:plot_benchmark(output_root,figure_base)
    return summary


def summarize_rows(rows,methods,config=None):
    """Summarize already compact benchmark result rows, without upstream files."""
    accumulator=AngularAccumulator(methods,config)
    for item in rows:accumulator.add(item[0] if isinstance(item,tuple) else item)
    return dict(events=accumulator.events,curves=accumulator.curves())


def plot_benchmark(input_root,figure_base):
    """Replot the saved numeric curves alone; no upstream data/model access."""
    import uproot
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    meta=manifest(input_root)
    if meta.get('stage')!='angular_benchmark' or meta.get('role')!='evaluation':raise ValueError('Not an angular benchmark ROOT')
    base=Path(figure_base).expanduser();base.parent.mkdir(parents=True,exist_ok=True)
    confidence_base=base.with_name(base.stem+'_confidence')
    for prefix in (base,confidence_base):
        for suffix in ('.png','.pdf'):
            if prefix.with_suffix(suffix).exists():raise FileExistsError(prefix.with_suffix(suffix))
    with uproot.open(input_root) as source:curves=source['angular_curves'].arrays(library='np')
    fig,axes=plt.subplots(2,3,figsize=(15,8),layout='constrained')
    occupied=(curves['stratum_group_index']==0)&(curves['all_truth_pairs']>0)
    extent=(float(curves['angle_low_deg'][occupied].min()),float(curves['angle_high_deg'][occupied].max())) if occupied.any() else (0.,180.)
    for m,name in enumerate(meta['methods']):
        color=plt.get_cmap('tab10')(m%10)
        select=(curves['method_index']==m)&(curves['stratum_group_index']==0)
        x=(curves['angle_low_deg'][select]+curves['angle_high_deg'][select])/2
        halfwidth=(curves['angle_high_deg'][select]-curves['angle_low_deg'][select])/2
        for ax,key,label in [(axes[0,0],'pair_recovered_before','Conditional MC pair recovery'),(axes[0,1],'pair_merge_before','Pair merge'),(axes[0,2],'pair_fake_split_before','Pair fake split')]:
            y=curves[key+'_fraction'][select];low=curves[key+'_wilson_low'][select];high=curves[key+'_wilson_high'][select]
            ax.errorbar(x,y,xerr=halfwidth,yerr=np.vstack((y-low,high-y)),marker='o',label=name,capsize=2,color=color);ax.set_ylim(-.02,1.02);ax.set_ylabel(label)
        for phase,style in [('before','--'),('after','-')]:
            marker='o' if phase=='before' else 's'
            axes[1,0].plot(x,curves['theta_2d_'+phase+'_rmse'][select],style,marker=marker,label=name+' '+phase,color=color)
            axes[1,1].plot(x,curves['theta_2d_'+phase+'_coverage'][select],style,marker=marker,label=name+' '+phase,color=color)
            axes[1,2].plot(x,curves['vertex_2d_'+phase+'_rmse'][select],style,marker=marker,label=name+' '+phase,color=color)
    for ax in axes.flat:ax.set_xlabel(r'Initial truth $|\Delta\theta_{uv}|$ [deg]');ax.grid(alpha=.2);ax.set_xlim(*extent)
    axes[0,0].legend(fontsize=8);axes[1,0].legend(fontsize=7)
    axes[1,0].set_ylabel('Conditional projected theta RMSE [deg]')
    axes[1,0].set_ylim(bottom=0.)
    axes[1,1].set_ylabel('Projected theta coverage (fixed denominator)');axes[1,1].set_ylim(-.02,1.02)
    axes[1,2].set_ylabel('Conditional projected vertex RMSE [mm]')
    axes[1,2].set_ylim(bottom=0.)
    fig.suptitle('Same fixed cohort; '+meta['pair_mode']+'; no 120-degree constraint\nResolutions are conditional: compare coverage and paired common-valid curves as well',fontsize=12)
    for suffix in ('.png','.pdf'):fig.savefig(base.with_suffix(suffix),dpi=160)
    plt.close(fig)
    result=dict(png=str(base.with_suffix('.png')),pdf=str(base.with_suffix('.pdf')))
    if 'pair_accepted_recovered_before_fraction' in curves:
        fig,axes=plt.subplots(1,3,figsize=(15,4.5),layout='constrained')
        specifications=[('pair_accepted_recovered_before','Method-accepted MC pair recovery'),
                        ('pair_method_confidence_assessed_before','Ambiguity declaration available'),
                        ('pair_method_ambiguous_before','Ambiguous among declared cases')]
        for m,name in enumerate(meta['methods']):
            select=(curves['method_index']==m)&(curves['stratum_group_index']==0)
            x=(curves['angle_low_deg'][select]+curves['angle_high_deg'][select])/2
            width=(curves['angle_high_deg'][select]-curves['angle_low_deg'][select])/2
            for ax,(key,label) in zip(axes,specifications):
                y=curves[key+'_fraction'][select];low=curves[key+'_wilson_low'][select];high=curves[key+'_wilson_high'][select]
                ax.errorbar(x,y,xerr=width,yerr=np.vstack((y-low,high-y)),marker='o',capsize=2,label=name,
                            color=plt.get_cmap('tab10')(m%10))
                ax.set_ylabel(label);ax.set_xlabel(r'Initial truth $|\Delta\theta_{uv}|$ [deg]')
                ax.set_ylim(-.02,1.02);ax.set_xlim(*extent);ax.grid(alpha=.2)
        axes[0].legend(fontsize=8)
        fig.suptitle('Method acceptance is separate from conditional MC recovery\nUnknown confidence is never accepted; a method declaration is not a calibrated certainty',fontsize=12)
        for suffix in ('.png','.pdf'):fig.savefig(confidence_base.with_suffix(suffix),dpi=160)
        plt.close(fig)
        result.update(confidence_png=str(confidence_base.with_suffix('.png')),confidence_pdf=str(confidence_base.with_suffix('.pdf')))
    return result
