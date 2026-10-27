"""Observation-only clean-seed -> soft-peel -> pair-region -> joint-waveform search.

The pair region is a DATA-DERIVED proposal/gating heuristic, not an independent
measurement added to the likelihood. Final amplitudes and geometry always fit
the original, unpeeled training waveforms, with whole neighboring pads held out.
No MC genealogy, 120-degree condition or equal-energy condition is accepted.
"""
from collections.abc import Mapping
from time import perf_counter
import numpy as np

from . import finding as wf
from . import geometry_core as geo
from ..preprocessing import drift_coordinate

HISTORY_COLUMNS=('stage','pair','seed','k','support_layers','clean_layers','clean_fraction',
                 'peeled_charge_fraction','selection_score','heldout_chi2','region_d2','eligible')
STAGES={'proposal':0,'first_seed':1,'soft_peel':2,'second_seed':3,'pair_fit':4,'pair_region':5,
        'third_proposal':6,'joint_test':7,'selected':8,'fallback':9,'fragment_rejected':10}


def _configuration(config):
    cfg=dict(config or {});seq=dict(cfg.get('sequential',{}))
    cfg.setdefault('k',3);cfg.setdefault('tb_bin',4);cfg.setdefault('activity_threshold_adc',40.)
    if isinstance(cfg['k'],bool) or int(cfg['k'])!=cfg['k'] or not 1<=int(cfg['k'])<=3:
        raise ValueError('Sequential finding supports maximum k in [1,2,3]')
    waveform=dict(cfg.get('waveform',{}));waveform.setdefault('nonlinear_max_nfev',8)
    waveform.setdefault('optimization_tb_stride',4);waveform.setdefault('max_hypotheses',6)
    cfg['waveform']=waveform
    alpha=float(seq.get('peel_fraction',.8))
    if not 0<alpha<1:raise ValueError('sequential.peel_fraction must be strictly between 0 and 1; hard erasure is forbidden')
    seq['peel_fraction']=alpha
    ks=seq.get('k_hypotheses',list(range(1,int(cfg['k'])+1)))
    if any(isinstance(k,bool) or int(k)!=k for k in ks) or not ks or set(ks)-{1,2,3}:
        raise ValueError('sequential.k_hypotheses must be an explicit subset of [1,2,3]')
    seq['k_hypotheses']=sorted(set(map(int,ks)))
    if max(ks)>cfg['k']:raise ValueError('k_hypotheses exceeds configured maximum k')
    if 3 in ks and 2 not in ks:raise ValueError('Three-track sequential finding requires a competing K=2 hypothesis')
    return cfg,seq


def _train_mask(cfg):
    options=cfg['waveform'];modulo=int(options.get('holdout_modulo',4));fold=int(options.get('holdout_fold',0))
    if modulo<2 or not 0<=fold<modulo:raise ValueError('Invalid whole-pad holdout rule')
    return (np.arange(64)[None,:]+2*np.arange(12)[:,None])%modulo!=fold


def soft_peel(original,prediction,confidence,fraction=.8):
    """Subtract only a confidence-weighted share of an EXPLAINED component.

    Every originally positive voxel remains strictly positive for fraction<1;
    a shared pad, a whole waveform, and unexplained charge are never deleted.
    This residual is a proposal aid ONLY, never the final fitting observation.
    """
    q=np.asarray(original,float);p=np.asarray(prediction,float);c=np.asarray(confidence,float)
    if q.shape!=p.shape or c.shape!=q.shape or not np.isfinite(q).all() or not np.isfinite(p).all() or not np.isfinite(c).all():
        raise ValueError('Soft peel requires aligned finite observed/predicted/confidence arrays')
    if np.any(q<0) or np.any(p<0) or np.any(c<0) or np.any(c>1) or not 0<fraction<1:
        raise ValueError('Invalid nonnegative/confidence/fraction contract')
    removed=fraction*c*np.minimum(p,q)
    return np.maximum(q-removed,0.),removed


def _longest_run(mask):
    best=current=0
    for x in mask:
        current=current+1 if x else 0;best=max(best,current)
    return best


def _pulse_align(parameters,q,template,seq):
    """Collapse time-tail fragments onto a local observed pulse-template maximum.

    This can merge unresolved coincident seeds, which is deliberately treated as
    lack of independent seed evidence, not proof that there is only one alpha.
    """
    p=np.asarray(parameters,float).copy();vs=np.arange(12)*12.-66.;observations=[]
    window=int(seq.get('pulse_peak_search_tb',24));sigma=float(seq.get('seed_pad_sigma_mm',1.2))
    shifts=np.arange(-window,window+1)
    for layer,v in enumerate(vs):
        expected_u=p[0]+p[1]*v;expected_t=p[2]+p[3]*v
        rows=np.abs(wf.PAD_U_MM-expected_u)<float(seq.get('seed_radius_mm',3.))
        signal=q[layer,rows].sum(0)
        if signal.max(initial=0)<=0:continue
        scores=[]
        for shift in shifts:
            kernel=np.interp(np.arange(512)-(expected_t+shift),template[0],template[1],left=0.,right=0.)
            scores.append(float(signal@kernel)/max(float(np.linalg.norm(kernel)),1e-12))
        observations.append((v,expected_t+shifts[int(np.argmax(scores))],max(scores)))
    if len(observations)>=2:
        a=np.asarray(observations);design=np.column_stack((np.ones(len(a)),a[:,0]))
        weight=np.sqrt(np.maximum(a[:,2],1e-9));fit=np.linalg.lstsq(design*weight[:,None],a[:,1]*weight,rcond=None)[0]
        p[2:]=fit
    return p


def _proposal_points(q,record,cfg,mode):
    if mode=='direct_waveform':return wf._coarse_points(q,cfg)[:2]
    needed=('point_layer','point_pad','point_tb','point_charge','points_uvw_mm')
    if not all(k in record for k in needed):raise ValueError('space_points proposal mode requires embedded measured point fields')
    layer=np.asarray(record['point_layer'],int);pad=np.asarray(record['point_pad'],int)
    tb=np.asarray(record['point_tb'],float);charge=np.asarray(record['point_charge'],float)
    xyz=np.asarray(record['points_uvw_mm'],float)
    if xyz.shape!=(len(layer),3) or any(a.shape!=(len(layer),) for a in (pad,tb,charge)):
        raise ValueError('Space-point proposal fields have incompatible shapes')
    if np.any(layer<0) or np.any(layer>=12) or np.any(pad<0) or np.any(pad>=64) or not np.isfinite(tb).all():
        raise ValueError('Invalid measured point native-pad/layer/time identity')
    original=np.asarray(record['positive_waveform'],float)
    ratio=np.divide(q.sum(-1),original.sum(-1),out=np.zeros((12,64)),where=original.sum(-1)>0)
    weight=charge*ratio[layer,pad]
    good=(weight>0)&np.isfinite(xyz[:,:2]).all(1)
    # Even a calibrated point record uses the SAME explicitly diagnostic time
    # feature for these geometric proposals. Physical w is handled separately.
    points=np.column_stack((xyz[:,0],xyz[:,1],tb*cfg['feature_time_scale']))
    return points[good],weight[good]


def _seed_quality(p,q,template,cfg,seq):
    options=dict(cfg['waveform'],pad_sigma_mm=float(seq.get('seed_pad_sigma_mm',cfg['waveform'].get('pad_sigma_mm',1.2))))
    mask=q.max(-1)>0
    # At least one train pad exists in every layer; zero-signal layers still
    # contribute a valid zero target. Never fit a heldout pad's amplitude.
    mask|=_train_mask(cfg)
    fit=wf._amplitude_fit(np.asarray(p)[None],q,mask,template,options,keep=True)
    model=fit['forward_components'][0]
    confidence=np.zeros_like(q);supported=np.zeros(12,bool);clean=np.zeros(12,bool);quality=np.zeros(12)
    for layer,v in enumerate(np.arange(12)*12.-66.):
        u=p[0]+p[1]*v;t=p[2]+p[3]*v
        core=(np.abs(wf.PAD_U_MM[:,None]-u)<float(seq.get('seed_core_radius_mm',2.)))
        core=core&(np.interp(np.arange(512)-t,template[0],template[1],left=0.,right=0.)[None,:]>.2)
        core=core&_train_mask(cfg)[layer,:,None]
        observed=q[layer][core].sum()
        if observed<float(seq.get('minimum_seed_layer_charge',80.)):continue
        supported[layer]=True
        error=float(np.abs(q[layer]-model[layer])[core].sum()/max(observed,1e-12))
        quality[layer]=max(0.,1.-error)
        clean[layer]=quality[layer]>=float(seq.get('minimum_clean_layer_fraction',.65))
        if clean[layer]:
            confidence[layer]=quality[layer]*np.minimum(1.,np.divide(model[layer],q[layer]+1e-12))
            confidence[layer]*=_train_mask(cfg)[layer,:,None]
    fraction=float(np.minimum(model,q).sum()/max(q.sum(),1e-12))
    eligible=bool(clean.sum()>=int(seq.get('minimum_clean_seed_layers',4)) and
                  _longest_run(clean)>=int(seq.get('minimum_consecutive_seed_layers',3)) and
                  fraction>=float(seq.get('minimum_seed_charge_fraction',.025)))
    score=float(np.sum(quality[clean])+np.log1p(np.minimum(model,q).sum())*.1)
    return dict(parameters=np.asarray(p),prediction=model,confidence=confidence,supported=supported,clean=clean,
                quality=quality,charge_fraction=fraction,eligible=eligible,rank_score=score)


def _same_fragment(a,b,seq):
    v=np.arange(12)*12.-66.
    du=np.sqrt(np.mean(((a[0]-b[0])+(a[1]-b[1])*v)**2))
    dt=np.sqrt(np.mean(((a[2]-b[2])+(a[3]-b[3])*v)**2))
    return bool(du<float(seq.get('fragment_u_rms_mm',.75)) and dt<float(seq.get('fragment_time_rms_tb',3.)))


def _pool(q,record,cfg,seq,template,mode,prior=None):
    points,charge=_proposal_points(q,record,cfg,mode)
    if len(points)<2:return []
    candidates=[];k=int(seq.get('proposal_slots',5));geometry=cfg.get('geometry',{'scales_mm':[2.,6.,4.]})
    for method in ('hough','ransac'):
        result=geo.infer(points,charge,k,method=method,seed=int(cfg.get('seed',20261027)),config=geometry)
        candidates.extend(wf._lines_to_parameters(result['lines'],cfg['feature_time_scale']))
    if prior is not None:
        pp,cc,active=wf._coarse_points(q,cfg)
        if len(pp):
            result=geo.refine(pp,cc,prior[:,active].T,lines=None,config=geometry)
            candidates.extend(wf._lines_to_parameters(result['lines'],cfg['feature_time_scale']))
    ranked=[]
    for p in candidates:
        p=_pulse_align(p,q,template,seq)
        if any(_same_fragment(p,old['parameters'],seq) for old in ranked):continue
        ranked.append(_seed_quality(p,q,template,cfg,seq))
    ranked.sort(key=lambda x:x['rank_score'],reverse=True)
    return ranked[:int(seq.get('maximum_seed_pool',8))]


def _calibration(cfg,record):
    cal=dict(cfg.get('drift',{}))
    if cal.get('calibrated_for_charge_mean') is not True:return None,'missing_charge_mean_drift_calibration_projected_uv_only'
    try:
        drift_coordinate(np.array([0.,1.]),cal)
        if 'tb_ns' in record and not np.isclose(float(record['tb_ns']),float(cal['tb_ns'])):
            raise ValueError('Observed TB duration and drift calibration differ')
        delay=np.asarray(cal.get('pad_delay_tb',np.zeros((12,64))),float)
        if delay.shape==(64,):delay=np.broadcast_to(delay,(12,64)).copy()
        if delay.shape!=(12,64) or not np.isfinite(delay).all():raise ValueError('Invalid explicit pad_delay_tb')
        cal['_delay']=delay
        return cal,'explicit_charge_mean_drift_calibration'
    except (ValueError,KeyError,TypeError) as error:return None,'invalid_drift_calibration: '+str(error)


def _local_state(component,cal,seq):
    charge=component.sum((1,2));ids=np.flatnonzero(charge>float(seq.get('minimum_seed_layer_charge',80.)))
    n=int(seq.get('local_tangent_layers',3))
    if len(ids)<n:raise ValueError('Insufficient distinct layers for a local upstream tangent')
    ids=ids[:n];v=ids*12.;reference=float(v[0]);dimension=3 if cal is not None else 2
    states=4 if dimension==3 else 2
    normal=np.zeros((states,states));rhs=np.zeros(states)
    for layer,vi in zip(ids,v):
        q=component[layer];total=q.sum();pw=q.sum(1)/total;u=float(pw@wf.PAD_U_MM)
        uvar=float(pw@((wf.PAD_U_MM-u)**2+wf.PAD_WIDTH_MM**2/12.))
        if dimension==2:
            if not np.isfinite(uvar) or uvar<=0:raise ValueError('Invalid native sampling/pulse-spread proposal covariance')
            h=np.array([1.,vi-reference]);normal+=np.outer(h,h)/uvar;rhs+=h*u/uvar
        else:
            times=np.arange(512)[None,:]-cal['_delay'][layer,:,None]
            tb=float((q*times).sum()/total);w=float(drift_coordinate(tb,cal));scale=float(cal['sign'])*float(cal['tb_ns'])*float(cal['velocity_mm_per_ns'])
            vv=float((q*(times-tb)**2).sum()/total)+1./12.
            uw=float((q*(wf.PAD_U_MM[:,None]-u)*(times-tb)).sum()/total)*scale
            cov=np.array([[uvar,uw],[uw,vv*scale*scale]])
            if not np.isfinite(cov).all() or np.linalg.eigvalsh(cov)[0]<=0:raise ValueError('Singular native sampling/pulse-spread proposal covariance')
            inverse=np.linalg.inv(cov);h=np.array([[1.,0.,vi-reference,0.],[0.,1.,0.,vi-reference]])
            normal+=h.T@inverse@h;rhs+=h.T@inverse@np.array([u,w])
    if np.linalg.matrix_rank(normal)<states:raise ValueError('Local tangent sampling covariance is singular')
    covariance=np.linalg.inv(normal);state=covariance@rhs
    if dimension==2:origin=np.array([state[0],reference]);direction=np.array([state[1],1.])
    else:origin=np.array([state[0],reference,state[1]]);direction=np.array([state[2],1.,state[3]])
    return origin,direction,covariance,reference,ids


def _pair_region(parameters,q_train,template,width,cfg,seq,cal,record):
    from .joint_vertex import vertex_region
    fit=wf._amplitude_fit(parameters,q_train,_train_mask(cfg),template,dict(cfg['waveform'],pad_sigma_mm=width),keep=True)
    model=fit['forward_components'];den=model.sum(0)+fit['forward_background']
    den+=np.maximum(q_train-den,0.)
    responsibility=np.divide(model,den[None],out=np.zeros_like(model),where=den[None]>0)
    components=responsibility*q_train[None]
    local=[_local_state(c,cal,seq) for c in components]
    origins=np.asarray([x[0] for x in local]);directions=np.asarray([x[1] for x in local]);covariance=np.asarray([x[2] for x in local]);references=np.asarray([x[3] for x in local])
    vertex=dict(seq.get('vertex',{}));vertex['physical_3d']=cal is not None
    if 'source_group' in record:vertex.setdefault('track_source_groups',[int(record['source_group'])]*2)
    region=vertex_region(origins,directions,covariance,references,vertex,dimension=3 if cal is not None else 2)
    region['state_covariance_scope']='sampling/cloud-spread proxy at earliest observed local tangent, NOT a calibrated centroid precision'
    region['pair_information_reused_as_independent_likelihood']=False
    return region,local


def _region_samples(region):
    envelope=region.get('transverse_region_envelope') or {}
    v=np.asarray(envelope.get('v_samples_mm',[]),float)
    center=np.asarray(envelope.get('centers_mm',[]),float);cov=np.asarray(envelope.get('covariances_mm2',[]),float)
    if v.ndim!=1 or not len(v):return v,center,cov
    intervals=region.get('v_intervals_mm')
    if intervals is None and region.get('v_interval_mm') is not None:intervals=[region['v_interval_mm']]
    if intervals is not None:
        keep=np.zeros(len(v),bool)
        for bounds in intervals:
            if np.asarray(bounds).shape==(2,):keep|=(v>=bounds[0]-1e-7)&(v<=bounds[1]+1e-7)
        if center.shape[0]==len(v) and cov.shape[0]==len(v):v,center,cov=v[keep],center[keep],cov[keep]
    return v,center,cov


def region_compatibility(parameters,region,cal,template,seq):
    """Minimum trajectory-to-region distance, conditional/proposal metric only."""
    if region.get('status')=='outside_reaction_volume':
        return dict(available=False,eligible=False,min_d2=None,status='pair_outside_explicit_reaction_volume')
    v,center,cov=_region_samples(region)
    dimension=int(region.get('dimension',3 if cal is not None else 2))
    transverse=2 if dimension==3 else 1
    if v.ndim!=1 or center.shape!=(len(v),transverse) or cov.shape!=(len(v),transverse,transverse) or not len(v):
        return dict(available=False,eligible=None,min_d2=None,status='missing_pair_region_envelope')
    p=np.asarray(parameters,float)
    u=p[0]+p[1]*(v-66.)
    curve=u[:,None]
    if dimension==3:
        if cal is None:return dict(available=False,eligible=None,min_d2=None,status='physical_region_without_calibration_refused')
        centroid=float(np.average(template[0],weights=template[1]))
        mean_tb=p[2]+p[3]*(v-66.)+centroid
        mean_tb-=_delay_at_uv(u,v,cal)
        curve=np.column_stack((u,drift_coordinate(mean_tb,cal)))
    distance=[]
    floor=float(seq.get('third_region_sigma_floor_mm',.5))
    for delta,c in zip(curve-center,cov):
        if not np.isfinite(delta).all() or not np.isfinite(c).all():distance.append(np.inf);continue
        distance.append(float(delta@np.linalg.pinv(c+np.eye(transverse)*floor*floor)@delta))
    good=np.asarray(distance);minimum=float(good.min(initial=np.inf))
    if not np.isfinite(minimum):return dict(available=False,eligible=None,min_d2=None,status='invalid_pair_region_envelope')
    open_sides=any(region.get('open_v_sides',[False,False]))
    resolved=region.get('status')=='resolved' and not open_sides
    eligible=minimum<=float(seq.get('third_region_gate_sigma',4.))**2
    return dict(available=True,eligible=bool(eligible) if resolved else None,min_d2=minimum,
                status='resolved_region_gate' if resolved else 'extended_region_soft_proposal_only',within_scanned_envelope=bool(eligible))


def _anchors(residual,record,cfg,seq,mode):
    if mode=='space_points':
        points,weight=_proposal_points(residual,record,cfg,mode)
        values=np.column_stack((points[:,:2],points[:,2]/cfg['feature_time_scale'],points[:,2]/cfg['feature_time_scale'],weight))
    else:
        values=[]
        for layer in range(12):
            peaks=residual[layer].max(1)
            for row in np.argsort(peaks)[-int(seq.get('anchor_pads_per_layer',3)):]:
                pulse=residual[layer,row];total=pulse.sum()
                if peaks[row]<=cfg['activity_threshold_adc'] or total<=0:continue
                values.append([wf.PAD_U_MM[row],layer*12.,float(pulse.argmax()),float(pulse@np.arange(512)/total),total])
        values=np.asarray(values,float).reshape(-1,5)
    return values[np.argsort(values[:,4])[::-1]][:int(seq.get('maximum_third_anchors',24))] if len(values) else np.empty((0,5))


def region_conditioned_proposals(region,anchors,time_slopes,cal,template,seq):
    """Sample uncertain pair region, not one forced intersection point."""
    if region.get('status')=='outside_reaction_volume':return []
    v,center,cov=_region_samples(region)
    dimension=int(region.get('dimension',3 if cal is not None else 2));transverse=2 if dimension==3 else 1
    if not len(v) or center.shape!=(len(v),transverse):return []
    indices=np.unique(np.linspace(0,len(v)-1,min(len(v),int(seq.get('region_samples',5)))).astype(int))
    centroid=float(np.average(template[0],weights=template[1]));proposals=[]
    for index in indices:
        rv=float(v[index]);base=center[index]
        if not np.isfinite(base).all():continue
        width=np.sqrt(max(0.,float(cov[index,0,0]))) if cov.shape==(len(v),transverse,transverse) else 0.
        for offset in (-1.,0.,1.):
            ru=float(base[0]+offset*min(width,float(seq.get('maximum_region_offset_mm',8.))))
            for anchor in anchors:
                au,av,peak,mean,_=anchor;dv=av-rv
                if dv<float(seq.get('minimum_third_lever_mm',12.)):continue
                slope=(au-ru)/dv
                if abs(slope)>float(seq.get('maximum_u_slope',2.)):continue
                if dimension==3 and cal is not None:
                    scale=float(cal['sign'])*float(cal['velocity_mm_per_ns'])*float(cal['tb_ns'])
                    vertex_mean=float(cal['t0_tb'])+(base[1]-float(cal['reference_w_mm']))/scale
                    # Map the corrected vertex mean back into the observed raw
                    # timing coordinate; pad-delay differences must not become
                    # an unmodelled apparent physical angle.
                    vertex_raw_mean=vertex_mean+float(_delay_at_uv(ru,rv,cal))
                    dt=(mean-vertex_raw_mean)/dv;slopes=[dt]
                else:slopes=time_slopes or [0.]
                for dt in slopes:
                    p=np.array([ru+slope*(66.-rv),slope,peak-dt*(av-66.),dt])
                    if abs(dt)<=3. and 0<=p[2]<=511. and not any(np.allclose(p,old,atol=1e-6) for old in proposals):proposals.append(p)
    return proposals


def _delay_at_uv(u,v,cal):
    """Explicit native-pad timing offsets interpolated at a proposal coordinate.

    This interpolation is a proposal mapping only. The local-state measurement
    uses the original charge-weighted per-pad correction exactly.
    """
    uu,vv=np.broadcast_arrays(np.asarray(u,float),np.asarray(v,float))
    delay=np.asarray(cal.get('_delay',np.zeros((12,64))),float)
    values=[]
    for x,y in zip(uu.ravel(),vv.ravel()):
        by_layer=np.array([np.interp(x,wf.PAD_U_MM,row) for row in delay])
        values.append(np.interp(y,np.arange(12)*12.,by_layer))
    return np.asarray(values).reshape(uu.shape)


def _materialize(q,cfg,parameters,template,options):
    fit=wf._amplitude_fit(parameters,q,np.ones((12,64),bool),template,options,keep=True)
    forward=fit['forward_components'];bg=fit['forward_background'];prediction=forward.sum(0)+bg
    denominator=prediction+np.maximum(q-prediction,0.)
    fraction=np.divide(forward,denominator[None],out=np.zeros_like(forward),where=denominator[None]>0)
    components=(fraction*q[None]).astype(np.float32)
    # Normalized positive shares can sum above one by an IEEE-754 ulp.
    # Closure is checked at float32 precision; background remains nonnegative.
    background=(np.maximum(1.-fraction.sum(0),0.)*q).astype(np.float32)
    tb=cfg['tb_bin'];k=len(parameters);charge=q.reshape(12,64,512//tb,tb).sum(-1)
    assigned=np.concatenate((components,background[None])).reshape(k+1,12,64,512//tb,tb).sum(-1)
    fractions=np.divide(assigned,charge[None],out=np.zeros_like(assigned),where=charge[None]>0);fractions[-1,charge==0]=1.
    return dict(components=components,background=background,fractions=fractions,
        feature_lines=wf._parameters_to_lines(parameters,cfg['feature_time_scale']),
        forward_components=forward.astype(np.float32),forward_background=bg.astype(np.float32),residual_waveform=(q-prediction).astype(np.float32)),fit


def _history_row(stage,pair=-1,seed=-1,k=np.nan,support=np.nan,clean=np.nan,fraction=np.nan,peeled=np.nan,score=np.nan,held=np.nan,d2=np.nan,eligible=np.nan):
    return [STAGES[stage],pair,seed,k,support,clean,fraction,peeled,score,held,d2,eligible]


def _numeric_history(result,history,seeds,pairs,cfg,selected=None,residual=None,candidates=None,chosen=None):
    tb=cfg['tb_bin'];masks=np.zeros((2,12,64,512//tb),np.float32)
    if selected is not None:
        for i,seed in enumerate(selected[:2]):masks[i]=seed['confidence'].reshape(12,64,512//tb,tb).mean(-1)
    region_rows=[];envelopes=[]
    status_code={'resolved':1,'extended_region':2,'unresolved':3,'invalid_input':4,'outside_reaction_volume':5}
    for pairid,pair in enumerate(pairs):
        r=pair.get('region',{});interval=r.get('v_interval_mm');point=r.get('point')
        interval=np.asarray(interval if interval is not None else [np.nan,np.nan],float)
        if interval.shape!=(2,):interval=np.full(2,np.nan)
        pp=np.full(3,np.nan)
        if point is not None:
            a=np.asarray(point,float)
            if len(a)==2:pp[:2]=a
            elif len(a)==3:pp=a
        sides=r.get('open_v_sides',[False,False])
        region_rows.append([pairid,r.get('dimension',2),status_code.get(r.get('status'),0),*interval,*pp,int(sides[0]),int(sides[1])])
        e=r.get('transverse_region_envelope') or {}
        for v,c,cov in zip(e.get('v_samples_mm',[]),e.get('centers_mm',[]),e.get('covariances_mm2',[])):
            c=np.asarray(c,float);cov=np.asarray(cov,float)
            envelopes.append([pairid,v,c[0],c[1] if len(c)>1 else np.nan,cov[0,0],cov[0,1] if len(c)>1 else np.nan,cov[1,1] if len(c)>1 else np.nan])
    result.update(sequential_history=np.asarray(history,float).reshape(-1,len(HISTORY_COLUMNS)),
        sequential_seed_lines=wf._parameters_to_lines(np.array([s['parameters'] for s in seeds]).reshape(-1,4),cfg['feature_time_scale']),
        sequential_seed_rank=np.array([[i,s['rank_score'],s['supported'].sum(),s['clean'].sum(),s['charge_fraction'],int(s['eligible'])] for i,s in enumerate(seeds)],float).reshape(-1,6),
        sequential_confident_masks=masks,sequential_seed_pair_indices=np.array([p['seed_ids'] for p in pairs],np.int64).reshape(-1,2),
        sequential_pair_regions=np.asarray(region_rows,float).reshape(-1,10),sequential_region_envelope=np.asarray(envelopes,float).reshape(-1,7),
        sequential_soft_residual=np.asarray(residual if residual is not None else result['components'].sum(0)+result['background'],np.float32))
    candidate_lines=[];candidate_scores=[]
    for item in candidates or []:
        lines=np.full((3,6),np.nan);k=len(item['parameters'])
        lines[:k]=wf._parameters_to_lines(item['parameters'],cfg['feature_time_scale'])
        gate=item.get('region_gate') or {};d2=gate.get('min_d2')
        proposal={'single_track_competitor':1,'two_track_pair':2,'vertex_region':3,'free_residual':4}.get(item.get('proposal'),0)
        candidate_lines.append(lines)
        candidate_scores.append([item.get('pair_id',-1),k,item['selection_score'],item['heldout_chi2'],int(item['eligible']),
                                 d2 if d2 is not None else np.nan,proposal,int(item is chosen)])
    result.update(sequential_candidate_lines=np.asarray(candidate_lines,float).reshape(-1,3,6),
                  sequential_candidate_scores=np.asarray(candidate_scores,float).reshape(-1,8))
    result['diagnostics'].update(sequential_history_columns=list(HISTORY_COLUMNS),history_stage_codes=STAGES,
        sequential_region_columns=['pair','dimension','status','v_min','v_max','u','v','w','open_low','open_high'],
        sequential_envelope_columns=['pair','v','center_u','center_w','cov_uu','cov_uw','cov_ww'],
        sequential_candidate_columns=['pair','k','score','heldout_chi2','eligible','region_d2','proposal_code','selected'],
        sequential_proposal_codes={'single_track_competitor':1,'two_track_pair':2,'vertex_region':3,'free_residual':4},
        masks_are_train_pad_coarse_confidence=True,original_shared_waveform_never_erased=True,
        pair_region_added_as_independent_likelihood=False,physics_accuracy_evaluated=False)
    return result


def find_event(record,config,prior=None):
    """Observation-record API; final K is a tested hypothesis, never forced three.

    Explicit space_points proposal mode uses embedded measured points for seeds;
    final hypothesis fitting still uses the original full native-pad waveforms.
    """
    started=perf_counter()
    if not isinstance(record,Mapping) or 'positive_waveform' not in record:raise ValueError('Sequential finding requires an observation record with positive_waveform')
    cfg,seq=_configuration(config);q,cfg=wf._event_input(record['positive_waveform'],cfg)
    train_mask=_train_mask(cfg);training=q*train_mask[...,None];mode=seq.get('proposal_mode','direct_waveform')
    if mode not in ('direct_waveform','space_points'):raise ValueError('Unknown sequential proposal mode')
    history=[];seeds=[];pairs=[];cal,cal_status=_calibration(cfg,record);checkpoint_sha=None
    def fallback(reason):
        fallback_cfg=dict(cfg,method='waveform_hypotheses');fallback_cfg['waveform']=dict(cfg['waveform'],compare_k=True,k_candidates=seq['k_hypotheses'])
        fallback_cfg['waveform']['max_hypotheses']=max(int(fallback_cfg['waveform'].get('max_hypotheses',6)),len(seq['k_hypotheses']))
        if prior is not None:
            fallback_cfg.pop('checkpoint',None)
            fallback_cfg['waveform']['prior_is_heldout_blind']=True
        result=wf.find_event(q,fallback_cfg,prior)
        result['diagnostics'].update(sequential_status=reason,sequential_fallback=True,proposal_mode=mode,
            drift_status=cal_status,pair_vertex_physical_3d=False,checkpoint_sha256=checkpoint_sha,seconds=perf_counter()-started)
        if result['diagnostics'].get('found_count')==0:
            result['diagnostics'].update(selected_k=0,fixed_k=False,k_hypotheses=seq['k_hypotheses'])
        history.append(_history_row('fallback',eligible=0))
        return _numeric_history(result,history,seeds,pairs,cfg)
    if not np.any(training>cfg['activity_threshold_adc']):return fallback('no_training_pad_evidence')
    template=wf._pulse_template(training,cfg['waveform'].get('pulse_template'))
    if cfg.get('checkpoint') is not None:
        if prior is not None:raise ValueError('Supply only one checkpoint or prior')
        from .waveform_models import predict_checkpoint
        checkpoint=predict_checkpoint(cfg['checkpoint'],training,device=cfg.get('device','cpu'),expected_k=cfg['k'],expected_tb_bin=cfg['tb_bin'],trusted_checkpoint=bool(cfg.get('trusted_checkpoint',False)))
        prior=checkpoint['fractions'];checkpoint_sha=checkpoint.get('checkpoint_sha256')
    elif prior is not None and not cfg['waveform'].get('prior_is_heldout_blind',False):raise ValueError('Sequential prior must be heldout-pad blind')
    if prior is not None:prior=wf._check_prior(prior,cfg['k'],512//cfg['tb_bin'])
    if cfg['k']==1:return fallback('single_track_competing_waveform_route')
    first_pool=_pool(training,record,cfg,seq,template,mode,prior)
    seeds.extend(first_pool)
    for i,s in enumerate(first_pool):history.append(_history_row('proposal',seed=i,k=1,support=s['supported'].sum(),clean=s['clean'].sum(),fraction=s['charge_fraction'],eligible=int(s['eligible'])))
    first_ids=[i for i,s in enumerate(first_pool) if s['eligible']][:int(seq.get('maximum_first_seeds',2))]
    if not first_ids:return fallback('no_clean_multilayer_seed')
    max_pairs=int(seq.get('maximum_pair_hypotheses',3))
    for first_id in first_ids:
        first=seeds[first_id]
        first_remaining,removed=soft_peel(q,first['prediction'],first['confidence'],seq['peel_fraction'])
        history.append(_history_row('first_seed',seed=first_id,k=1,clean=first['clean'].sum(),fraction=first['charge_fraction'],eligible=1))
        history.append(_history_row('soft_peel',seed=first_id,k=1,peeled=removed.sum()/max(q.sum(),1e-12)))
        second_pool=_pool(first_remaining*train_mask[...,None],record,cfg,seq,template,mode,None)
        second_count=0
        for second in second_pool:
            second_id=len(seeds);seeds.append(second)
            if not second['eligible']:continue
            common=int(np.count_nonzero(first['supported']&second['supported']))
            if _same_fragment(first['parameters'],second['parameters'],seq) or common<int(seq.get('minimum_pair_common_layers',3)):
                history.append(_history_row('fragment_rejected',seed=second_id,k=2,support=common,eligible=0));continue
            parameters=np.stack((first['parameters'],second['parameters']))
            if any(all(any(_same_fragment(p,old,seq) for old in pair['fit']['parameters']) for p in parameters) for pair in pairs):continue
            history.append(_history_row('second_seed',seed=second_id,k=2,support=common,clean=second['clean'].sum(),fraction=second['charge_fraction'],eligible=1))
            joint=wf._fit_hypothesis(q,train_mask,parameters,template,cfg['waveform'])
            try:region,local=_pair_region(joint['parameters'],training,template,joint['pad_sigma_mm'],cfg,seq,cal,record)
            except (ValueError,KeyError,ImportError,np.linalg.LinAlgError) as error:
                region={'status':'unresolved','dimension':3 if cal is not None else 2,'message':str(error)};local=[]
            residual,removed_second=soft_peel(first_remaining,second['prediction'],second['confidence'],seq['peel_fraction'])
            pair=dict(seed_ids=[first_id,second_id],seeds=[first,second],fit=joint,region=region,local=local,residual=residual)
            pairs.append(pair);pid=len(pairs)-1
            second_count+=1
            history.append(_history_row('pair_fit',pair=pid,k=2,score=joint['selection_score'],held=joint['heldout_chi2'],eligible=1))
            history.append(_history_row('pair_region',pair=pid,k=2,eligible=int(region.get('status')=='resolved')))
            history.append(_history_row('soft_peel',pair=pid,seed=second_id,k=2,peeled=removed_second.sum()/max(q.sum(),1e-12)))
            if len(pairs)>=max_pairs:break
            if second_count>=int(seq.get('maximum_seconds_per_first',2)):break
        if len(pairs)>=max_pairs:break
    if not pairs:return fallback('no_independent_second_seed_or_same_track_fragments')
    # Keep more than one pair hypothesis; do not commit the first peel as truth.
    candidates=[]
    if 1 in seq['k_hypotheses']:
        for first_id in first_ids:
            joint=wf._fit_hypothesis(q,train_mask,seeds[first_id]['parameters'][None],template,cfg['waveform'])
            candidates.append(dict(joint,pair_id=-1,seed_id=first_id,region_gate=None,proposal='single_track_competitor',eligible=True))
    for pid,pair in enumerate(pairs):
        if 2 in seq['k_hypotheses']:candidates.append(dict(pair['fit'],pair_id=pid,region_gate=None,proposal='two_track_pair',eligible=True))
        if 3 not in seq['k_hypotheses']:continue
        residual=pair['residual']*train_mask[...,None]
        pool=_pool(residual,record,cfg,seq,template,mode)
        free=[s['parameters'] for s in pool if not any(_same_fragment(s['parameters'],p,seq) for p in pair['fit']['parameters'])]
        slopes=[float(p[3]) for p in free[:3]] or [0.]
        anchors=_anchors(residual,record,cfg,seq,mode)
        conditioned=region_conditioned_proposals(pair['region'],anchors,slopes,cal,template,seq)
        proposal_scores=[]
        points,charge=_proposal_points(residual,record,cfg,mode)
        geometry=geo._config(cfg.get('geometry',{'scales_mm':[2.,6.,4.]}))
        _,metric,weights,scales=geo._input(points,charge,geometry)
        for origin,p in [('vertex_region',x) for x in conditioned]+[('free_residual',x) for x in free]:
            if any(_same_fragment(p,old,seq) for old in pair['fit']['parameters']):continue
            gate=region_compatibility(p,pair['region'],cal,template,seq)
            line=wf._parameters_to_lines(p[None],cfg['feature_time_scale'])[0]
            direction=geo._canonical(line[3:]/scales)
            score=float(weights@np.exp(-.5*(geo._distance(metric,line[:3]/scales,direction)/geometry.assignment_width)**2)) if len(points) else 0.
            if gate['eligible'] is False:score=-np.inf
            proposal_scores.append((score,origin,p,gate))
        proposal_scores.sort(key=lambda x:x[0],reverse=True)
        kept=[]
        for score,origin,p,gate in proposal_scores:
            if not np.isfinite(score):continue
            if any(_same_fragment(p,old,seq) for old in kept):continue
            kept.append(p)
            history.append(_history_row('third_proposal',pair=pid,k=3,score=score,d2=gate['min_d2'] if gate['min_d2'] is not None else np.nan,eligible=1))
            initial=np.vstack((pair['fit']['parameters'],p))
            joint=wf._fit_hypothesis(q,train_mask,initial,template,cfg['waveform'])
            checked=region_compatibility(joint['parameters'][2],pair['region'],cal,template,seq)
            eligible=checked['eligible'] is not False
            candidates.append(dict(joint,pair_id=pid,region_gate=checked,proposal=origin,eligible=eligible))
            history.append(_history_row('joint_test',pair=pid,k=3,score=joint['selection_score'],held=joint['heldout_chi2'],d2=checked['min_d2'] if checked['min_d2'] is not None else np.nan,eligible=int(eligible)))
            if len(kept)>=int(seq.get('maximum_third_hypotheses_per_pair',3)):break
    eligible=[c for c in candidates if c['eligible']]
    if not eligible:return fallback('no_waveform_and_vertex_consistent_hypothesis')
    one=[c for c in eligible if len(c['parameters'])==1]
    two=[c for c in eligible if len(c['parameters'])==2];three=[c for c in eligible if len(c['parameters'])==3]
    best1=min(one,key=lambda c:c['selection_score']) if one else None
    best2=min(two,key=lambda c:c['selection_score']) if two else None
    best3=min(three,key=lambda c:c['selection_score']) if three else None
    lower=[candidate for candidate in (best1,best2) if candidate is not None]
    best_lower=min(lower,key=lambda c:c['selection_score']) if lower else None
    gain=(best_lower['selection_score']-best3['selection_score']) if best_lower is not None and best3 is not None else None
    if best3 is not None and (best_lower is None or gain>float(seq.get('minimum_third_evidence_gain',10.))):chosen=best3
    elif best_lower is not None:chosen=best_lower
    else:return fallback('third_hypotheses_lack_predictive_support')
    pair=pairs[chosen['pair_id']] if chosen['pair_id']>=0 else None
    options=dict(cfg['waveform'],pad_sigma_mm=chosen['pad_sigma_mm'])
    result,fit=_materialize(q,cfg,chosen['parameters'],template,options)
    k=len(chosen['parameters']);relative=float(np.abs(result['residual_waveform']).sum()/max(q.sum(),1e-12))
    active=q.sum((1,2))>0;ill=bool(np.any(fit['template_gram_condition'][active]>float(seq.get('maximum_gram_condition',1e8))))
    uncertain=pair is not None and (pair['region'].get('status')!='resolved' or any(pair['region'].get('open_v_sides',[True,True])))
    small=bool(np.any(result['components'].sum((1,2,3))<q.sum()*float(seq.get('minimum_component_fraction',.005))))
    ambiguous=bool(ill or uncertain or small or relative>float(seq.get('maximum_relative_waveform_l1',.25)) or not chosen['optimizer_success'])
    history.append(_history_row('selected',pair=chosen['pair_id'],k=k,score=chosen['selection_score'],held=chosen['heldout_chi2'],eligible=1))
    evidence=[]
    for c in candidates:
        evidence.append({key:(value.tolist() if isinstance(value,np.ndarray) else value) for key,value in c.items()})
    result['diagnostics']=dict(status='ambiguous' if ambiguous else 'sequential_observation_hypothesis_supported_NOT_physics_validated',
        method='sequential_waveform_vertex_region',sequential_status='completed',sequential_fallback=False,ambiguous=ambiguous,
        selected_k=k,requested_maximum_k=cfg['k'],fixed_k=False,k_hypotheses=seq['k_hypotheses'],
        found_count=k,calibrated_physical_w=False,
        third_evidence_gain=gain,third_track_supported=k==3,hypotheses=evidence,pair_hypothesis_count=len(pairs),
        pair_regions=[p['region'] for p in pairs],chosen_pair_id=chosen['pair_id'],third_proposal_source=chosen['proposal'],
        third_region_gate=chosen['region_gate'],pair_vertex_physical_3d=cal is not None and pair is not None,drift_status=cal_status,
        proposal_mode=mode,full_waveform_relative_l1=relative,response_template_source=template[2],checkpoint_sha256=checkpoint_sha,
        uncalibrated_feature_time_is_not_physical_w=True,local_tangent_scope='first observed layers, not known production direction',
        uncertainty_scope='response/conditioning and proxy covariance; no calibrated separation or vertex probability',
        heldout_note='Candidate geometry, pulse template, local seed states and pair region exclude whole heldout pads; predictive scores use their original ADC',
        final_fit_uses_original_waveform=True,seconds=perf_counter()-started,
        max_charge_conservation_error_adc=float(np.max(abs(result['components'].sum(0)+result['background']-q))),
        finding_coordinate='feature lines (u mm,v mm,TB*feature_time_scale), NEVER silently physical 3D')
    selected=pair['seeds'] if pair is not None else [seeds[chosen['seed_id']]]
    residual=pair['residual'] if pair is not None else q
    return _numeric_history(result,history,seeds,pairs,cfg,selected,residual,candidates,chosen)
