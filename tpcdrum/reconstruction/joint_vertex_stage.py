"""ROOT-record bridge for region seeds and common-vertex joint fitting.

Original measured clouds are consumed once. An internally derived vertex is
never inserted as an independent measurement of those same tracks. Projected
UV results cannot become calibrated physical UVW results downstream.
"""
import copy
import numpy as np

from .vertex_measurements import observed_tracks,local_state


def _empty(k):
    n=lambda shape:np.full(shape,np.nan)
    return dict(vertex_uvw_mm=n(3),vertex_valid=np.int64(0),vertex_dimension=np.int64(0),
        vertex_max_dca_mm=np.float64(np.nan),vertex_condition_number=np.float64(np.nan),
        vertex_candidate_uvw_mm=n(3),vertex_covariance_uvw_mm2=n((3,3)),
        vertex_projected_uv_mm=n(2),vertex_projected_valid=np.int64(0),
        vertex_free_projected_uv_mm=n(2),vertex_free_projected_valid=np.int64(0),
        vertex_v_interval_mm=n(2),vertex_region_status_code=np.int64(-1),
        vertex_used_track_indices=np.empty(0,dtype=np.int64),vertex_seed_origins=np.empty((0,3)),
        vertex_seed_directions=np.empty((0,3)),vertex_seed_covariances=np.empty((0,4,4)),
        vertex_seed_reference_v_mm=np.empty(0),vertex_joint_covariance=np.empty((0,0)),
        vertex_joint_projected_paths_uv_mm=n((k,12,2)),vertex_joint_track_direction_uvw=n((k,3)),
        vertex_joint_paths_uvw_mm=n((k,12,3)),vertex_free_paths_uvw_mm=n((k,12,3)),
        vertex_free_theta_uv_deg=n(k),vertex_joint_theta_uv_deg=n(k),
        vertex_joint_comparison_valid=np.zeros(k,dtype=np.uint8),
        vertex_joint_applied=np.int64(0),vertex_joint_physical_3d=np.int64(0),
        vertex_joint_constraint_is_independent_measurement=np.int64(0),
        track_vertex_directions_uvw=n((k,3)),vertex_direction_valid=np.zeros(k,dtype=np.uint8),
        vertex_direction_joint_covariance=n((3*k,3*k)))


def vertex_event(record,diagnostic,config):
    from .algorithms.joint_vertex import vertex_region,joint_fit
    k=len(record['components']);values=_empty(k)
    cfg=copy.deepcopy(config.get('vertex',{}).get('joint',{}))
    requested=cfg.pop('dimension','auto')
    dimension=(3 if diagnostic.get('fitting',{}).get('physical_3d') is True else 2) if requested=='auto' else int(requested)
    cfg['dimension']=dimension
    cfg['physical_3d']=dimension==3
    result=dict(status='missing_measurements',physical_3d=False,dimension=dimension,
        observation_source='original per-layer measured clouds, never previous fitted points',
        covariance_scope='conditional on supplied cloud/sampling proxy; detector precision not certified',
        vertex_as_independent_measurement=False,physics_accuracy_evaluated=False)
    # Fixed fields regardless of per-event success; preserve free track estimates.
    fitkeys=('fit_states','fit_covariances','fit_points_uvw_mm','fit_valid_layer','fit_ok',
             'track_origins_uvw_mm','track_directions_uvw','track_direction_covariances',
             'fit_reference_v_mm','fit_state_dimension','fit_energy_first_point_mev')
    for key in fitkeys:
        if key in record:
            values[key]=np.asarray(record[key]).copy()
            values['pre_vertex_'+key]=np.asarray(record.get('pre_vertex_'+key,record[key])).copy()
    from .refit_stage import _downstream_defaults
    invalidated=_downstream_defaults(record,k)
    # Keep branch sets stable on success/failure while invalidating downstream
    # only if physical trajectories are actually replaced.
    protected=set(_empty(k))
    for key,default in invalidated.items():
        if key in protected:continue
        values[key]=np.asarray(record.get(key,default)).copy()
        values['pre_vertex_'+key]=np.asarray(record.get('pre_vertex_'+key,record.get(key,default))).copy()
    baseline={key:np.asarray(value).copy() for key,value in values.items()}
    merged=dict(diagnostic)
    try:
        tracks,covs,layers,slots=observed_tracks(record,config,dimension)
        values['vertex_dimension']=np.int64(dimension)
        if len(tracks)<2:
            result.update(status='fewer_than_two_measured_tracks',observed_tracks=len(tracks))
            return values,dict(merged,vertex=result)
        local=[];usable=[]
        for i,(points,covariance) in enumerate(zip(tracks,covs)):
            try:local.append(local_state(points,covariance,int(cfg.get('seed_layers',4))));usable.append(i)
            except ValueError:continue
        if len(local)<2:
            result['status']='fewer_than_two_local_directions'
            return values,dict(merged,vertex=result)
        tracks=[tracks[i] for i in usable];covs=[covs[i] for i in usable];layers=[layers[i] for i in usable]
        slots=slots[usable]
        origins=np.stack([s['origin'] for s in local]);directions=np.stack([s['direction'] for s in local])
        statecov=np.stack([s['covariance'] for s in local]);reference=np.array([s['reference_v'] for s in local])
        region=vertex_region(origins,directions,statecov,reference,cfg,dimension=dimension)
        codes={'resolved':1,'extended_region':2,'unresolved':3,'invalid_input':4}
        values['vertex_region_status_code']=np.int64(codes.get(region.get('status'),0))
        interval=region.get('v_interval_mm')
        if interval is not None:values['vertex_v_interval_mm']=np.asarray(interval,float)
        embedded_origins=np.full((len(slots),3),np.nan);embedded_dirs=embedded_origins.copy()
        embedded_cov=np.full((len(slots),4,4),np.nan)
        embedded_origins[:,:dimension]=origins;embedded_dirs[:,:dimension]=directions
        if dimension==3:embedded_cov[:]=statecov
        else:embedded_cov[:,0,0]=statecov[:,0,0];embedded_cov[:,0,2]=statecov[:,0,1];embedded_cov[:,2,0]=statecov[:,1,0];embedded_cov[:,2,2]=statecov[:,1,1]
        values.update(vertex_used_track_indices=slots,vertex_seed_origins=embedded_origins,
            vertex_seed_directions=embedded_dirs,vertex_seed_covariances=embedded_cov,vertex_seed_reference_v_mm=reference)
        point=region.get('point')
        if point is not None:
            values['vertex_candidate_uvw_mm'][:dimension]=np.asarray(point,float)
            values['vertex_free_projected_uv_mm']=np.asarray(point,float)[:2]
            values['vertex_free_projected_valid']=np.int64(region.get('status')=='resolved')
        fitted=joint_fit(tracks,covs,cfg)
        result.update(status=fitted.get('status','unknown'),physical_3d=dimension==3,
            seed_region=region,joint_fit=fitted,track_slots=slots.tolist(),
            same_measurement_process_model=fitted.get('same_measurement_process_model',False))
        # The performance baseline uses the SAME free-fit model/data as the
        # joint fit, rather than the shorter first-layer proposal seed above.
        free_region=fitted.get('region',{})
        values['vertex_free_projected_uv_mm'][:]=np.nan
        values['vertex_free_projected_valid']=np.int64(0)
        if free_region.get('status')=='resolved' and free_region.get('point') is not None:
            values['vertex_free_projected_uv_mm']=np.asarray(free_region['point'],float)[:2]
            values['vertex_free_projected_valid']=np.int64(1)
        # Free fits remain in the comparison even when the common-origin
        # hypothesis fails. Otherwise before/after would be success-selected.
        for i,slot in enumerate(slots):
            if len(fitted.get('vertex_free_theta_uv_deg',[]))==len(slots):
                values['vertex_free_theta_uv_deg'][slot]=fitted['vertex_free_theta_uv_deg'][i]
            if len(fitted.get('free_fitted_points',[]))==len(slots):
                values['vertex_free_paths_uvw_mm'][slot,layers[i],:dimension]=np.asarray(fitted['free_fitted_points'][i])
            if len(fitted.get('vertex_joint_theta_uv_deg',[]))==len(slots):
                values['vertex_joint_theta_uv_deg'][slot]=fitted['vertex_joint_theta_uv_deg'][i]
        candidate=fitted.get('candidate_vertex')
        if candidate is not None and np.asarray(candidate).shape==(dimension,):
            values['vertex_candidate_uvw_mm'][:dimension]=np.asarray(candidate,float)
        if not fitted.get('success',False):return values,dict(merged,vertex=result)
        vertex=np.asarray(fitted['vertex'],float)
        if vertex.shape!=(dimension,) or not np.isfinite(vertex).all():raise ValueError('Invalid successful joint vertex')
        fullcov=np.asarray(fitted.get('joint_covariance',np.empty((0,0))),float)
        fullcov[~np.isfinite(fullcov)]=np.nan
        values['vertex_joint_covariance']=fullcov
        for i,slot in enumerate(slots):
            values['vertex_free_theta_uv_deg'][slot]=fitted['vertex_free_theta_uv_deg'][i]
            values['vertex_joint_theta_uv_deg'][slot]=fitted['vertex_joint_theta_uv_deg'][i]
            values['vertex_joint_comparison_valid'][slot]=np.uint8(np.isfinite([values['vertex_free_theta_uv_deg'][slot],values['vertex_joint_theta_uv_deg'][slot]]).all())
            path=np.asarray(fitted['fitted_points'][i],float);direction=np.asarray(fitted['local_directions'][i],float)
            if path.shape!=tracks[i].shape or direction.shape!=path.shape:raise ValueError('Joint path/layer shape mismatch')
            values['vertex_joint_projected_paths_uv_mm'][slot,layers[i]]=path[:,:2]
            values['vertex_joint_paths_uvw_mm'][slot,layers[i],:dimension]=path
            values['vertex_free_paths_uvw_mm'][slot,layers[i],:dimension]=np.asarray(fitted['free_fitted_points'][i])
            values['vertex_joint_track_direction_uvw'][slot,:dimension]=np.asarray(fitted['joint_track_directions'][i])
        values['vertex_projected_uv_mm']=vertex[:2].copy();values['vertex_projected_valid']=np.int64(1)
        values['vertex_joint_applied']=np.int64(1)
        v_cov=np.asarray(fitted.get('vertex_covariance'),float)
        if v_cov.shape==(dimension,dimension):values['vertex_covariance_uvw_mm2'][:dimension,:dimension]=v_cov
        if dimension==3:
            for key,default in invalidated.items():
                if key not in protected:values[key]=np.asarray(default).copy()
            values.update(vertex_uvw_mm=vertex,vertex_candidate_uvw_mm=vertex.copy(),
                vertex_valid=np.int64(1),vertex_joint_physical_3d=np.int64(1))
            for i,slot in enumerate(slots):
                ids=layers[i];path=np.asarray(fitted['fitted_points'][i]);direction=np.asarray(fitted['local_directions'][i])
                values['fit_points_uvw_mm'][slot]=np.nan;values['fit_states'][slot]=np.nan;values['fit_covariances'][slot]=np.nan
                values['fit_valid_layer'][slot]=0;values['fit_valid_layer'][slot,ids]=1
                values['fit_points_uvw_mm'][slot,ids]=path
                values['fit_states'][slot,ids,:4]=np.c_[path[:,0],path[:,2],direction[:,0]/direction[:,1],direction[:,2]/direction[:,1]]
                values['fit_ok'][slot]=1;values['fit_state_dimension'][slot]=4
                values['track_origins_uvw_mm'][slot]=path[0];values['track_directions_uvw'][slot]=direction[0]
                values['track_direction_covariances'][slot]=np.nan
                values['fit_reference_v_mm'][slot]=path[0,1];values['fit_energy_first_point_mev'][slot]=np.nan
            if fitted.get('vertex_track_directions') is not None:
                d=np.asarray(fitted['vertex_track_directions'],float)
                if d.shape!=(len(slots),3) or not np.isfinite(d).all():raise ValueError('Invalid joint vertex-direction model')
                values['track_vertex_directions_uvw'][slots]=d;values['vertex_direction_valid'][slots]=1
                dc=np.asarray(fitted.get('vertex_track_direction_joint_covariance'),float)
                if dc.shape==(3*len(slots),3*len(slots)):
                    indices=np.array([3*slot+j for slot in slots for j in range(3)])
                    values['vertex_direction_joint_covariance'][np.ix_(indices,indices)]=dc
                merged['vertex_direction']=dict(source='joint common-vertex conditional trajectory model',observation_only=True,
                    unobserved_deflections_recovered_exactly=False,model_scope=fitted.get('vertex_direction_scope'))
            merged['pre_vertex_fitting']=copy.deepcopy(diagnostic.get('fitting',{}))
            merged['fitting']=dict(status='joint_vertex_conditional_refit',physical_3d=True,
                covariance_model='global joint covariance stored separately; stale local covariances cleared',
                physics_accuracy_evaluated=False)
        return values,dict(merged,vertex=result)
    except (ValueError,KeyError,TypeError,IndexError,np.linalg.LinAlgError) as error:
        result.update(status='missing_or_invalid_joint_inputs',message=str(error))
        return baseline,dict(diagnostic,vertex=result)
