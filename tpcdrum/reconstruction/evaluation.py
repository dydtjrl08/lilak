"""Explicit paired-ROOT evaluation, isolated from observation-only inference.

One globally selected alpha permutation per EVENT is reused for charge and
all direction metrics; it is never optimized independently per layer/metric.
Background is fixed. Evaluation files cannot be downstream observation inputs.
"""
from pathlib import Path
import json
import numpy as np

from .stage_io import StageWriter,read_stage,manifest,sha256,jsonable
from .training_finding import separation_metrics


def _identity(row):
    if not {'file_index','entry','source_group'}<=set(row):
        raise ValueError('Paired ROOT rows require file_index, entry and source_group')
    key=tuple(int(row[n]) for n in ('file_index','entry'))
    if min(key)<0:raise ValueError('Negative event identity')
    return key


def _reference_index(path):
    """Scalar identities only: never retain full truth waveforms for a corpus."""
    import uproot
    index={}
    with uproot.open(path) as source:
        tree=source['events']
        fields=tree.arrays(['file_index','entry','source_group'],library='np')
        for position,(fi,entry,group) in enumerate(zip(fields['file_index'],fields['entry'],fields['source_group'])):
            key=int(fi),int(entry)
            if min(key)<0 or key in index:raise ValueError('Negative/duplicate reference event identity')
            index[key]=(position,int(group))
    return index


def _fractions(row,reference,tb_bin):
    q=np.asarray(row['positive_waveform'],float)
    pred=np.asarray(row['components'],float)
    bg=np.asarray(row.get('background_waveform',row.get('background')),float)
    target=np.asarray(reference['truth_components'],float)
    target_bg=np.asarray(reference['truth_background'],float)
    if q.shape!=(12,64,512) or pred.ndim!=4 or target.ndim!=4 or pred.shape[1:]!=q.shape or target.shape[1:]!=q.shape:
        raise ValueError('Invalid observed/reference component waveform shapes')
    if bg.shape!=q.shape or target_bg.shape!=q.shape:raise ValueError('Missing waveform background')
    for a in (q,pred,bg,target,target_bg):
        if not np.isfinite(a).all() or np.any(a<0):raise ValueError('Nonfinite/negative waveform attribution')
    scale=max(1.,float(q.max()))
    if not np.allclose(pred.sum(0)+bg,q,atol=2e-6*scale,rtol=2e-6):
        raise ValueError('Predicted components do not conserve measured positive ADC')
    if not np.allclose(target.sum(0)+target_bg,q,atol=2e-6*scale,rtol=2e-6):
        raise ValueError('Reference attribution differs from the same observed positive ADC')
    kp,kt=len(pred),len(target);k=max(kp,kt)
    if k>6 or min(kp,kt)<1:raise ValueError('Exact eventwise assignment supports positive K<=6')
    if 512%tb_bin:raise ValueError('Evaluation tb_bin must divide 512')
    def coarse(components,background):
        padded=np.zeros((k+1,12,64,512));padded[:len(components)]=components;padded[-1]=background
        return padded.reshape(k+1,12,64,512//tb_bin,tb_bin).sum(-1)
    charge=q.reshape(12,64,512//tb_bin,tb_bin).sum(-1)
    p,t=coarse(pred,bg),coarse(target,target_bg)
    for values in (p,t):
        np.divide(values,charge[None],out=values,where=charge[None]>0)
        values[-1,charge==0]=1.
    m=separation_metrics(p,t,charge)
    # Do not dilute overlap MAE with dummy alpha slots when multiplicities differ.
    overlap=(np.sum(t[:kt]>.1,axis=0)>=2)&(charge>40.)
    aligned=m.pop('aligned_fractions')
    m['overlap_fraction_mae']=float(np.abs(aligned[:kt,overlap]-t[:kt,overlap]).mean()) if overlap.any() else float('nan')
    m['predicted_k']=kp;m['reference_k']=kt
    m['matched_prediction_index']=np.asarray([i if i<kp else -1 for i in m['slot_permutation'][:kt]],dtype=np.int64)
    m['positive_charge']=float(q.sum())
    m['has_signal']=bool(q.sum()>0)
    return m


def _directions(row,diagnostic,reference,matching):
    target=np.asarray(reference.get('truth_direction_uvw',np.empty((0,3))),float)
    kt=len(matching)
    angular=np.full(kt,np.nan);theta=np.full(kt,np.nan);valid=np.zeros(kt,dtype=np.uint8)
    direction=row.get('track_directions_uvw')
    physical=diagnostic.get('fitting',{}).get('physical_3d') is True
    if 'physical_3d' in row:physical=physical or bool(np.asarray(row['physical_3d']).item())
    if target.shape!=(kt,3):return angular,theta,valid,'missing_reference_initial_direction'
    if direction is None or not physical:return angular,theta,valid,'missing_calibrated_physical_fit'
    direction=np.asarray(direction,float);ok=np.asarray(row.get('fit_ok',np.ones(len(direction))),bool)
    if direction.ndim!=2 or direction.shape[1]!=3 or ok.shape!=(len(direction),):raise ValueError('Invalid fitted direction arrays')
    for j,i in enumerate(matching):
        if i<0 or i>=len(direction) or not ok[i]:continue
        a,b=direction[i],target[j]
        if not np.isfinite(a).all() or not np.isfinite(b).all():continue
        na,nb=np.linalg.norm(a),np.linalg.norm(b)
        if na<=0 or nb<=0:continue
        angular[j]=np.degrees(np.arccos(np.clip(np.dot(a,b)/(na*nb),-1.,1.)))
        theta[j]=(np.degrees(np.arctan2(b[0],b[1])-np.arctan2(a[0],a[1]))+180.)%360.-180.
        valid[j]=1
    return angular,theta,valid,'ok' if np.all(valid) else 'partial_or_invalid_physical_fit'


def _projected_ols(row,reference,matching):
    """u-v projection needs no drift conversion; it is not a physical 3D fit."""
    residual=np.full(len(matching),np.nan)
    values=row.get('fit_theta_pad_2d_ols_deg');target=reference.get('truth_direction_uvw')
    if values is None or target is None:return residual
    values=np.asarray(values,float);target=np.asarray(target,float)
    if values.ndim!=1 or target.shape!=(len(matching),3):raise ValueError('Invalid projected direction arrays')
    for j,i in enumerate(matching):
        if i<0 or i>=len(values) or not np.isfinite(values[i]) or not np.isfinite(target[j,:2]).all():continue
        if np.linalg.norm(target[j,:2])<=0:continue
        residual[j]=(np.degrees(np.arctan2(target[j,0],target[j,1]))-values[i]+180.)%360.-180.
    return residual



def _decode_mass_json(value):
    array=np.asarray(value)
    if array.ndim!=1 or array.dtype.kind not in 'iu' or np.any(array<0) or np.any(array>255):
        raise ValueError('Invalid embedded mass-provenance UTF8 array')
    return json.loads(array.astype(np.uint8).tobytes().decode('utf8'))


def _mass_reference(row,reference,cfg,truth_meta):
    supplied=cfg.get('masses',truth_meta.get('masses',{}))
    alpha=supplied.get('alpha_mev');ground=supplied.get('carbon12_ground_mev')
    source='explicit_evaluation_masses' if alpha is not None or ground is not None else None
    if alpha is None and ground is None:
        embedded=np.asarray(row.get('kinematics_mass_provenance_json_utf8',[]))
        if embedded.size:
            provenance=_decode_mass_json(embedded)
            if not provenance.get('source'):raise ValueError('Embedded masses need their original provenance')
            alpha=row.get('mass_alpha_mev');ground=row.get('mass_carbon12_ground_mev');source='embedded_kinematics_masses'
        elif np.asarray(row.get('stopping_metadata_json_utf8',[])).size:
            provenance=_decode_mass_json(row['stopping_metadata_json_utf8'])
            if provenance.get('schema')!='tpcdrum-geant4-material-export-v1' or not provenance.get('calibration_sha256'):
                raise ValueError('Embedded stopping masses lack initialized-Geant4 schema/hash')
            m=provenance.get('masses_mev',{});alpha=m.get('alpha');ground=m.get('carbon12_ground');source='embedded_initialized_Geant4_masses'
    def value(x):
        if x is None:return None
        x=float(x)
        if not np.isfinite(x) or x<=0:raise ValueError('Mass reference must be positive finite MeV')
        return x
    return value(alpha),value(ground),source or 'missing_explicit_or_embedded_masses'


def _independent_threebody(vectors,ground_mass=None):
    """Independent direct four-vector arithmetic; imports no inference physics.

    Matrix-free Lorentz transform to the sum-rest frame. No 120-degree,
    equal-energy, mass-window or conservation adjustment is imposed.
    """
    a=np.asarray(vectors,float)
    if a.shape!=(3,4) or not np.isfinite(a).all() or np.any(a[:,3]<=0):raise ValueError('Need three finite positive-energy initial four-vectors')
    daughter_m2=a[:,3]**2-np.einsum('ij,ij->i',a[:,:3],a[:,:3])
    total=a.sum(0);m2=total[3]**2-float(total[:3]@total[:3])
    if np.any(daughter_m2<=0) or m2<=0:raise ValueError('Reference/prediction four-vectors must be timelike')
    mass=np.sqrt(m2);beta=total[:3]/total[3];b2=float(beta@beta)
    rest=a.copy()
    if b2>0:
        gamma=total[3]/mass;dot=a[:,:3]@beta
        rest[:,:3]=a[:,:3]+(((gamma-1.)*dot/b2-gamma*a[:,3])[:,None]*beta)
        rest[:,3]=gamma*(a[:,3]-dot)
    norms=np.linalg.norm(rest[:,:3],axis=1);angles=np.full(3,np.nan)
    for i,(j,k) in enumerate(((0,1),(0,2),(1,2))):
        if norms[j]>1e-12 and norms[k]>1e-12:
            angles[i]=np.degrees(np.arccos(np.clip(float(rest[j,:3]@rest[k,:3])/(norms[j]*norms[k]),-1.,1.)))
    kinetic=rest[:,3]-np.sqrt(daughter_m2)
    if np.any(kinetic < -1e-7):raise ValueError('Negative COM kinetic energy beyond roundoff')
    kinetic=np.maximum(kinetic,0.);fraction=kinetic/kinetic.sum() if kinetic.sum()>1e-12 else np.full(3,np.nan)
    dalitz=np.array([(fraction[1]-fraction[0])/np.sqrt(3.),fraction[2]-1/3.])
    return dict(mass=float(mass),excitation=float(mass-ground_mass) if ground_mass is not None else np.nan,
        angles=angles,dalitz=dalitz,fraction=fraction,com_vectors=rest,
        momentum_closure=float(np.linalg.norm(rest[:,:3].sum(0))))


def _physics_errors(row,diagnostic,reference,matching,cfg,truth_meta):
    kt=len(matching);nan=lambda n:np.full(n,np.nan)
    values=dict(vertex_error_uvw_mm=nan(3),vertex_error_norm_mm=np.float64(np.nan),vertex_error_valid=np.int64(0),
        reference_vertex_uvw_mm=nan(3),reference_initial_kinetic_mev=nan(kt),
        reference_alpha_mass_mev=np.float64(np.nan),reference_carbon_ground_mass_mev=np.float64(np.nan),
        vertex_candidate_error_uvw_mm=nan(3),vertex_candidate_error_norm_mm=np.float64(np.nan),vertex_candidate_error_valid=np.int64(0),
        kinetic_energy_error_mev=nan(kt),kinetic_energy_valid=np.zeros(kt,dtype=np.uint8),
        kinetic_energy_candidate_error_mev=nan(kt),kinetic_energy_candidate_valid=np.zeros(kt,dtype=np.uint8),
        reference_invariant_mass_mev=np.float64(np.nan),reference_excitation_mev=np.float64(np.nan),
        reference_com_pair_angles_deg=nan(3),reference_dalitz_xy=nan(2),reference_four_vectors_mev=nan((3,4)),
        reference_kinematics_valid=np.int64(0),kinematics_accepted_error_valid=np.int64(0),kinematics_candidate_error_valid=np.int64(0),
        invariant_mass_error_mev=np.float64(np.nan),excitation_error_mev=np.float64(np.nan),
        com_pair_angle_error_deg=nan(3),dalitz_error_xy=nan(2),
        candidate_invariant_mass_error_mev=np.float64(np.nan),candidate_excitation_error_mev=np.float64(np.nan),
        candidate_com_pair_angle_error_deg=nan(3),candidate_dalitz_error_xy=nan(2),
        invariant_mass_stage_consistency_mev=np.float64(np.nan),excitation_stage_consistency_mev=np.float64(np.nan),
        com_pair_angle_stage_consistency_deg=nan(3),dalitz_stage_consistency_xy=nan(2))
    notes=dict(residual_sign='reconstructed minus initial MC reference',pair_order=[[0,1],[0,2],[1,2]],
        matching_reused_for_energy_and_correlations=True,vertex_status='missing_reference_or_reconstruction',
        energy_status='missing_vertex_energy_or_initial_MC_kinetic',kinematics_status='missing_three_body_reference',
        accepted_and_candidate_metrics_separate=True,no_120_degree_constraint=True)
    target_vertex=np.asarray(reference.get('truth_vertex_uvw_mm',[]),float)
    if target_vertex.shape==(3,) and np.isfinite(target_vertex).all():
        values['reference_vertex_uvw_mm']=target_vertex.copy()
        reconstructed=np.asarray(row.get('vertex_uvw_mm',[]),float)
        candidate=row.get('vertex_candidate_uvw_mm',diagnostic.get('vertex',{}).get('candidate_vertex_mm'))
        if candidate is not None:
            candidate=np.asarray(candidate,float)
            if candidate.shape==(3,) and np.isfinite(candidate).all():
                diff=candidate-target_vertex
                values.update(vertex_candidate_error_uvw_mm=diff,vertex_candidate_error_norm_mm=np.float64(np.linalg.norm(diff)),vertex_candidate_error_valid=np.int64(1))
        if bool(row.get('vertex_valid',False)) and reconstructed.shape==(3,) and np.isfinite(reconstructed).all() and int(row.get('vertex_dimension',3))==3:
            diff=reconstructed-target_vertex
            values.update(vertex_error_uvw_mm=diff,vertex_error_norm_mm=np.float64(np.linalg.norm(diff)),vertex_error_valid=np.int64(1))
            notes['vertex_status']='accepted_physical_3D_vertex'
        else:notes['vertex_status']='rejected_or_missing_physical_3D_vertex'
    target_t=np.asarray(reference.get('truth_kinetic_mev',[]),float)
    if target_t.shape==(kt,):values['reference_initial_kinetic_mev']=target_t.copy()
    energy=np.asarray(row.get('track_vertex_energy_mev',[]),float)
    ok=np.asarray(row.get('energy_ok',np.zeros(len(energy))),bool)
    if target_t.shape==(kt,) and energy.ndim==1 and ok.shape==energy.shape:
        for j,i in enumerate(matching):
            if i<0 or i>=len(energy) or not np.isfinite(energy[i]) or energy[i]<0 or not np.isfinite(target_t[j]) or target_t[j]<0:continue
            values['kinetic_energy_candidate_error_mev'][j]=energy[i]-target_t[j]
            values['kinetic_energy_candidate_valid'][j]=1
            if ok[i]:
                values['kinetic_energy_error_mev'][j]=energy[i]-target_t[j];values['kinetic_energy_valid'][j]=1
        notes['energy_status']='matched_vertex_kinetic_vs_MC_generation_kinetic'
    if kt!=3 or len(set(int(i) for i in matching))!=3 or np.any(np.asarray(matching)<0):
        notes['kinematics_status']='requires_three_matched_alpha_slots';return values,notes
    try:
        alpha,ground,mass_source=_mass_reference(row,reference,cfg,truth_meta);notes['mass_source']=mass_source
        values['reference_alpha_mass_mev']=np.float64(alpha if alpha is not None else np.nan)
        values['reference_carbon_ground_mass_mev']=np.float64(ground if ground is not None else np.nan)
        if 'truth_four_vectors_mev' in reference:
            initial=np.asarray(reference['truth_four_vectors_mev'],float)
        else:
            momentum=np.asarray(reference.get('truth_momentum_uvw_mev',[]),float)
            if momentum.shape!=(3,3) or target_t.shape!=(3,) or alpha is None:
                notes['kinematics_status']='missing_MC_initial_four_vectors_or_explicit_alpha_mass';return values,notes
            initial=np.column_stack((momentum,target_t+alpha))
        ref=_independent_threebody(initial,ground)
        values.update(reference_four_vectors_mev=initial,reference_invariant_mass_mev=np.float64(ref['mass']),
            reference_excitation_mev=np.float64(ref['excitation']),reference_com_pair_angles_deg=ref['angles'],
            reference_dalitz_xy=ref['dalitz'],reference_kinematics_valid=np.int64(1))
        notes['reference_COM_momentum_closure_mev']=ref['momentum_closure']
        notes['reference_angles_defined']=bool(np.isfinite(ref['angles']).all())
        notes['reference_dalitz_defined']=bool(np.isfinite(ref['dalitz']).all())
        if ground is None:notes['excitation_reference_status']='missing_explicit_ground_mass; mass/angles can still be evaluated'
        vectors=np.asarray(row.get('alpha_four_vectors_mev',[]),float)
        candidate_flag=bool(row.get('kinematics_candidate_valid',row.get('kinematics_ok',False)))
        if vectors.shape!=(3,4) or not candidate_flag:
            notes['kinematics_status']='missing_or_failed_reconstructed_kinematics';return values,notes
        order=np.asarray(matching,dtype=int)
        if np.any(order>=3):raise ValueError('Reconstructed three-body slots do not match reference')
        own=_independent_threebody(vectors,ground);pred=_independent_threebody(vectors[order],ground)
        reported_mass=float(row.get('invariant_mass_mev',own['mass']))
        reported_ex=float(row.get('excitation_mev',own['excitation']))
        values['invariant_mass_stage_consistency_mev']=reported_mass-own['mass']
        values['excitation_stage_consistency_mev']=reported_ex-own['excitation']
        angles=pred['angles']
        if 'com_opening_angles_deg' in row:
            declared=np.asarray(row['com_opening_angles_deg'],float)
            pairs=np.asarray(row.get('com_pair_indices',[[0,1],[0,2],[1,2]]),int)
            if declared.shape!=(3,) or pairs.shape!=(3,2):raise ValueError('Invalid stage COM pair arrays')
            lookup={tuple(sorted(map(int,p))):float(a) for p,a in zip(pairs,declared)}
            if set(lookup)!={(0,1),(0,2),(1,2)}:raise ValueError('Stage COM pair ordering must enumerate each pair once')
            angles=np.array([lookup[tuple(sorted((order[i],order[j])))] for i,j in ((0,1),(0,2),(1,2))])
            values['com_pair_angle_stage_consistency_deg']=angles-pred['angles']
        dalitz=pred['dalitz']
        if 'dalitz_xy' in row:
            d=np.asarray(row['dalitz_xy'],float)
            if d.shape!=(2,):raise ValueError('Invalid stage Dalitz coordinates')
            # Invert the labelled Dalitz map, permute energies by the same global
            # charge matching, then reapply it. Never sort energies/angles.
            f=np.array([(2/3.-d[1]-np.sqrt(3.)*d[0])/2.,(2/3.-d[1]+np.sqrt(3.)*d[0])/2.,d[1]+1/3.])
            f=f[order];dalitz=np.array([(f[1]-f[0])/np.sqrt(3.),f[2]-1/3.])
            values['dalitz_stage_consistency_xy']=d-own['dalitz']
        values.update(kinematics_candidate_error_valid=np.int64(1),
            candidate_invariant_mass_error_mev=np.float64(reported_mass-ref['mass']),
            candidate_excitation_error_mev=np.float64(reported_ex-ref['excitation']),
            candidate_com_pair_angle_error_deg=angles-ref['angles'],candidate_dalitz_error_xy=dalitz-ref['dalitz'])
        if bool(row.get('kinematics_ok',False)):
            values.update(kinematics_accepted_error_valid=np.int64(1),
                invariant_mass_error_mev=values['candidate_invariant_mass_error_mev'],
                excitation_error_mev=values['candidate_excitation_error_mev'],
                com_pair_angle_error_deg=values['candidate_com_pair_angle_error_deg'].copy(),dalitz_error_xy=values['candidate_dalitz_error_xy'].copy())
            notes['kinematics_status']='accepted_direction_model_compared_to_MC_initial_four_vectors'
        else:notes['kinematics_status']='candidate_only_direction_model_not_accepted'
    except (ValueError,TypeError,KeyError,IndexError) as error:
        notes.update(kinematics_status='invalid_reference_or_stage_kinematics',message=str(error))
    return values,notes

def evaluate_stage(observation_root,truth_root,output_root,output_json=None,config=None):
    """Compare a self-contained finding/fitting/etc ROOT stage to separate truth.

    Required namespaces must match, event keys are unique, and source groups
    must agree before scoring. Missing physical directions stay NaN with explicit
    coverage/status. This evaluates conditional MC targets, not exact detector
    physics or new DDE fidelity. No inference algorithms are called here.
    """
    cfg=dict(config or {});obs=manifest(observation_root);truth=manifest(truth_root)
    if output_json is not None and Path(output_json).exists():raise FileExistsError(output_json)
    if obs.get('role')!='observations' or truth.get('role')!='truth':
        raise ValueError('Expected observation-role stage and separate truth-role ROOT')
    if Path(observation_root).resolve()==Path(truth_root).resolve():raise ValueError('Observation/truth files must differ')
    namespace=obs.get('source_dataset_uuid')
    if not namespace or namespace!=truth.get('source_dataset_uuid'):
        raise ValueError('Observation/reference source_dataset_uuid namespace mismatch')
    verified=bool(obs.get('source_group_partition_verified',False))
    if not verified and not cfg.get('allow_unverified_source_groups',False):
        raise ValueError('Original source-group partition is not verified')
    ref=_reference_index(truth_root)
    expected=None
    if cfg.get('event_keys') is not None:expected={tuple(map(int,k)) for k in cfg['event_keys']}
    seen=set();records=[]
    metadata=dict(role='evaluation',evaluation_only=True,source_dataset_uuid=namespace,
        source_group_partition_verified=verified,observation_root=str(Path(observation_root).resolve()),
        observation_sha256=sha256(observation_root),reference_root=str(Path(truth_root).resolve()),reference_sha256=sha256(truth_root),
        matching='Single alpha-slot permutation per whole event minimizing total attributed-charge L1; background fixed; zero dummy slots penalize multiplicity mismatch',
        direction_definition='Directed fitted tangent at first observed plane versus MC generation direction; includes propagation/reference-plane differences, NOT pure measurement resolution',
        no_120_degree_constraint=True,physics_accuracy_evaluated=False,config=cfg)
    with StageWriter(output_root,'evaluation',metadata) as writer:
        for row,diagnostic in read_stage(observation_root):
            key=_identity(row)
            if expected is not None and key not in expected:continue
            if key in seen:raise ValueError('Duplicate observation identity')
            seen.add(key)
            if key not in ref:raise ValueError('Missing reference event '+str(key))
            position,source_group=ref[key]
            if int(row['source_group'])!=source_group:raise ValueError('Paired event source_group mismatch')
            target,_=next(read_stage(truth_root,allow_truth=True,entry_indices=[position]))
            if _identity(target)!=key or int(target['source_group'])!=source_group:
                raise ValueError('Reference row identity changed after scalar indexing')
            m=_fractions(row,target,int(cfg.get('tb_bin',truth.get('tb_bin',4))))
            opening,theta,valid,status=_directions(row,diagnostic,target,m['matched_prediction_index'])
            projected_ols=_projected_ols(row,target,m['matched_prediction_index'])
            physics,physics_notes=_physics_errors(row,diagnostic,target,m['matched_prediction_index'],cfg,truth)
            arrays=dict(file_index=np.int64(key[0]),entry=np.int64(key[1]),source_group=row['source_group'],
                charge_l1=np.float64(m['charge_l1']),overlap_fraction_mae=np.float64(m['overlap_fraction_mae']),
                positive_charge=np.float64(m['positive_charge']),has_observed_signal=np.int64(m['has_signal']),
                predicted_k=np.int64(m['predicted_k']),reference_k=np.int64(m['reference_k']),
                matched_prediction_index=m['matched_prediction_index'],
                soft_precision=m['precision'],soft_recall=m['recall'],
                recovered_slots=np.asarray(m['recovered'],dtype=np.uint8),visible_slots=np.asarray(m['true_visible'],dtype=np.uint8),
                direction_opening_error_deg=opening,theta_uv_true_minus_fitted_deg=theta,direction_valid=valid,
                theta_uv_ols_true_minus_fitted_deg=projected_ols,theta_uv_ols_valid=np.isfinite(projected_ols).astype(np.uint8))
            arrays.update(physics)
            writer.write(arrays,dict(direction_status=status,reference_is_evaluation_only=True,
                match_permutation_full=m['slot_permutation'],matching_reused_for_every_metric=True,
                physical_observable_evaluation=physics_notes,
                missing_direction_is_not_zero=not bool(np.all(valid))))
            records.append(dict(file_index=key[0],entry=key[1],charge_l1=m['charge_l1'],positive_charge=m['positive_charge'],
                absolute_charge_error=m['absolute_charge_error'],overlap_fraction_mae=m['overlap_fraction_mae'],
                visible=int(m['true_visible'].sum()),recovered=int(m['recovered'].sum()),direction_valid=int(valid.sum()),
                direction_reference_slots=len(valid),direction_status=status,direction_errors=opening,theta_residuals=theta,
                projected_ols_residuals=projected_ols,physical_errors=physics,physical_status=physics_notes))
        if expected is not None and seen!=expected:raise ValueError('Observation stage lacks requested event keys')
        signal=[r for r in records if r['positive_charge']>0]
        overlap=[r['overlap_fraction_mae'] for r in signal if np.isfinite(r['overlap_fraction_mae'])]
        angles=np.concatenate([r['direction_errors'] for r in records]) if records else np.empty(0)
        projected=np.concatenate([r['theta_residuals'] for r in records]) if records else np.empty(0)
        projected_ols=np.concatenate([r['projected_ols_residuals'] for r in records]) if records else np.empty(0)
        summary=dict(events=len(records),signal_events=len(signal),no_TPC_events=len(records)-len(signal),
            mean_event_charge_l1=float(np.mean([r['charge_l1'] for r in signal])) if signal else None,
            pooled_charge_l1=float(sum(r['absolute_charge_error'] for r in signal)/sum(r['positive_charge'] for r in signal)) if signal else None,
            mean_event_overlap_fraction_mae=float(np.mean(overlap)) if overlap else None,
            overlap_events=len(overlap),visible_reference_slots=sum(r['visible'] for r in records),
            recovered_visible_slots=sum(r['recovered'] for r in records),
            direction_valid_slots=sum(r['direction_valid'] for r in records),direction_reference_slots=sum(r['direction_reference_slots'] for r in records),
            projected_ols_valid_slots=int(np.isfinite(projected_ols).sum()),
            physics_accuracy_evaluated=False)
        summary.update(vertex_accepted_events=sum(int(r['physical_errors']['vertex_error_valid']) for r in records),
            vertex_candidate_events=sum(int(r['physical_errors']['vertex_candidate_error_valid']) for r in records),
            energy_valid_slots=sum(int(r['physical_errors']['kinetic_energy_valid'].sum()) for r in records),
            energy_candidate_slots=sum(int(r['physical_errors']['kinetic_energy_candidate_valid'].sum()) for r in records),
            reference_four_vector_events=sum(int(r['physical_errors']['reference_kinematics_valid']) for r in records),
            kinematics_accepted_events=sum(int(r['physical_errors']['kinematics_accepted_error_valid']) for r in records),
            kinematics_candidate_events=sum(int(r['physical_errors']['kinematics_candidate_error_valid']) for r in records))
        physical_histograms={}
        error_fields=('vertex_error_norm_mm','vertex_candidate_error_norm_mm','kinetic_energy_error_mev',
            'kinetic_energy_candidate_error_mev','invariant_mass_error_mev','excitation_error_mev',
            'com_pair_angle_error_deg','dalitz_error_xy','candidate_invariant_mass_error_mev',
            'candidate_excitation_error_mev','candidate_com_pair_angle_error_deg','candidate_dalitz_error_xy',
            'invariant_mass_stage_consistency_mev','excitation_stage_consistency_mev',
            'com_pair_angle_stage_consistency_deg','dalitz_stage_consistency_xy')
        for field in error_fields:
            all_values=np.concatenate([np.asarray(r['physical_errors'][field]).reshape(-1) for r in records]) if records else np.empty(0)
            finite=all_values[np.isfinite(all_values)];physical_histograms[field]=finite
            summary[field+'_valid_entries']=len(finite)
            if len(finite):
                summary[field+'_mean']=float(finite.mean());summary[field+'_rmse']=float(np.sqrt(np.mean(finite**2)))
        metrics={k:float(v) for k,v in summary.items() if isinstance(v,(int,float)) and v is not None}
        writer.metrics(metrics)
        writer.histogram('charge_l1',[r['charge_l1'] for r in signal],bins=40,range=(0,2))
        writer.histogram('overlap_fraction_mae',overlap,bins=40,range=(0,1))
        writer.histogram('direction_opening_error_deg',angles,bins=90,range=(0,180))
        writer.histogram('theta_uv_true_minus_fitted_deg',projected,bins=120,range=(-30,30))
        writer.histogram('theta_uv_ols_true_minus_fitted_deg',projected_ols,bins=120,range=(-30,30))
        for field,finite in physical_histograms.items():writer.histogram(field,finite,bins=80)
    result=dict(metadata=metadata,summary=summary,per_event=records)
    if output_json is not None:
        with Path(output_json).open('x') as stream:json.dump(jsonable(result),stream,indent=2,allow_nan=False)
    return result
