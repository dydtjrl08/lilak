"""Independently runnable, observation-only reconstruction stages with ROOT handoff."""
from pathlib import Path
import copy
import json
import subprocess
import sys
import time
import numpy as np

from .stage_io import StageWriter,read_stage,manifest,sha256,jsonable
from .raw_source import read_observed_events,dataset_metadata
from .preprocessing import preprocess,space_points

STAGE_ORDER=('prepare','spacepoints','finding','fitting','vertex','association',
             'material_request','material_import','energy','kinematics')


def _load_json(value):
    if value is None:return None
    if isinstance(value,dict):return value
    return json.loads(Path(value).expanduser().read_text())


def ingest(config,output):
    paths=config['input_roots']
    if not paths:raise ValueError('ingest requires at least one explicit original digitization ROOT')
    entries=config.get('entries')
    meta=dataset_metadata(paths)
    meta.update(config=config, role='observations', stage_scope='RawPad and observed per-Si-module totals only')
    charges=[]
    with StageWriter(output,'ingest',meta) as writer:
        for i,(arrays,diag) in enumerate(read_observed_events(paths,entries,config.get('root_executable'))):
            writer.write(arrays,diag)
            charges.append(float(arrays['raw_waveform'].sum()))
            print('ingest event',i+1,'file',int(arrays['file_index']),'entry',int(arrays['entry']),flush=True)
        writer.metrics({'events':len(charges),'has_physics_accuracy_evaluation':0.})
        writer.histogram('raw_adc_sum',charges)
    return Path(output)


def prepare_event(record,diagnostic,config):
    values,diag=preprocess(record['raw_waveform'],config.get('preprocessing'))
    return values,dict(diagnostic,preprocessing=diag)


def spacepoints_event(record,diagnostic,config):
    values,diag=space_points(record['positive_waveform'],config.get('space_points'),config.get('drift'))
    return values,dict(diagnostic,space_points=diag)


def finding_event(record,diagnostic,config):
    cfg=copy.deepcopy(config.get('finding',{}));cfg.setdefault('k',config.get('n_tracks',3))
    representation=config.get('representation','direct_waveform')
    if cfg.get('method')=='sequential_vertex':
        from .algorithms.sequential_finding import find_event
        cfg['representation']=representation;cfg['drift']=copy.deepcopy(config.get('drift',{}))
        cfg.setdefault('sequential',{})['proposal_mode']=representation
        found=find_event(record,cfg)
    elif representation=='space_points':
        from .spacepoint_finding import find_spacepoint_event
        found=find_spacepoint_event(record,cfg)
    elif representation=='direct_waveform':
        from .algorithms.finding import find_event
        found=find_event(record['positive_waveform'],cfg)
    else:raise ValueError('representation must be space_points or direct_waveform')
    q=record['positive_waveform']
    k=len(found['components'])
    values={'components':np.asarray(found['components'],dtype=np.float32),
            'background_waveform':np.asarray(found['background'],dtype=np.float32),
            'component_fractions':np.asarray(found['fractions'],dtype=np.float32),
            'feature_lines':np.asarray(found['feature_lines'],dtype=np.float64),
            'finding_ambiguous':np.int64(found['diagnostics'].get('ambiguous',True)),
            'finding_max_charge_error_adc':np.float64(np.max(abs(found['components'].sum(0)+found['background']-q)))}
    for key in ('forward_components','forward_background','residual_waveform','point_fractions'):
        if key in found:values[key]=np.asarray(found[key])
    for key,value in found.items():
        if key.startswith('sequential_'):
            values[key]=np.asarray(value)
    return values,dict(diagnostic,finding=dict(found['diagnostics'],representation=representation,
                alpha_slots_not_mc_ids=True,zero_120_degree_constraint=True,configured_k=cfg['k'],output_k=k))


def fitting_event(record,diagnostic,config):
    if config.get('fitting',{}).get('method')=='legacy_sequence_cnn':
        from .algorithms.sequence_fit import fit_event
    else:
        from .algorithms.track_fit import fit_event
    values,diag=fit_event(record,config)
    return values,dict(diagnostic,fitting=diag)


def vertex_event(record,diagnostic,config):
    if config.get('vertex',{}).get('method')=='joint_tracks':
        from .joint_vertex_stage import vertex_event as handler
        return handler(record,diagnostic,config)
    from .algorithms.vertex import fit_common_vertex
    k=len(record['components'])
    origins=np.asarray(record['track_origins_uvw_mm'],float)
    directions=np.asarray(record['track_directions_uvw'],float)
    ok=np.asarray(record['fit_ok'],bool)
    cfg=dict(config.get('vertex',{}))
    values={'vertex_uvw_mm':np.full(3,np.nan),'vertex_valid':np.int64(0),
            'vertex_dimension':np.int64(3),'vertex_max_dca_mm':np.float64(np.nan),
            'vertex_condition_number':np.float64(np.nan)}
    if len(ok)!=k or not np.all(ok):
        return values,dict(diagnostic,vertex={'status':'incomplete_physical_track_set','valid':False})
    allowed={n:cfg[n] for n in ('max_backward_mm','max_forward_mm','condition_limit','max_dca_mm') if n in cfg}
    result=fit_common_vertex(origins,directions,**allowed)
    if result['valid']:
        values['vertex_uvw_mm']=np.asarray(result['vertex_mm'])
        values['vertex_valid']=np.int64(1)
    values['vertex_max_dca_mm']=np.float64(result['max_dca_mm'] if result['max_dca_mm'] is not None else np.nan)
    values['vertex_condition_number']=np.float64(result['condition_number'] if result['condition_number'] is not None else np.nan)
    return values,dict(diagnostic,vertex=result)


def energy_event(record,diagnostic,config):
    from .physics_stages import energy_event as handler
    return handler(record,diagnostic,config)


def kinematics_event(record,diagnostic,config):
    from .kinematics_stage import kinematics_event as handler
    return handler(record,diagnostic,config)


from .physics_stages import associate_event,material_request_event,import_material_paths_event
from .refit_stage import refitting_event

HANDLERS={'association':associate_event,'material_request':material_request_event,
          'material_import':import_material_paths_event,'prepare':prepare_event,'spacepoints':spacepoints_event,'finding':finding_event,
          'fitting':fitting_event,'refitting':refitting_event,'vertex':vertex_event,'energy':energy_event,'kinematics':kinematics_event}


def run_stage(stage,input_root,output_root,config):
    if stage not in HANDLERS:raise ValueError('Unknown stage '+str(stage))
    parent=manifest(input_root)
    metadata={k:v for k,v in parent.items() if k not in ('numeric_fields','events','complete','stage')}
    metadata.update(config=config,parent_root=str(Path(input_root).resolve()),parent_sha256=sha256(input_root),
                    stage=stage,role='observations',physics_accuracy_evaluated=False)
    success=[];conservation=[];angles=[];masses=[];dca=[];ambiguous=[];candidates=[];accepted_angles=[];projected_vertex=[];direction_compatible=[]
    started=time.monotonic()
    with StageWriter(output_root,stage,metadata) as writer:
        for i,(record,diagnostic) in enumerate(read_stage(input_root)):
            updated,diag=HANDLERS[stage](record,diagnostic,config)
            row=dict(record);row.update(updated)
            writer.write(row,diag)
            if 'finding_max_charge_error_adc' in row:conservation.append(float(row['finding_max_charge_error_adc']))
            if 'vertex_max_dca_mm' in row:dca.append(float(row['vertex_max_dca_mm']))
            if 'com_opening_angles_deg' in row:angles.extend(row['com_opening_angles_deg'])
            if 'excitation_mev' in row:masses.append(float(row['excitation_mev']))
            if 'finding_ambiguous' in row:ambiguous.append(float(row['finding_ambiguous']))
            if 'vertex_projected_valid' in row:projected_vertex.append(float(row['vertex_projected_valid']))
            if 'kinematics_candidate_valid' in row:candidates.append(float(row['kinematics_candidate_valid']))
            if 'kinematics_direction_model_compatible' in row:
                direction_compatible.append(float(row['kinematics_direction_model_compatible']))
            if bool(row.get('kinematics_method_accepted',False)):accepted_angles.extend(row['com_opening_angles_deg'])
            flags={'fitting':'fit_ok','refitting':'refit_ok','vertex':'vertex_valid','association':'si_association_unique',
                   'material_request':'material_request_valid','material_import':'material_paths_embedded_valid',
                   'energy':'energy_ok','kinematics':'kinematics_method_accepted'}
            success.append(float(np.all(row[flags[stage]])) if stage in flags else 1.)
            print(stage,'event',i+1,'entry',int(row['entry']),'stage_output_valid',bool(success[-1]),flush=True)
        summary={'events':len(success),'stage_output_valid_fraction':float(np.mean(success)),
                 'elapsed_seconds':time.monotonic()-started,'physics_accuracy_evaluated':0.}
        if ambiguous:summary['finding_ambiguous_event_fraction']=float(np.mean(ambiguous))
        if candidates:summary['kinematics_candidate_event_fraction']=float(np.mean(candidates))
        if direction_compatible:summary['kinematics_direction_model_compatible_fraction']=float(np.mean(direction_compatible))
        if projected_vertex:summary['projected_vertex_valid_fraction']=float(np.mean(projected_vertex))
        writer.metrics(summary)
        if conservation:writer.histogram('charge_conservation_error_adc',conservation)
        if dca:writer.histogram('common_dca_mm',dca)
        if angles:writer.histogram('candidate_com_opening_angle_deg',angles,bins=180,range=(0.,180.))
        if masses:writer.histogram('candidate_excitation_mev',masses,bins=120,range=(0.,15.))
        if accepted_angles:writer.histogram('accepted_method_com_angle_deg',accepted_angles,bins=180,range=(0.,180.))
    return Path(output_root)


def _kinematics_counts(input_root):
    """Count final candidates and explicit method acceptance from saved rows.

    Old kinematics_ok-only files remain inspectable candidates. Their missing
    finding acceptance declaration must not become a successful physical event.
    """
    events=0;accepted=0;candidates=0;direction_compatible=0
    for arrays,_ in read_stage(input_root):
        events+=1
        accepted+=int(bool(arrays.get('kinematics_method_accepted',False)))
        candidates+=int(bool(arrays.get('kinematics_candidate_valid',arrays.get('kinematics_ok',False))))
        direction_compatible+=int(bool(arrays.get('kinematics_direction_model_compatible',False)))
    return dict(events=events,valid_kinematics=accepted,accepted_events=accepted,
                candidate_kinematics_events=candidates,
                direction_model_compatible_events=direction_compatible,
                all_events_have_physical_result=events>0 and accepted==events)


def run_all(config,output_dir):
    out=Path(output_dir).expanduser().resolve()
    out.mkdir(parents=True,exist_ok=True)
    config=copy.deepcopy(config)
    if config.get('navigator',{}).get('enabled') is True and (
            config.get('si_geometry') is None or config.get('energy',{}).get('stopping_tables') is None):
        initialization=out/'native_geometry_initialization'
        _export_native(initialization,config)
        config['si_geometry']=str(initialization/'geant4_materials.json')
        config.setdefault('energy',{})['stopping_tables']=str(initialization/'geant4_materials.json')
    previous=ingest(config,out/'00_observed.root')
    outputs={'ingest':str(previous)}
    for i,stage in enumerate(STAGE_ORDER,1):
        if stage=='spacepoints' and config.get('representation','direct_waveform')!='space_points':continue
        target=out/('%02d_%s.root'%(i,stage))
        previous=run_stage(stage,previous,target,config);outputs[stage]=str(previous)
        if stage=='material_request' and config.get('navigator',{}).get('enabled') is True:
            _navigate(previous,out/'native_trace',config)
        if stage=='energy' and config.get('refitting',{}).get('enabled') is True:
            for j,repeat in enumerate(('refitting','vertex','association','material_request','material_import','energy'),1):
                target=out/('09_%02d_%s_refined.root'%(j,repeat))
                previous=run_stage(repeat,previous,target,config);outputs[repeat+'_refined']=str(previous)
                if repeat=='material_request':
                    if config.get('navigator',{}).get('enabled') is True:
                        _navigate(previous,out/'native_trace_refined',config)
                    else:
                        # The original CSV is bound to the OLD path. Never reuse it
                        # as if a changed trajectory had been traced again.
                        config.setdefault('energy',{})['material_paths_validated']=False
    result=dict(_kinematics_counts(previous),outputs=outputs,physics_accuracy_evaluated=False,
                truth_was_an_inference_input=False)
    (out/'run_summary.json').write_text(json.dumps(jsonable(result),indent=2)+'\n')
    return result


def _navigate(request_root,output,config):
    """Optional explicit server Geant4 step, with only reconstructed paths as input."""
    from .physics_stages import write_material_requests
    if not any(bool(row.get('material_request_valid',False)) for row,_ in read_stage(request_root)):
        print('native geometry: no valid reconstructed paths to trace; retaining event failures',flush=True)
        return
    request_csv=Path(str(output)+'_requests.csv')
    written=write_material_requests(request_root,request_csv,skip_invalid=True)
    _export_native(output,config,request_csv)
    energy=config.setdefault('energy',{})
    energy.update(stopping_tables=str(output/'geant4_materials.json'),material_path_request_csv=str(request_csv),
                  material_path_csv=str(output/'reco_material_paths.csv'),
                  material_path_receipt=str(output/'export_receipt.json'),material_paths_validated=True)
    config['si_geometry']=str(output/'geant4_materials.json')
    (output/'request_selection.json').write_text(json.dumps(jsonable(written),indent=2)+'\n')


def _export_native(output,config,request_csv=None):
    navigator=config['navigator']
    if not navigator.get('repo') or not navigator.get('parameters'):
        raise ValueError('navigator.enabled requires explicit repo and simulation parameters')
    helper=Path(__file__).resolve().parent/'calibration/export_build_run.py'
    command=[sys.executable,str(helper),'--repo',navigator['repo'],'--parameters',navigator['parameters'],
             '--output',str(output)]
    if request_csv is not None:command.extend(['--path-segments',str(request_csv)])
    for key,argument in [('build_dir','--build-dir'),('link_file','--link-file')]:
        if navigator.get(key):command.extend([argument,navigator[key]])
    subprocess.run(command,check=True)
