"""Observation-only, independently reloadable ROOT sinks for frozen finding.

This module does not call a model, align slots with truth, score an event, or
change training. Write the fresh inference result BEFORE evaluation-only PIT
alignment. Every method gets its own atomic StageWriter and a stable field set.
A failed call is an explicit K=0 placeholder, not a background prediction claim.
"""
from __future__ import annotations

from contextlib import ExitStack
from copy import deepcopy
from pathlib import Path
from collections.abc import Mapping
import re
import sys

import numpy as np

from .stage_io import StageWriter, jsonable

_FORBIDDEN = {'slot_permutation','aligned_fractions','precision','recall','recovered',
              'true_visible','charge_l1','charge_l1_defined','absolute_charge_error',
              'evaluation_slot_alignment','reference_fractions','predicted_fractions',
              'pair_recovered','pair_reference_visible'}

_WAVEFORM_AVAILABILITY_BASIS = {
    0: 'unavailable: no complete, aligned finite forward waveform arrays',
    1: 'unavailable: explicit boolean False declaration',
    2: 'available: explicit boolean True and complete aligned finite arrays',
    3: 'available: declaration absent; derived from complete aligned finite arrays',
    4: 'unavailable: declaration present but not a boolean',
}


def _reject_evaluation_keys(value,path=''):
    if isinstance(value,Mapping):
        for key,item in value.items():
            name=str(key)
            if name in _FORBIDDEN or name.startswith(('truth_','reference_initial_','reference_pair_','mc_target_','MCTag','MCTrack')):
                raise ValueError('Evaluation/truth input is forbidden in finding handoff: '+path+name)
            _reject_evaluation_keys(item,path+name+'.')
    elif isinstance(value,(list,tuple)):
        for item in value:_reject_evaluation_keys(item,path)


def _array(value,name,shape=None,nonnegative=False,allow_nan=False,dtype=np.float32):
    original=np.asarray(value)
    if np.dtype(dtype).kind in 'iu' and (original.dtype.kind not in 'biuf' or
            not np.isfinite(original).all() or np.any(original!=np.floor(original))):
        raise ValueError(name+' requires exact finite integer values')
    result=np.asarray(value,dtype=dtype)
    if shape is not None and result.shape!=shape:raise ValueError(name+' has invalid shape')
    if np.isinf(result).any() or (not allow_nan and not np.isfinite(result).all()):
        raise ValueError(name+' contains invalid/nonfinite data')
    if nonnegative and np.any(result<0):raise ValueError(name+' contains negative data')
    return result.copy()


def _identity(value,name):
    a=np.asarray(value)
    if a.size!=1 or a.dtype.kind not in 'iu' or int(a.reshape(-1)[0])<0:
        raise ValueError('Need nonnegative integer scalar '+name)
    return np.int64(a.reshape(-1)[0])


def _observations(record):
    if not isinstance(record,Mapping):raise ValueError('Observed record must be a mapping')
    q=_array(record['positive_waveform'],'positive_waveform',(12,64,512),True)
    out={name:_identity(record[name],name) for name in ('file_index','entry','source_group')}
    out.update(positive_waveform=q,
        baseline=_array(record.get('baseline',np.full((12,64),np.nan)),'baseline',(12,64),allow_nan=True),
        noise_sigma=_array(record.get('noise_sigma',np.full((12,64),np.nan)),'noise_sigma',(12,64),True,True),
        tb_ns=np.float64(record.get('tb_ns',np.nan)),has_tb_ns=np.int64('tb_ns' in record),
        si_edep_mev=_array(record.get('si_edep_mev',np.empty(0)),'si_edep_mev',nonnegative=True,dtype=np.float64),
        si_center_uvw_mm=_array(record.get('si_center_uvw_mm',np.empty((0,3))),'si_center_uvw_mm',allow_nan=True,dtype=np.float64),
        has_observed_si_edep=np.int64(bool(record.get('has_observed_si_edep',False))),
        points_uvw_mm=_array(record.get('points_uvw_mm',np.empty((0,3))),'points_uvw_mm',allow_nan=True,dtype=np.float64),
        point_covariance_mm2=_array(record.get('point_covariance_mm2',np.empty((0,3,3))),'point_covariance_mm2',allow_nan=True,dtype=np.float64),
        point_charge=_array(record.get('point_charge',np.empty(0)),'point_charge',nonnegative=True,dtype=np.float64),
        point_tb=_array(record.get('point_tb',np.empty(0)),'point_tb',dtype=np.float64),
        point_layer=_array(record.get('point_layer',np.empty(0)),'point_layer',dtype=np.int64),
        point_pad=_array(record.get('point_pad',np.empty(0)),'point_pad',dtype=np.int64),
        physical_3d=np.int64(bool(record.get('physical_3d',False))))
    if np.isinf(out['tb_ns']) or (out['has_tb_ns'] and (not np.isfinite(out['tb_ns']) or out['tb_ns']<=0)):
        raise ValueError('A supplied TB duration must be positive finite ns')
    n=len(out['point_charge'])
    if (out['points_uvw_mm'].shape!=(n,3) or out['point_covariance_mm2'].shape!=(n,3,3)
            or any(out[name].shape!=(n,) for name in ('point_tb','point_layer','point_pad'))):
        raise ValueError('Observed space-point fields are not aligned')
    if n and (np.any(out['point_layer']<0) or np.any(out['point_layer']>11) or np.any(out['point_pad']<0) or np.any(out['point_pad']>63)):
        raise ValueError('Observed space-point native pad/layer IDs are invalid')
    if out['si_edep_mev'].ndim!=1 or out['si_center_uvw_mm'].ndim!=2 or out['si_center_uvw_mm'].shape[1:]!=(3,):
        raise ValueError('Observed silicon arrays have invalid shape')
    return out


def _observed_si_identity(diagnostic,energies,centers):
    """Only observed per-module identity may cross the diagnostic allowlist."""
    if diagnostic is not None and not isinstance(diagnostic,Mapping):
        raise ValueError('observation_diagnostic must be a mapping or None')
    source=diagnostic or {};result={}
    for key in ('si_detector_names','observed_si_names'):
        if key not in source:continue
        names=source[key]
        if not isinstance(names,(list,tuple)) or any(not isinstance(name,str) or not name for name in names):
            raise ValueError('Observed Si names must be an ordered list of nonempty strings')
        if len(names)!=len(energies) or len(set(names))!=len(names):
            raise ValueError('Observed Si names/energies must be one-to-one and aligned')
        result[key]=list(names)
    if result and any(names!=next(iter(result.values())) for names in result.values()):
        raise ValueError('Conflicting observed Si identity lists')
    if len(energies) and not result:
        raise ValueError('Nonempty observed Si energies require their observed module names')
    if len(centers) not in (0,len(energies)):
        raise ValueError('Observed Si center/name/energy arrays are not aligned')
    return result


def _waveform_availability(found,arrays,k,shape):
    """Numerical forward prediction availability, never fit/physics validity.

    Finding baselines carry NaN placeholders under these same field names.
    Explicit False vetoes availability. A missing declaration may be derived
    from all three finite, aligned arrays (the sequential completion route).
    A present non-boolean declaration, including 0/1 or None, is ambiguous and
    conservatively unavailable rather than silently treated as True/False.
    """
    diagnostic=found.get('diagnostics',{})
    declared='forward_waveform_model_available' in diagnostic
    value=diagnostic.get('forward_waveform_model_available')
    if declared:
        if not isinstance(value,(bool,np.bool_)):return False,4
        if not value:return False,1
    names=('forward_components','forward_background','residual_waveform')
    complete=(all(name in found for name in names)
              and arrays['forward_components'].shape==(k,)+shape
              and arrays['forward_background'].shape==shape
              and arrays['residual_waveform'].shape==shape
              and all(np.isfinite(arrays[name]).all() for name in names))
    if not complete:return False,0
    return True,2 if declared else 3


def _optional_arrays(found,q,k,nt):
    # All rows of a method use these field names and primitive dtypes, including
    # fallback/zero/failure rows. Array lengths may change; ROOT shape branches
    # preserve their actual geometry. Empty/default arrays carry explicit flags.
    seq_shapes=dict(sequential_history=(0,12),sequential_seed_lines=(0,6),sequential_seed_rank=(0,6),
        sequential_confident_masks=(0,12,64,nt),sequential_seed_pair_indices=(0,2),
        sequential_pair_regions=(0,10),sequential_region_envelope=(0,7),
        sequential_candidate_lines=(0,3,6),sequential_candidate_scores=(0,8))
    result={}
    for name,shape in seq_shapes.items():
        dtype=np.int64 if name=='sequential_seed_pair_indices' else np.float64
        result[name]=_array(found.get(name,np.empty(shape)),name,allow_nan=True,dtype=dtype)
        if result[name].ndim!=len(shape) or (len(shape)>1 and result[name].shape[1:]!=shape[1:]):
            raise ValueError(name+' has inconsistent dimensions')
    result.update(sequential_soft_residual=_array(found.get('sequential_soft_residual',q),'sequential_soft_residual',q.shape,allow_nan=True),
        sequential_arrays_available=np.int64(any(name.startswith('sequential_') for name in found)),
        forward_components=_array(found.get('forward_components',np.empty((0,12,64,512))),'forward_components',allow_nan=True),
        forward_background=_array(found.get('forward_background',np.full(q.shape,np.nan)),'forward_background',q.shape,allow_nan=True),
        residual_waveform=_array(found.get('residual_waveform',np.full(q.shape,np.nan)),'residual_waveform',q.shape,allow_nan=True),
        finding_point_fractions=_array(found.get('point_fractions',np.empty((0,k+1))),'point_fractions',nonnegative=True))
    if result['forward_components'].ndim!=4 or result['forward_components'].shape[1:]!=(12,64,512):
        raise ValueError('forward_components geometry mismatch')
    if result['finding_point_fractions'].ndim!=2:raise ValueError('point_fractions must be a matrix')
    available,basis=_waveform_availability(found,result,k,q.shape)
    result.update(waveform_model_available=np.int64(available),waveform_model_availability_basis=np.int64(basis))
    return result


class FindingHandoff:
    """Context-managed method->new ROOT path sinks, no inference or evaluation.

    write(method, observed_record, found=None, error=None) accepts exactly one
    fresh finding result or failed-call error. Optional observation_diagnostic
    carries ONLY si_detector_names/observed_si_names; all other keys are ignored.
    Numerical/schema/closure and I/O
    errors raise; the context aborts instead of silently dropping an event.
    """
    def __init__(self,outputs,observation_metadata=None,*,config=None,
                 checkpoint_sha256=None,split_manifest_sha256=None):
        if not isinstance(outputs,Mapping) or not outputs:raise ValueError('Need method->NEW ROOT path mapping')
        self.outputs={}
        for method,path in outputs.items():
            if not isinstance(method,str) or not re.fullmatch(r'[A-Za-z][A-Za-z0-9_]*',method):
                raise ValueError('Invalid method name')
            target=Path(path).expanduser().resolve()
            if target.exists() or target.is_symlink():raise FileExistsError(target)
            if target in self.outputs.values():raise ValueError('Duplicate handoff output paths')
            self.outputs[method]=target
        metadata=deepcopy(dict(observation_metadata or {}))
        if metadata.get('role','observations')!='observations':raise ValueError('Parent metadata must have observation role')
        self.meta={key:metadata[key] for key in ('source_dataset_uuid','source_group_partition_verified',
                    'original_source_partition_verified','source_file_uuid','source_input_roots','preprocessing') if key in metadata}
        self.config=deepcopy(dict(config or {}));_reject_evaluation_keys(self.config)
        finding=self.config.get('finding',self.config)
        maximum=finding.get('k',self.config.get('n_tracks',3));tb=finding.get('tb_bin',self.config.get('tb_bin',4))
        if any(isinstance(value,(bool,np.bool_)) or not np.isfinite(value) or int(value)!=value for value in (maximum,tb)):
            raise ValueError('Configured K and TB bin must be exact integers')
        self.maximum_k=int(maximum);self.tb_bin=int(tb)
        if not 1<=self.maximum_k<=6 or self.tb_bin<1 or 512%self.tb_bin:raise ValueError('Invalid configured K/TB bin')
        for digest in (checkpoint_sha256,split_manifest_sha256):
            if digest is not None and (not isinstance(digest,str) or not re.fullmatch(r'[0-9a-fA-F]{64}',digest)):
                raise ValueError('Provenance hashes must be SHA-256 hex strings or None')
        self.meta.update(role='observations',inference_only_handoff=True,
            comparison_config=jsonable(self.config),inference_config=jsonable(self.config),checkpoint_sha256=checkpoint_sha256,
            checkpoint_digest_scope='shared comparison reference; does not assert this method consumed the checkpoint',
            split_manifest_sha256=split_manifest_sha256,alpha_slots_not_mc_ids=True,
            slot_order='native inference order; NO evaluation PIT/angle alignment or padding',
            feature_line_coordinates='(u_mm,v_mm,scaled_time_feature); not calibrated physical w',
            physical_w_calibration_claimed=False,truth_was_an_inference_input=False,
            downstream_default_representation='direct_waveform components; optional point shares preserved as finding_point_fractions',
            waveform_model_availability_contract='Numerical forward prediction availability only; explicit False vetoes, explicit True requires all aligned finite forward/background/residual arrays; absent declaration permits finite-array derivation; non-boolean declarations are unavailable. Not fit quality, separation confidence or physics validation.',
            waveform_model_availability_basis_codes=_WAVEFORM_AVAILABILITY_BASIS,
            physics_accuracy_evaluated=False,failure_representation='K=0 explicit unavailable prediction, background=q placeholder')
        self.writers={};self.stack=None;self.closed=False;self.source_groups={}

    def __enter__(self):
        if self.stack is not None or self.closed:raise RuntimeError('Handoff context cannot be reopened')
        self.stack=ExitStack()
        try:
            for method,path in self.outputs.items():
                method_config=deepcopy(self.config)
                if 'finding' in method_config:
                    method_config['finding']=dict(method_config['finding'],method=method)
                else:method_config['method']=method
                self.writers[method]=self.stack.enter_context(StageWriter(path,'finding',
                    dict(self.meta,method=method,inference_config=jsonable(method_config))))
        except BaseException:
            self.stack.__exit__(*sys.exc_info());self.closed=True
            raise
        return self

    def write(self,method,observed_record,found=None,error=None,*,observation_diagnostic=None):
        if self.stack is None or self.closed:raise RuntimeError('Write inside an open FindingHandoff context')
        if method not in self.writers:raise ValueError('Method has no declared handoff output')
        if (found is None)==(error is None):raise ValueError('Supply exactly one fresh found result or error')
        out=_observations(observed_record);q=out['positive_waveform'];nt=512//self.tb_bin
        total_charge=float(q.sum(dtype=np.float64))
        observed_diag=_observed_si_identity(observation_diagnostic,out['si_edep_mev'],out['si_center_uvw_mm'])
        identity=(int(out['file_index']),int(out['entry']));group=int(out['source_group'])
        if identity in self.source_groups and self.source_groups[identity]!=group:
            raise ValueError('Source-group provenance changed across methods')
        self.source_groups[identity]=group
        failed=error is not None
        if failed:
            components=np.empty((0,12,64,512),np.float32);background=q.copy()
            fractions=np.ones((1,12,64,nt),np.float32);lines=np.empty((0,6),np.float64);k=0
            diag=dict(status='inference_failed',ambiguous=True,found_count=0,
                      error_type=type(error).__name__,error_message=str(error),physics_accuracy_evaluated=False)
            fresh={}
        else:
            if not isinstance(found,Mapping):raise ValueError('Fresh finding result must be a mapping')
            _reject_evaluation_keys(found)
            components=_array(found['components'],'components',nonnegative=True)
            if components.ndim!=4 or components.shape[1:]!=(12,64,512):raise ValueError('Invalid components geometry')
            k=len(components)
            if not 0<=k<=self.maximum_k:raise ValueError('Output slot count exceeds configured maximum')
            background=_array(found.get('background',found.get('background_waveform')),'background',q.shape,True)
            fractions=_array(found.get('fractions',found.get('component_fractions')),'fractions',(k+1,12,64,nt),True)
            lines=_array(found.get('feature_lines',np.full((k,6),np.nan)),'feature_lines',(k,6),allow_nan=True,dtype=np.float64)
            diag=deepcopy(dict(found.get('diagnostics',{})));fresh=found
            if not np.allclose(fractions.sum(0),1.,atol=2.e-5,rtol=0.):raise ValueError('Finding fractions are not simplexes')
        scale=max(1.,float(q.max()))
        closure=components.sum(0,dtype=np.float64)+background-q
        if not np.allclose(closure,0.,atol=2.e-6*scale,rtol=0.):raise ValueError('Components/background do not conserve observed charge')
        coarse=q.reshape(12,64,nt,self.tb_bin).sum(-1)
        assigned=np.concatenate((components,background[None])).reshape(k+1,12,64,nt,self.tb_bin).sum(-1)
        if not np.allclose(fractions*coarse[None],assigned,atol=2.e-5*max(1.,float(coarse.max())),rtol=2.e-5):
            raise ValueError('Finding fractions differ from embedded observed component attribution')
        ambiguity_known=isinstance(diag.get('ambiguous'),(bool,np.bool_))
        ambiguous=bool(diag['ambiguous']) if ambiguity_known else True
        ambiguous=ambiguous or failed or total_charge==0.
        found_count=0 if failed or total_charge==0. else int(diag.get('found_count',-1))
        if not -1<=found_count<=k:raise ValueError('Reported found_count is inconsistent with native output slots')
        out.update(components=components,background_waveform=background,component_fractions=fractions,
            feature_lines=lines,finding_prediction_available=np.int64(not failed),finding_failed=np.int64(failed),
            finding_ambiguous=np.int64(ambiguous or failed),finding_confidence_assessed=np.int64(ambiguity_known),
            finding_found_count=np.int64(found_count),finding_output_slot_count=np.int64(k),
            finding_configured_maximum_k=np.int64(self.maximum_k),finding_observed_zero_charge=np.int64(total_charge==0.),
            finding_max_charge_error_adc=np.float64(np.max(abs(closure))),
            finding_feature_line_valid=np.isfinite(lines).all(1).astype(np.uint8))
        out.update(_optional_arrays(fresh,q,k,nt))
        if len(out['finding_point_fractions']) and out['finding_point_fractions'].shape!=(len(out['point_charge']),k+1):
            raise ValueError('point_fractions do not align with observed points/output slots')
        info=dict(diag,ambiguous=bool(ambiguous),method_reported_ambiguous=diag.get('ambiguous'),
                  alpha_slots_not_mc_ids=True,output_k=k,configured_k=self.maximum_k,
                  prediction_available=not failed,placeholder_due_to_failure=failed,
                  no_evaluation_slot_alignment=True,physical_w_calibration_claimed=False)
        info['waveform_model_availability']=dict(available=bool(out['waveform_model_available']),
            basis=_WAVEFORM_AVAILABILITY_BASIS[int(out['waveform_model_availability_basis'])],
            physics_validated=False)
        self.writers[method].write(out,dict(observed_diag,finding=info,handoff_observed_only=True))

    def __exit__(self,exc_type,exc_value,traceback):
        if self.stack is None:return False
        try:
            if exc_type is None:
                counts={method:writer.count for method,writer in self.writers.items()}
                identities=[writer.identities for writer in self.writers.values()]
                if (len(set(counts.values()))!=1 or not counts or min(counts.values())==0
                        or any(keys!=identities[0] for keys in identities[1:])):
                    raise ValueError('All declared methods must retain the same nonempty event cohort')
                for method,writer in self.writers.items():
                    writer.metrics(dict(handoff_events=writer.count,physics_accuracy_evaluated=0.))
            return self.stack.__exit__(exc_type,exc_value,traceback)
        except BaseException:
            self.stack.__exit__(*sys.exc_info())
            raise
        finally:self.closed=True
