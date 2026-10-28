"""Calibration/evaluation-only historical MC-step extraction.

No inference module imports this file. MCStepTPCDrum in the audited 3C1 digi is
ALREADY pad (u,v,w) mm, while initial MCTrack positions/momenta remain G4 axes.
The reader preserves on-file values/order and NEVER applies a coordinate map.
These saved positions are Geant4 step MIDPOINTS, not exact layer crossings.
"""
from pathlib import Path
import numpy as np
from ..raw_native.reader import ObservedReader

STEP_COLUMNS=('track_id','u_mm','v_mm','w_mm','edep_mev','time_ns')
TRACK_COLUMNS=('track_id','VX_mm','VY_mm','VZ_mm','PX_mev','PY_mev','PZ_mev','kinetic_mev')


class EmpiricalStepReader(ObservedReader):
    program_name='step_reader'
    ready_header='EPREADY'

    def read(self,*args,**kwargs):
        raise TypeError('Evaluation truth helper exposes steps(), never observed inference waveforms')

    def _request(self,command,entry,track_id):
        if isinstance(entry,bool) or int(entry)!=entry or not 0<=int(entry)<self.n:
            raise IndexError('Entry outside truth ROOT tree')
        if track_id is not None and (isinstance(track_id,bool) or int(track_id)!=track_id or int(track_id)<=0):
            raise ValueError('Explicit own alpha track_id must be a positive integer')
        if self.proc is None or self.proc.poll() is not None:raise RuntimeError('Truth step reader is closed')
        requested=-1 if track_id is None else int(track_id)
        self.proc.stdin.write(('%s %d %d\n'%(command,entry,requested)).encode('ascii'));self.proc.stdin.flush()
        return requested

    def track(self,entry,track_id=None):
        """Return selected initial [id,VX,VY,VZ,PX,PY,PZ,T], G4 frame, mm/MeV."""
        requested=self._request('T',entry,track_id)
        header=self._header('EPTRACK');e,selected,n=map(int,header[1:])
        if e!=entry or selected<=0 or n!=8 or (requested!=-1 and requested!=selected):
            raise RuntimeError('Initial track reply contract failed')
        result=np.frombuffer(self._read(8*8),dtype='=f8').copy()
        if not np.isfinite(result).all() or result[0]!=selected or result[-1]<0:
            raise ValueError('Invalid own-alpha initial track values')
        return result

    def steps(self,entry,track_id=None):
        """Return saved ordered [id,u,v,w,edep,time]; coordinate transform forbidden."""
        requested=self._request('S',entry,track_id)
        header=self._header('EPSTEP')
        e,selected,n=map(int,header[1:])
        if e!=entry or selected<=0 or (requested!=-1 and selected!=requested) or not 0<n<=1000000:
            raise RuntimeError('Truth step reply identity/count contract failed')
        result=np.frombuffer(self._read(n*6*8),dtype='=f8').copy().reshape(n,6)
        if not np.isfinite(result).all() or np.any(result[:,0]!=selected) or np.any(result[:,4]<0):
            raise ValueError('Invalid own-alpha step values')
        return result


def step_provenance(path,entries):
    """Read selected scalar/identity metadata only; never waveform or MC tracks.

    Every sibling alpha from one original source event stays in the same group;
    candidate/combination fields are retained for audit, never used to split
    candidates of a common event across train and holdout.
    """
    import uproot
    from ..raw_source import scalar_source_group
    path=Path(path).expanduser().resolve(strict=True)
    entries=list(entries)
    if not entries or any(isinstance(e,bool) or int(e)!=e for e in entries) or len(set(entries))!=len(entries):
        raise ValueError('Selected truth entries must be nonempty unique integers')
    with uproot.open(path) as root:
        tree=root['event'];meta=root.get('HoyleMeta')
        if any(e<0 or e>=tree.num_entries for e in entries):raise IndexError('Truth provenance entry outside file')
        if meta is not None and meta.num_entries!=tree.num_entries:raise ValueError('HoyleMeta is not aligned with event')
        original_uuid=str(root['HoyleSourceUUID'].member('fTitle')) if 'HoyleSourceUUID' in root else None
        info=dict(file_uuid=str(root.file.uuid),original_source_uuid=original_uuid)
        fields=('HoyleSourceEntry','HoyleCandidateIndex','HoyleCombinationIndex','HoyleSubsetMultiplicity',
                'HoyleOriginalCarbon12TrackID','HoyleSelectionMode','HoyleAlphaIndicesInTriplet',
                'HoyleOriginalTrackIDs','HoyleOriginalParentIDs','HoyleTripletOriginalTrackIDs')
        result=[]
        for e in entries:
            saved={}
            if meta is not None:
                for name in fields:
                    if name in meta:saved[name]=meta[name].array(entry_start=e,entry_stop=e+1,library='ak').to_list()[0]
            source=None
            if 'HoyleSourceEntry' in tree:source=int(tree['HoyleSourceEntry'].array(entry_start=e,entry_stop=e+1,library='np')[0])
            if 'HoyleSourceEntry' in saved:
                if source is not None and source!=int(saved['HoyleSourceEntry']):raise ValueError('event/HoyleMeta source identity mismatch')
                source=int(saved['HoyleSourceEntry'])
            verified=source is not None
            if source is None:source=int(e)
            input_entry=int(tree['HoyleInputEntry'].array(entry_start=e,entry_stop=e+1,library='np')[0]) if 'HoyleInputEntry' in tree else None
            result.append(dict(entry=int(e),input_entry=input_entry,source_entry=source,
                source_group=scalar_source_group(info,source),file_uuid=info['file_uuid'],
                original_source_uuid=original_uuid,source_group_partition_verified=verified,
                cross_file_source_identity_verified=bool(original_uuid),
                grouping_scope='original source EVENT within verified UUID namespace' if original_uuid else 'source EVENT within this digi-file UUID only',
                candidate_metadata=saved,role='calibration_evaluation_truth_only_NOT_inference'))
    return result


def export_step_truth(path,entries,output,root_executable=None):
    """Streaming separate truth ROOT; no step values enter observation ROOT."""
    from ..stage_io import StageWriter
    provenance=step_provenance(path,entries)
    metadata=dict(role='truth',input=str(Path(path).resolve()),truth_step_columns=list(STEP_COLUMNS),
        truth_initial_track_columns=list(TRACK_COLUMNS),truth_initial_track_frame='recorded Geant4 frame mm, MeV/c, kinetic MeV',
        coordinate='on-file pad u,v,w mm; NO additional coordinate transform',
        step_definition='saved Geant4 step midpoint positions, pre-step time and total deposited energy',
        no_observed_waveform_read=True,no_calibration_fit=True,
        source_group_partition_verified=all(row['source_group_partition_verified'] for row in provenance),
        cross_file_sibling_identity_verified=all(row['cross_file_source_identity_verified'] for row in provenance))
    with EmpiricalStepReader(path,root_executable) as reader,StageWriter(output,'mc_steps_truth',metadata) as writer:
        for row in provenance:
            steps=reader.steps(row['entry'])
            initial=reader.track(row['entry'],int(steps[0,0]))
            candidates=row['candidate_metadata'];original=np.asarray(candidates.get('HoyleOriginalTrackIDs',[]),dtype=np.int64)
            writer.write(dict(file_index=np.int64(0),entry=np.int64(row['entry']),source_group=np.int64(row['source_group']),
                source_entry=np.int64(row['source_entry']),input_entry=np.int64(row['input_entry'] if row['input_entry'] is not None else -1),
                truth_own_alpha_track_id=np.int64(steps[0,0]),truth_steps_uvw_edep_time=steps,truth_initial_track_g4=initial,
                truth_original_track_ids=original,truth_candidate_index=np.int64(candidates.get('HoyleCandidateIndex',-1)),
                truth_combination_index=np.int64(candidates.get('HoyleCombinationIndex',-1))),row)
    return dict(output=str(Path(output).resolve()),events=len(provenance),metadata=metadata)
