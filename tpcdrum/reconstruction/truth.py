"""Explicit MC evaluation export. This module is never imported by inference.

MC tags supervise observed positive ADC attribution, not calibrated electron
counts. Targets are a separate ROOT file; no reconstruction input accepts them.
"""
from pathlib import Path
import numpy as np

from .raw_native.reader import ObservedReader
from .raw_source import inspect,scalar_source_group,dataset_metadata
from .stage_io import StageWriter


class TruthReader(ObservedReader):
    program_name='truth_reader'
    ready_header='CNNREADY'

    def _send(self,command):
        if self.proc is None or self.proc.poll() is not None:
            raise RuntimeError('Evaluation reader is closed')
        self.proc.stdin.write((command+'\n').encode('ascii'));self.proc.stdin.flush()

    def read(self,*args):
        raise TypeError('Use the separate observed RawPad reader for inference')

    def tracks(self,entry):
        self._send('E %d'%int(entry));p=self._header('CNNE')
        if int(p[1])!=entry:raise ValueError('MC track entry mismatch')
        return np.frombuffer(self._read(int(p[2])*8*8),dtype='=f8').reshape(-1,8).copy()

    def labels(self,entry):
        self._send('L %d'%int(entry));p=self._header('CNNL')
        if int(p[1])!=entry:raise ValueError('MC tag entry mismatch')
        n=int(p[2])
        vox=np.frombuffer(self._read(n*4),dtype='=i4').copy()
        ids=np.frombuffer(self._read(n*4),dtype='=i4').copy()
        weights=np.frombuffer(self._read(n*4),dtype='=f4').copy()
        return vox,ids,weights


def export_truth(paths,output,entries,root_executable=None,k=3,tb_bin=4):
    import uproot
    if not 1<=k<=3 or 512%tb_bin:
        raise ValueError('Invalid truth multiplicity/time binning')
    meta={'role':'truth','k':k,'tb_bin':tb_bin,
          'target':'MC-tag magnitude fractions of observed positive coarse-bin ADC',
          'not_an_inference_input':True,'source_paths':[str(Path(p).resolve()) for p in paths]}
    meta.update(dataset_metadata(paths))
    with StageWriter(output,'evaluation_truth',meta) as writer:
        for fi,path in enumerate(paths):
            info=inspect(path)
            if not info['has_truth_mc_track'] or not info['has_truth_mc_tag']:
                raise ValueError('Evaluation requires MCTrack and MCTagTPCDrum')
            with uproot.open(path) as source,TruthReader(path,root_executable) as reader, \
                    ObservedReader(path,root_executable) as observed:
                tree=source['event']
                for entry in entries:
                    tracks=reader.tracks(int(entry))
                    tracks=tracks[np.argsort(tracks[:,0])]
                    if tracks.shape!=(k,8) or len(np.unique(tracks[:,0]))!=k:
                        raise ValueError('Use already-isolated K-alpha digi, with recoil excluded')
                    vox,ids,weights=reader.labels(int(entry))
                    target_full=np.zeros((k+1,12*64*512),dtype=np.float64)
                    slots=np.full(len(ids),k,dtype=int)
                    for slot,track_id in enumerate(tracks[:,0].astype(int)):
                        slots[ids==track_id]=slot
                    np.add.at(target_full,(slots,vox),weights)
                    total=target_full.sum(0)
                    target_full[:,total>0]/=total[total>0]
                    target_full[-1,total==0]=1.
                    raw=observed.read(int(entry))
                    baseline=np.median(raw[...,:30],axis=-1)
                    positive=np.maximum(raw-baseline[...,None],0.)
                    attributed=(target_full*positive.reshape(1,-1)).reshape(k+1,12,64,512)
                    coarse_charge=attributed.reshape(k+1,12,64,512//tb_bin,tb_bin).sum(-1)
                    coarse_total=coarse_charge.sum(0)
                    target=np.divide(coarse_charge,coarse_total[None],out=np.zeros_like(coarse_charge),where=coarse_total[None]>0)
                    target[-1,coarse_total==0]=1.
                    momentum=np.column_stack((-tracks[:,4],tracks[:,6],tracks[:,5]))
                    positions=np.column_stack((-tracks[:,1],tracks[:,3]+66.,tracks[:,2]+128.8))
                    norm=np.linalg.norm(momentum,axis=1)
                    if not np.isfinite(tracks).all() or np.any(norm<=0):
                        raise ValueError('Invalid evaluation MC track')
                    source_entry=int(entry)
                    if info['has_source_entry']:
                        source_entry=int(tree['HoyleSourceEntry'].array(entry_start=entry,entry_stop=entry+1,library='np')[0])
                    writer.write({'file_index':np.int64(fi),'entry':np.int64(entry),
                                  'source_group':np.int64(scalar_source_group(info,source_entry)),
                                  'truth_fraction':target.reshape(k+1,12,64,512//tb_bin).astype(np.float32),
                                  'truth_components':attributed[:-1].astype(np.float32),
                                  'truth_background':attributed[-1].astype(np.float32),
                                  'truth_direction_uvw':momentum/norm[:,None],
                                  'truth_momentum_uvw_mev':momentum,
                                  'truth_kinetic_mev':tracks[:,7],
                                  'truth_vertex_uvw_mm':positions[0],
                                  'truth_alpha_track_ids':tracks[:,0].astype(np.int64)},
                                 {'evaluation_only':True,'initial_MC_direction_not_local_path_direction':True,
                                  'tag_geometry_assumption':'MCTagTPCDrum index is native physical pad ID, as in uploaded digitizer'})
