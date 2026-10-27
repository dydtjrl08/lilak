"""RawPad-only native waveform ingestion + observed Si per-module totals.

No MCTrack/MCTag/MCStep branch is read by this module. Truth extraction is a
separate evaluation operation. Missing Si energies remain missing, never zero.
"""
from pathlib import Path
import hashlib
import numpy as np

from .raw_native.reader import ObservedReader


def inspect(path):
    import uproot
    path = Path(path).expanduser().resolve(strict=True)
    with uproot.open(path) as source:
        if 'event' not in source:
            raise ValueError('Original digi needs event TTree')
        tree = source['event']
        keys = tree.keys()
        if 'RawPad' not in keys:
            raise ValueError('Input is not RawPad digitization data')
        original_uuid = None
        if 'HoyleSourceUUID' in source:
            original_uuid = source['HoyleSourceUUID'].member('fTitle')
        file_uuid = str(source.file.uuid)
        return dict(path=str(path), bytes=path.stat().st_size, events=tree.num_entries,
                    file_uuid=file_uuid, original_source_uuid=original_uuid,
                    has_source_entry='HoyleSourceEntry' in keys,
                    source_namespace_verified=bool(original_uuid),
                    has_observed_si_edep='EdepSumSiMeV' in keys,
                    has_truth_mc_track='MCTrack' in keys,
                    has_truth_mc_tag='MCTagTPCDrum' in keys,
                    truth_read_by_this_module=False)


def scalar_source_group(info, source_entry):
    # A verified original UUID shares a namespace across split files and K.
    # Without it, do NOT assert cross-file sibling identity from equal integers.
    namespace = info.get('original_source_uuid') or info['file_uuid']
    value = (str(namespace)+':'+str(int(source_entry))).encode()
    return int.from_bytes(hashlib.blake2b(value,digest_size=8).digest(),'little') & ((1<<63)-1)


def dataset_metadata(paths):
    sources=[inspect(p) for p in paths]
    namespace='|'.join(s['file_uuid'] for s in sources)
    return {'source_dataset_uuid':hashlib.sha256(namespace.encode()).hexdigest(),
            'observed_sources':sources,
            'source_group_partition_verified':all(s['has_source_entry'] for s in sources) and
                (len(sources)==1 or all(s['source_namespace_verified'] for s in sources)),
            'cross_file_sibling_identity_verified':all(s['source_namespace_verified'] for s in sources)}


def read_observed_events(paths, entries=None, root_executable=None):
    import uproot
    for file_index, path in enumerate(paths):
        info = inspect(path)
        selected = list(range(info['events'])) if entries is None else [int(e) for e in entries]
        if not selected or len(set(selected)) != len(selected):
            raise ValueError('Event selection is empty or duplicated')
        with uproot.open(info['path']) as source, ObservedReader(info['path'],root_executable) as reader:
            tree = source['event']
            for entry in selected:
                if entry < 0 or entry >= info['events']:
                    raise IndexError('Entry outside observed input')
                raw = reader.read(entry)
                source_entry = entry
                if info['has_source_entry']:
                    source_entry = int(tree['HoyleSourceEntry'].array(
                        entry_start=entry,entry_stop=entry+1,library='np')[0])
                si = np.empty(0,dtype=np.float64)
                si_names=[]
                if info['has_observed_si_edep']:
                    si = np.asarray(tree['EdepSumSiMeV'].array(
                        entry_start=entry,entry_stop=entry+1,library='np')[0],dtype=float)
                    if si.ndim != 1 or not np.isfinite(si).all() or np.any(si < 0):
                        raise ValueError('Invalid observed Si per-detector energy totals')
                    if 'EdepSumSiDetectorNames' in tree:
                        si_names=tree['EdepSumSiDetectorNames'].array(
                            entry_start=entry,entry_stop=entry+1,library='ak').to_list()[0]
                        if len(si_names)!=len(si) or len(set(si_names))!=len(si_names):
                            raise ValueError('Observed Si detector name/energy alignment failure')
                arrays = dict(file_index=np.asarray(file_index,dtype=np.int64),
                              entry=np.asarray(entry,dtype=np.int64),
                              source_group=np.asarray(scalar_source_group(info,source_entry),dtype=np.int64),
                              source_entry=np.asarray(source_entry,dtype=np.int64),
                              raw_waveform=raw, si_edep_mev=si,
                              has_observed_si_edep=np.asarray(info['has_observed_si_edep'],dtype=np.uint8))
                yield arrays, dict(source=info, observed_only=True,
                                   si_detector_names=si_names,
                                   detector_energy_definition='sum per Si detector; no MC track IDs',
                                   missing_si_is_not_zero=not info['has_observed_si_edep'])
