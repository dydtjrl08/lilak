"""Read actual stored LKParameterContainer metadata without LILAK or event reads.

Only whitelisted top-level metadata keys are deserialized. No event/HoyleMeta
TTree is opened and no branch/MC/waveform array is read. Stored settings prove
what was written, not successful loading of an external response/transport
asset or the effective behavior of a historical binary. No drift calibration
is inferred or enabled by this exporter.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path


_ALLOWED={"RunHeader","ParameterContainer","HoyleMetaCopyInputFile"}
_ASSET_PARAMETERS=("TPCDrum/STDGarfieldData","TPCDrum/FieldData","TPCDrum/PulseShapeData","TPCDrum/NoiseShapeData")


def _hash(data):
    return hashlib.sha256(data).hexdigest()


def _members(obj):
    members=getattr(obj,"all_members",None)
    if not isinstance(members,dict):raise ValueError("Unrecognized streamed metadata object")
    return members


def _array_base(obj):
    if getattr(obj,"classname",None)=="TObjArray":return obj
    for base in getattr(obj,"bases",[]):
        try:return _array_base(base)
        except ValueError:pass
    raise ValueError("LKParameterContainer has no decoded TObjArray base")


def decode_container(obj):
    """Preserve ordering, duplicates, raw strings, evaluated values and comments.

    LKParameter::GetRaw() returns TNamed::fTitle; GetValue() returns fValue.
    Multiple-key parameters such as branch/input must not be collapsed by a
    dictionary. Empty/comment lines are retained but never promoted to settings.
    """
    rows=[]
    for index,parameter in enumerate(_array_base(obj)):
        if parameter is None:
            rows.append(dict(index=index,null=True));continue
        m=_members(parameter)
        if not all(name in m for name in ("fName","fTitle","fValue","fType")):
            raise ValueError("Unexpected object in parameter container; no guessed decoding")
        flags=int(m["fType"])
        rows.append(dict(index=index,null=False,class_name=getattr(parameter,"classname",type(parameter).__name__),
            name=str(m["fName"]),raw=str(m["fTitle"]),value=str(m["fValue"]),
            unit=str(m.get("fUnit","")),comment=str(m.get("fComment","")),
            group=str(m.get("fGroup","")),main_name=str(m.get("fMainName","")),
            n_values=int(m.get("fNumValues",0)),values=[str(x) for x in m.get("fValueArray",[])],
            type_flags=flags,is_line_comment=bool(flags&(1<<8)),is_comment_out=bool(flags&(1<<9))))
    return rows


def values_by_name(rows):
    result={}
    for row in rows:
        if row.get("null") or not row.get("name") or row.get("is_line_comment") or row.get("is_comment_out"):continue
        result.setdefault(row["name"],[]).append(row["value"])
    return result


def select_container_key(keys,basename="ParameterContainer"):
    """Choose the explicitly largest numeric ROOT cycle, never iteration order."""
    candidates=[]
    for key in keys:
        name,separator,cycle=str(key).rpartition(';')
        if not separator:name,cycle=str(key),'0'
        if name==basename:
            if not cycle.isdigit():raise ValueError('Invalid ROOT metadata cycle: '+str(key))
            candidates.append((int(cycle),str(key)))
    if not candidates:return None
    maximum=max(number for number,key in candidates)
    selected=[key for number,key in candidates if number==maximum]
    if len(selected)!=1:raise ValueError('Ambiguous highest metadata cycle')
    return selected[0]


def _scalar(settings,name):
    found=settings.get(name,[])
    if len(found)!=1:return None
    return found[0]


def _asset_record(settings,name,asset_roots):
    filename=_scalar(settings,name);base=_scalar(settings,"TPCDrum/SimDataPath")
    result=dict(parameter=name,stored_filename=filename,stored_sim_data_path=base,
                actual_asset_bytes_verified=False,sha256=None)
    if filename is None:
        result["status"]="missing_or_duplicate_stored_filename";return result
    path=Path(filename)
    result["stored_joined_path"]=str(path if path.is_absolute() else Path(base or "")/path)
    # A local same-name copy is only a candidate. The historical asset's content
    # hash was not embedded in the digi file; matching basenames cannot prove it.
    candidates=[]
    for root in asset_roots:
        candidate=Path(root)/path.name
        if candidate.is_file():
            digest=hashlib.sha256()
            with candidate.open("rb") as stream:
                for chunk in iter(lambda:stream.read(1024*1024),b""):digest.update(chunk)
            candidates.append(dict(path=str(candidate.resolve()),size=candidate.stat().st_size,sha256=digest.hexdigest(),
                historical_identity_verified=False))
    result["local_candidates"]=candidates
    result["status"]="local_same_name_candidates_UNVERIFIED_historical_identity" if candidates else "external_asset_not_supplied_or_not_checked"
    return result


def read_stored_parameters(path,asset_roots=()):
    import uproot
    path=Path(path)
    before=path.stat()
    result=dict(schema='tpcdrum-stored-parameters-uproot-v1',input=str(path.resolve()),file_uuid=None,file_size=before.st_size,file_mtime_ns=before.st_mtime_ns,
        hash_scope="individual stored metadata key bytes and uncompressed object payload; NOT the entire ROOT file",
        event_trees_opened=False,waveforms_read=False,mc_truth_read=False,
        calibrated_for_charge_mean=False,calibration_status="stored_settings_only_NOT_validated_timing_calibration",
        containers={},named_metadata={},key_hashes={},metadata_keys=[])
    with uproot.open(path) as root:
        result["file_uuid"]=str(root.file.uuid)
        for keyname,classname in root.classnames(recursive=False).items():
            key=root.key(keyname)
            result["metadata_keys"].append(dict(name=keyname,class_name=classname,seek_key=int(key.fSeekKey),
                stored_nbytes=int(key.fNbytes),object_uncompressed_nbytes=int(key.fObjlen)))
            name=keyname.rsplit(";",1)[0]
            if name not in _ALLOWED:continue
            if classname not in ("LKParameterContainer","TNamed"):
                raise ValueError("Whitelisted metadata key has unexpected class: "+keyname+" "+classname)
            if key.fNbytes>8*1024*1024 or key.fObjlen>32*1024*1024:
                raise ValueError("Unexpectedly large metadata key; refusing unbounded read")
            with path.open("rb") as stream:
                stream.seek(key.fSeekKey);raw=stream.read(key.fNbytes)
            if len(raw)!=key.fNbytes:raise IOError("Truncated stored metadata key")
            chunk,_=key.get_uncompressed_chunk_cursor()
            payload=chunk.raw_data.tobytes()
            result["key_hashes"][keyname]=dict(stored_key_sha256=_hash(raw),object_payload_sha256=_hash(payload),
                stored_nbytes=len(raw),object_payload_nbytes=len(payload))
            obj=key.get()
            if classname=="LKParameterContainer":
                rows=decode_container(obj)
                result["containers"][keyname]=dict(class_name=classname,rows=rows,values_by_name=values_by_name(rows))
            else:
                m=_members(obj)
                result["named_metadata"][keyname]=dict(name=str(m["fName"]),title=str(m["fTitle"]))
        result["uproot_source_requested_bytes"]=int(root.file.source.num_requested_bytes)
        result["streamer_schema"]={}
        for name in ("LKParameter","LKParameterContainer"):
            versions=root.file.streamers.get(name,{})
            result["streamer_schema"][name]=[dict(version=int(version),checksum=int(_members(s).get("fCheckSum",0)))
                for version,s in versions.items()]
    # Do not claim binary-effective values by resolving conflicts/duplicates.
    selected=select_container_key(result['containers'])
    result['selected_parameter_container_key']=selected
    latest=result['containers'][selected]['values_by_name'] if selected is not None else {}
    result["stored_setting_evidence"]={name:_scalar(latest,name) for name in (
        "TPCDrum/GasPressure","TPCDrum/GasContourMixRatio","TPCDrum/TimeBucketUnit","TPCDrum/PulseDelay",
        "TPCDrum/TriggerDelay","TPCDrum/ElectronStepSize","TPCDrum/DynamicRange","TPCDrum/FieldDistortionOn",
        "TPCDrum/GarfieldGasOn","TPCDrum/GainCorrectionOn","TPCDrum/GatingGrid","TPCDrum/NoiseShapeOn",
        "TPCDrum/PulseShapeOn","STDPulseAnalyzer/PeakingTime")}
    result["external_assets"]=[_asset_record(latest,name,asset_roots) for name in _ASSET_PARAMETERS]
    result["not_established"]=[
        "historical binary and source snapshot identity",
        "successful loading/content identity of historical gas/field/pulse/noise assets",
        "effective field-dependent drift velocity and longitudinal/transverse diffusion",
        "effective GEM gain distribution and central-pad gain factors",
        "charge-mean timing offset after response convolution, pad delays and event trigger",
        "absolute physical drift w from a raw peak or charge-mean TB",
        "effective pedestal/noise merely from the stored on/off switch"]
    after=path.stat()
    if (before.st_size,before.st_mtime_ns)!=(after.st_size,after.st_mtime_ns):raise RuntimeError("ROOT file changed during metadata audit")
    result["input_stat_unchanged"]=True
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inputs",nargs="+");parser.add_argument("--output-dir",required=True)
    parser.add_argument("--asset-root",action="append",default=[])
    args=parser.parse_args();folder=Path(args.output_dir);folder.mkdir(parents=True,exist_ok=True)
    for path in args.inputs:
        destination=folder/(Path(path).stem+"_stored_parameters.json")
        if destination.exists():raise FileExistsError("Refusing to overwrite audit: "+str(destination))
        result=read_stored_parameters(path,args.asset_root)
        with destination.open("x") as stream:json.dump(result,stream,ensure_ascii=False,indent=2)
        print(str(destination))


if __name__=="__main__":main()
