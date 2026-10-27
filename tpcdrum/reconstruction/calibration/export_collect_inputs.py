#!/usr/bin/env python3
"""Collect bounded calibration assets/source and an optional digi header.

This does not copy a many-GB digi or infer missing calibration values. Existing
files are read-only. Every missing input is listed in collection_manifest.json.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import sys
import tarfile

HERE = Path(__file__).resolve().parent
ASSET_KEYS=('STDGarfieldData','FieldData','PulseShapeData','NoiseShapeData')
SOURCES = [
    'tpcdrum/geant4/TPCDrumConstruction.cpp', 'tpcdrum/geant4/TPCDrumConstruction.h',
    'tpcdrum/detector/TPCDrum.cpp', 'tpcdrum/detector/TPCDrum.h',
    'tpcdrum/detector/STDSiArray.cpp', 'tpcdrum/detector/STDSiArray.h',
    'tpcdrum/detector/STDPadPlane.cpp', 'tpcdrum/detector/STDPadPlane.h',
    'tpcdrum/task/STDDriftElectronMaker.cpp', 'tpcdrum/task/STDDriftElectronMaker.h',
    'tpcdrum/task/STDElectronicsMaker.cpp', 'tpcdrum/task/STDElectronicsMaker.h',
    'tpcdrum/task/STDSiEdepSumTask.cpp', 'tpcdrum/task/STDSiEdepSumTask.h',
    'tpcdrum/tool/STDSimTuningManager.cpp', 'tpcdrum/tool/STDSimTuningManager.h',
    'tpcdrum/tool/STDFieldDistortionMaker.cpp', 'tpcdrum/tool/STDFieldDistortionMaker.h',
    'tpcdrum/tool/STDPulseAnalyzer.cpp', 'tpcdrum/tool/STDPulseAnalyzer.h',
    'tpcdrum/simulation/PhysicsListOption.txt', 'tpcdrum/simulation/run_sim.cc',
    'source/geant4/LKG4RunManager.cpp', 'source/geant4/LKG4RunManager.h',
]


def sha256(path):
    hasher = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024*1024), b''):
            hasher.update(block)
    return hasher.hexdigest()


def parse_parameters(path):
    """Read simple section/key .mac syntax; retain exact original file too."""
    values, group = {}, ''
    for line in Path(path).read_text().splitlines():
        tokens = shlex.split(line, comments=True)
        if not tokens:
            continue
        if tokens[0].endswith('/') and len(tokens) == 1:
            group = tokens[0].rstrip('/')
        elif len(tokens) > 1:
            key = tokens[0] if '/' in tokens[0] else group+'/'+tokens[0]
            values[key] = ' '.join(tokens[1:])
    return values


def expand(value, repo):
    value = value.replace('e{LILAK_PATH}', str(repo)).replace('${LILAK_PATH}', str(repo))
    return Path(os.path.expandvars(value)).expanduser()


def stored_parameter_values(document):
    """Accept native ROOT or optional Uproot export; retain duplicate ambiguity.

    This parses metadata JSON using the standard library only. It never resolves
    a missing stored setting from today's SimParameters.mac.
    """
    if not isinstance(document,dict):raise ValueError('Stored metadata must be a JSON object')
    containers=document.get('containers')
    if isinstance(containers,dict):
        choices=[]
        for key,container in containers.items():
            name,separator,cycle=key.rpartition(';')
            if not separator:name,cycle=key,'0'
            if name=='ParameterContainer':
                if not cycle.isdigit():raise ValueError('Invalid stored parameter cycle')
                choices.append((int(cycle),key,container))
        if not choices:raise ValueError('Stored JSON has no ParameterContainer')
        cycle,key,container=max(choices,key=lambda item:item[0])
        if sum(item[0]==cycle for item in choices)!=1:raise ValueError('Ambiguous highest stored parameter cycle')
        raw=container.get('values_by_name')
        if isinstance(raw,dict):
            values={str(name):list(items) for name,items in raw.items() if isinstance(items,list)}
        else:
            values={}
            for row in container.get('rows',[]):
                if row.get('name') and not row.get('null') and not row.get('is_line_comment') and not row.get('is_comment_out'):
                    values.setdefault(str(row['name']),[]).append(str(row['value']))
    elif isinstance(containers,list):
        choices=[container for container in containers if container.get('key')=='ParameterContainer']
        if len(choices)!=1:raise ValueError('Native stored JSON must have one ParameterContainer')
        key='ParameterContainer';values={}
        for row in choices[0].get('parameters',[]):
            flags=int(row.get('type_flags',0))
            if row.get('is_line_comment') or row.get('is_comment_out') or flags&((1<<8)|(1<<9)):continue
            if row.get('name'):values.setdefault(str(row['name']),[]).append(str(row['value']))
    else:raise ValueError('Unrecognized stored parameter JSON schema')
    duplicates={name:items for name,items in values.items() if len(items)!=1}
    return key,{name:str(items[0]) for name,items in values.items() if len(items)==1},duplicates


def stored_activity_semantics(document):
    """Legacy native JSON cannot establish which named rows were commented."""
    containers=document.get('containers')
    if isinstance(containers,list):
        rows=[row for container in containers if container.get('key')=='ParameterContainer'
              for row in container.get('parameters',[]) if row.get('name')]
        known=all('type_flags' in row or ('is_line_comment' in row and 'is_comment_out' in row) for row in rows)
        return dict(stored_parameter_activity_verified=known,
            parameter_activity_semantics='native_explicit_comment_flags' if known else 'legacy_native_missing_comment_flags_UNVERIFIED',
            references_may_include_commented_parameters=not known)
    choices=[]
    for key,container in containers.items():
        name,separator,cycle=key.rpartition(';')
        if not separator:name,cycle=key,'0'
        if name=='ParameterContainer':choices.append((int(cycle),container))
    container=max(choices,key=lambda item:item[0])[1]
    rows=[row for row in container.get('rows',[]) if row.get('name')]
    known=bool(rows) and all('type_flags' in row for row in rows)
    return dict(stored_parameter_activity_verified=known,
                parameter_activity_semantics='uproot_decoded_type_flags_filter' if known else 'stored_values_map_without_comment_flags_UNVERIFIED',
                references_may_include_commented_parameters=not known)


def resolve_digi(value,repo):
    digi=expand(value,repo)
    if not digi.is_absolute():
        candidates=list(dict.fromkeys(p.resolve() for p in (repo/digi,repo/'tpcdrum/simulation'/digi) if p.is_file()))
        if len(candidates)!=1:raise ValueError('Digi relative path is missing or ambiguous; supply an absolute path')
        digi=candidates[0]
    return digi.resolve(strict=True)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--repo', required=True)
    p.add_argument('--parameters', default='tpcdrum/simulation/SimParameters.mac')
    p.add_argument('--digi', help='optional NEW digi path; header/parameters only, never full file copied')
    p.add_argument('--root-executable', default=shutil.which('root'))
    p.add_argument('--skip-effective-parameters', action='store_true', help='for a ROOT without LILAK dictionaries')
    p.add_argument('--stored-parameter-json', action='append',default=[],
                   help='optional native effective-parameter or Uproot stored-parameter JSON; repeat for separate input evidence')
    p.add_argument('--asset-root',action='append',default=[],
                   help='explicit same-basename candidate folder; collected separately, NEVER certified historical identity')
    p.add_argument('--output', required=True, help='NEW directory only')
    p.add_argument('--max-asset-mb', type=float, default=256.)
    p.add_argument('--geant4-material-export', action='store_true')
    p.add_argument('--dry-run-geant4', action='store_true')
    args = p.parse_args(argv)
    repo = Path(args.repo).expanduser().resolve(strict=True)
    parameter = expand(args.parameters, repo)
    if not parameter.is_absolute(): parameter = repo/parameter
    parameter = parameter.resolve(strict=True)
    output = Path(args.output).expanduser().resolve()
    if output.exists(): raise FileExistsError('Refusing to overwrite '+str(output))
    if not math.isfinite(args.max_asset_mb) or args.max_asset_mb <= 0: raise ValueError('Asset size limit must be finite positive')
    digi=resolve_digi(args.digi,repo) if args.digi else None
    output.mkdir(parents=True)
    archive = output.with_suffix('.tar.gz')
    if archive.exists(): raise FileExistsError('Refusing to overwrite '+str(archive))
    report = dict(schema='tpcdrum-calibration-collection-v2',
                  created_utc=datetime.now(timezone.utc).isoformat(),
                  repository=str(repo), parameter_file=str(parameter),
                  files=[], missing=[], skipped=[], errors=[],
                  raw_digi_copied=False, physical_calibration_validated=False,
                  asset_collections=[],stored_parameter_evidence=[],
                  historical_asset_identity_verified=False)
    protected_sources={digi} if digi is not None else set()
    evidence_paths=[]
    for name in args.stored_parameter_json:
        try:
            source=resolve_digi(name,repo)
            if source.suffix.lower()!='.json' or source.stat().st_size>args.max_asset_mb*1024*1024:
                report['skipped'].append(dict(path=str(source),purpose='stored_parameter_json',
                    reason='must be bounded JSON metadata, not a ROOT/raw input'))
                continue
            document=json.loads(source.read_text())
            stored_parameter_values(document)
            if isinstance(document.get('input'),str):
                try:protected_sources.add(resolve_digi(document['input'],repo))
                except (OSError,ValueError):pass
            evidence_paths.append((source,'supplied'))
        except OSError as error:
            report['missing'].append(dict(path=name,purpose='stored_parameter_json',reason=str(error)))
        except (ValueError,TypeError,KeyError) as error:
            report['errors'].append('Stored metadata '+str(name)+': '+str(error))

    def copy(source, name, optional=False):
        source = Path(source)
        target=None;owns_new_target=False
        try:
            if source.resolve() in protected_sources:
                report['skipped'].append(dict(path=str(source),purpose=name,reason='protected raw digi input is never copied'))
                return None
            if not source.is_file():
                report['missing'].append(dict(path=str(source), purpose=name))
                return None
            if source.stat().st_size > args.max_asset_mb*1024*1024:
                report['skipped'].append(dict(path=str(source),purpose=name,reason='exceeds configured asset size limit'))
                return None
            target=output/name
            if target.exists():
                report['errors'].append('Refusing collection destination collision: '+str(target))
                return None
            owns_new_target=True;target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, target)
            record=dict(source=str(source),copy=str(target.relative_to(output)),bytes=target.stat().st_size,sha256=sha256(target))
            report['files'].append(record)
            return target
        except OSError as error:
            if owns_new_target and target is not None:
                try:target.unlink(missing_ok=True)
                except OSError:pass
            report['skipped'].append(dict(path=str(source),purpose=name,reason='read/copy failed: '+type(error).__name__))
            report['errors'].append('Cannot collect '+str(source)+' for '+name+': '+type(error).__name__+': '+str(error))
            return None

    copy(parameter, 'parameters/'+parameter.name)
    values = parse_parameters(parameter)
    (output/'parsed_parameter_values.json').write_text(json.dumps(values, indent=2)+'\n')
    for name in SOURCES:
        copy(repo/name, 'source/'+name)
    reaction = values.get('NPTool/ReactionFile')
    if reaction:
        reaction = expand(reaction, repo)
        if not reaction.is_absolute(): reaction = repo/'tpcdrum/simulation'/reaction
        copy(reaction, 'parameters/'+reaction.name)
    def collect_assets(settings,label,prefix,metadata=None,flat=False):
        collection=dict(origin=label,metadata=metadata,assets=[],historical_asset_identity_verified=False)
        result={};data_path=settings.get('TPCDrum/SimDataPath')
        for key in ASSET_KEYS:
            name=settings.get('TPCDrum/'+key)
            evidence=dict(parameter='TPCDrum/'+key,stored_or_current_filename=name,
                          reference_directory=data_path,historical_asset_identity_verified=False)
            if metadata is not None:
                evidence['reference_activity_verified']=metadata.get('stored_parameter_activity_verified')
                evidence['reference_activity_semantics']=metadata.get('parameter_activity_semantics','candidate reference only')
            if not name or (not expand(name,repo).is_absolute() and not data_path):
                evidence['status']='missing_or_ambiguous_reference'
                report['missing'].append(dict(purpose=key,origin=label,reason='No unique exact referenced asset path'))
            else:
                actual=expand(name,repo)
                if not actual.is_absolute():
                    folder=expand(data_path,repo)
                    if not folder.is_absolute():folder=repo/'tpcdrum/simulation'/folder
                    actual=folder/actual
                evidence['referenced_path']=str(actual)
                destination=prefix+'/'+('' if flat else key+'/')+Path(name).name
                if flat and (output/destination).exists():destination=prefix+'/'+key+'/'+Path(name).name
                copied=copy(actual,destination)
                evidence['status']='copied_bytes_at_collection_time_ONLY' if copied else 'not_copied_see_missing_or_skipped'
                if copied:
                    copied_record=report['files'][-1]
                    evidence.update(copy=str(copied.relative_to(output)),sha256=copied_record['sha256'],bytes=copied_record['bytes'])
                    result[key]=str(copied)
            collection['assets'].append(evidence)
        report['asset_collections'].append(collection)
        return result
    assets=collect_assets(values,'current_config','assets',flat=True)
    reference_groups=[('current_config',values,None)]
    (output/'asset_collection_layout.json').write_text(json.dumps(dict(current_config='assets/',
        stored_metadata='assets_stored_parameters/',unverified_candidates='assets_unverified_candidates/',
        historical_identity_verified=False),indent=2)+'\n')

    root = Path(args.root_executable).expanduser().resolve(strict=True) if args.root_executable else None
    if root:
        report['root_version'] = subprocess.check_output([str(root), '--version'], text=True).strip()
    if digi is not None:
        report['digi'] = dict(path=str(digi), bytes=digi.stat().st_size)
        if not root:
            report['errors'].append('ROOT executable missing: digi header not exported')
        else:
            for macro, destination, needs_logon in [
                ('export_file_header.C', 'digi_header.json', False),
                ('export_effective_parameters.C', 'digi_effective_parameters.json', True)]:
                if needs_logon and args.skip_effective_parameters:
                    report['skipped'].append(dict(purpose=macro, reason='explicitly skipped'))
                    continue
                expression = '%s(%s,%s)' % (HERE/macro, json.dumps(str(digi)), json.dumps(str(output/destination)))
                cmd = [str(root), '-n', '-l', '-b', '-q']
                if needs_logon: cmd.append(str(repo/'macros/rootlogon.C'))
                cmd.append(expression)
                done = subprocess.run(cmd, cwd=repo/'tpcdrum/simulation', text=True,
                                      stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
                (output/(macro+'.log')).write_text(done.stdout)
                if done.returncode or not (output/destination).is_file():
                    report['errors'].append(macro+' failed; inspect log')
                else:
                    try: json.loads((output/destination).read_text())
                    except ValueError: report['errors'].append(macro+' output is invalid JSON')
    native=output/'digi_effective_parameters.json'
    if native.is_file():evidence_paths.append((native,'native_ROOT_export'))
    stored_groups=[]
    for index,(source,origin) in enumerate(evidence_paths):
        if source.suffix.lower()!='.json' or source.stat().st_size>args.max_asset_mb*1024*1024:
            report['skipped'].append(dict(path=str(source),purpose='stored_parameter_json',
                reason='must be bounded JSON metadata, not a ROOT/raw input'))
            continue
        label='stored_%02d_%s'%(index,sha256(source)[:10])
        metadata=dict(origin=origin,json_source=str(source),json_sha256=sha256(source))
        try:
            document=json.loads(source.read_text());key,settings,duplicates=stored_parameter_values(document)
            if isinstance(document.get('input'),str):
                try:protected_sources.add(resolve_digi(document['input'],repo))
                except (OSError,ValueError):pass
            metadata.update(input=document.get('input'),uuid=document.get('uuid',document.get('file_uuid')),
                            selected_parameter_container_key=key,duplicate_settings=duplicates,
                            supplied_claims_are_not_auto_validated=True)
            metadata.update(stored_activity_semantics(document))
            header=output/'digi_header.json'
            if header.is_file():
                try:
                    actual_uuid=json.loads(header.read_text()).get('uuid')
                    metadata['matches_requested_digi_uuid']=(actual_uuid==metadata['uuid']) if actual_uuid and metadata['uuid'] else None
                except (ValueError,AttributeError):metadata['matches_requested_digi_uuid']=None
            if source.parent!=output:copy(source,'stored_parameter_evidence/'+label+'.json')
            # Even an asset matching its historical filename/path is a current
            # snapshot. A historical content hash/load receipt is still needed.
            collected=collect_assets(settings,'stored_metadata', 'assets_stored_parameters/'+label,metadata)
            stored_groups.append((label,collected))
            reference_groups.append((label,settings,metadata))
        except (ValueError,TypeError,KeyError,OSError) as error:
            metadata['error']=str(error);report['errors'].append('Stored metadata '+str(source)+': '+str(error))
        report['stored_parameter_evidence'].append(metadata)

    candidate_groups=[]
    for root_index,folder in enumerate(args.asset_root):
        candidate_root=expand(folder,repo)
        if not candidate_root.is_absolute():candidate_root=repo/candidate_root
        for label,settings,reference_metadata in reference_groups:
            # Deliberately do NOT replace/modify either the current .mac or the
            # stored document. A candidate folder is a separate evidence class.
            candidate_settings={'TPCDrum/SimDataPath':str(candidate_root)}
            for key in ASSET_KEYS:
                name=settings.get('TPCDrum/'+key)
                if name:candidate_settings['TPCDrum/'+key]=Path(name).name
            candidate_label='candidate_%02d_%s'%(root_index,label)
            metadata=dict(candidate_root=str(candidate_root),reference_group=label,
                          reference_metadata=reference_metadata,
                          same_basename_does_not_prove_historical_identity=True,
                          historical_asset_identity_verified=False)
            collected=collect_assets(candidate_settings,'unverified_local_candidate_root',
                'assets_unverified_candidates/'+candidate_label,metadata)
            candidate_groups.append((candidate_label,collected))

    def export_transport(group_assets,destination):
        if not root or not ('STDGarfieldData' in group_assets or 'PulseShapeData' in group_assets):return
        expression = '%s(%s,%s,%s)' % (HERE/'export_transport_assets.C',
            json.dumps(group_assets.get('STDGarfieldData', '')), json.dumps(group_assets.get('PulseShapeData', '')),
            json.dumps(str(output/destination)))
        done = subprocess.run([str(root), '-n', '-l', '-b', '-q', expression],
                              text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        (output/(destination+'.log')).write_text(done.stdout)
        if done.returncode or not (output/destination).is_file():
            report['errors'].append('Transport asset export '+destination+' failed; inspect log')
        else:
            try: json.loads((output/destination).read_text())
            except ValueError: report['errors'].append('Transport asset export produced invalid JSON')
    export_transport(assets,'transport_assets.json')
    for label,group_assets in stored_groups:export_transport(group_assets,'transport_assets_'+label+'.json')
    for label,group_assets in candidate_groups:export_transport(group_assets,'transport_assets_'+label+'.json')
    if args.geant4_material_export or args.dry_run_geant4:
        command = [sys.executable, str(HERE/'export_build_run.py'), '--repo', str(repo),
                   '--parameters', str(parameter), '--output', str(output/'geant4')]
        if args.dry_run_geant4: command.append('--dry-run')
        done = subprocess.run(command, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        (output/'geant4_build_driver.log').write_text(done.stdout)
        if done.returncode: report['errors'].append('Geant4 export/build preparation failed; inspect log')
    report['collection_has_missing_inputs'] = bool(report['missing'] or report['skipped'] or report['errors'])
    (output/'collection_manifest.json').write_text(json.dumps(report, ensure_ascii=False, indent=2)+'\n')
    with tarfile.open(archive, 'w:gz') as target:
        target.add(output, arcname=output.name)
    print('Collected package:', archive)
    print('Missing:', len(report['missing']), 'skipped:', len(report['skipped']), 'errors:', len(report['errors']))
    print('This package is evidence for calibration work; it does not certify that physical calibration is complete.')
    return 0


if __name__ == '__main__':
    try: raise SystemExit(main())
    except Exception as exc:
        print('COLLECTION ERROR:', exc, file=sys.stderr)
        raise SystemExit(1)
