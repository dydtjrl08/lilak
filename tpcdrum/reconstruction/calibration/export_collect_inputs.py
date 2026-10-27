#!/usr/bin/env python3
"""Collect bounded calibration assets/source and an optional digi header.

This does not copy a many-GB digi or infer missing calibration values. Existing
files are read-only. Every missing input is listed in collection_manifest.json.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import sys
import tarfile

HERE = Path(__file__).resolve().parent
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


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--repo', required=True)
    p.add_argument('--parameters', default='tpcdrum/simulation/SimParameters.mac')
    p.add_argument('--digi', help='optional NEW digi path; header/parameters only, never full file copied')
    p.add_argument('--root-executable', default=shutil.which('root'))
    p.add_argument('--skip-effective-parameters', action='store_true', help='for a ROOT without LILAK dictionaries')
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
    if args.max_asset_mb <= 0: raise ValueError('Asset size limit must be positive')
    output.mkdir(parents=True)
    archive = output.with_suffix('.tar.gz')
    if archive.exists(): raise FileExistsError('Refusing to overwrite '+str(archive))
    report = dict(schema='tpcdrum-calibration-collection-v1',
                  created_utc=datetime.now(timezone.utc).isoformat(),
                  repository=str(repo), parameter_file=str(parameter),
                  files=[], missing=[], skipped=[], errors=[],
                  raw_digi_copied=False, physical_calibration_validated=False)

    def copy(source, name, optional=False):
        source = Path(source)
        if not source.is_file():
            report['missing'].append(dict(path=str(source), purpose=name))
            return None
        if source.stat().st_size > args.max_asset_mb*1024*1024:
            report['skipped'].append(dict(path=str(source), reason='exceeds configured asset size limit'))
            return None
        target = output/name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
        report['files'].append(dict(source=str(source), copy=str(target.relative_to(output)),
                                    bytes=target.stat().st_size, sha256=sha256(target)))
        return target

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
    assets = {}
    data_path = values.get('TPCDrum/SimDataPath')
    for key in ('STDGarfieldData', 'FieldData', 'PulseShapeData', 'NoiseShapeData'):
        name = values.get('TPCDrum/'+key)
        if not data_path or not name:
            report['missing'].append(dict(purpose=key, reason='No exact configured asset path'))
            continue
        folder = expand(data_path, repo)
        if not folder.is_absolute(): folder = repo/'tpcdrum/simulation'/folder
        actual = folder/name
        copied = copy(actual, 'assets/'+Path(name).name)
        if copied: assets[key] = str(actual)

    root = Path(args.root_executable).expanduser().resolve(strict=True) if args.root_executable else None
    if root:
        report['root_version'] = subprocess.check_output([str(root), '--version'], text=True).strip()
    if args.digi:
        digi = expand(args.digi, repo)
        if not digi.is_absolute():
            repo_relative, simulation_relative = repo/digi, repo/'tpcdrum/simulation'/digi
            candidates = [p.resolve() for p in (repo_relative, simulation_relative) if p.is_file()]
            candidates = list(dict.fromkeys(candidates))
            if len(candidates) != 1:
                raise ValueError('Digi relative path is missing or ambiguous; supply an absolute path')
            digi = candidates[0]
        digi = digi.resolve(strict=True)
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
    if root and ('STDGarfieldData' in assets or 'PulseShapeData' in assets):
        expression = '%s(%s,%s,%s)' % (HERE/'export_transport_assets.C',
            json.dumps(assets.get('STDGarfieldData', '')), json.dumps(assets.get('PulseShapeData', '')),
            json.dumps(str(output/'transport_assets.json')))
        done = subprocess.run([str(root), '-n', '-l', '-b', '-q', expression],
                              text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        (output/'transport_assets.log').write_text(done.stdout)
        if done.returncode or not (output/'transport_assets.json').is_file():
            report['errors'].append('Transport asset export failed; inspect log')
        else:
            try: json.loads((output/'transport_assets.json').read_text())
            except ValueError: report['errors'].append('Transport asset export produced invalid JSON')
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
