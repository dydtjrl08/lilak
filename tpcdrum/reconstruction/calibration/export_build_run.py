#!/usr/bin/env python3
"""Build the material exporter with an existing run_sim.exe toolchain.

Only a NEW output directory is written. Existing source/build/MC files are not
modified. The initialized exporter deliberately does not call BeamOn or Run.
Actual Geant4/NPTool linking must be checked on the user's simulation server.
"""
import argparse
import hashlib
import json
from pathlib import Path
import shlex
import subprocess
import sys


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def expand_response(arguments, cwd, depth=0):
    if depth > 5:
        raise ValueError('Nested linker response files exceed limit')
    out = []
    for arg in arguments:
        if arg.startswith('@'):
            path = Path(arg[1:])
            if not path.is_absolute():
                path = cwd/path
            out.extend(expand_response(shlex.split(path.read_text()), cwd, depth+1))
        else:
            out.append(arg)
    return out


def commands(repo, build_dir, link_file, output):
    matches = [link_file] if link_file else list(build_dir.rglob('run_sim.exe.dir/link.txt'))
    if len(matches) != 1:
        raise ValueError('Need one run_sim.exe link.txt; supply --link-file explicitly. Found: '+str(matches))
    link_file = Path(matches[0]).resolve(strict=True)
    flags_file = link_file.with_name('flags.make')
    flags = {}
    for line in flags_file.read_text().splitlines():
        if line.startswith(('CXX_DEFINES =', 'CXX_INCLUDES =', 'CXX_FLAGS =')):
            name, value = line.split('=', 1)
            if '$(' in value:
                raise ValueError('Unexpanded Make variable in compiler flags')
            flags[name.strip()] = shlex.split(value)
    if len(flags) != 3:
        raise ValueError('Missing compiler definitions/includes/flags in flags.make')
    link = expand_response(shlex.split(link_file.read_text()), build_dir)
    if not link or any(arg in ('&&', ';', '|') for arg in link):
        raise ValueError('Expected a single linker invocation, not a shell program')
    prefix = link[:2] if Path(link[0]).name in ('ccache', 'sccache') else link[:1]
    source = Path(__file__).resolve().with_name('export_geant4_materials.cc')
    obj = output/'export_geant4_materials.o'
    exe = output/'export_geant4_materials.exe'
    compile_command = prefix + flags['CXX_DEFINES'] + flags['CXX_INCLUDES'] + flags['CXX_FLAGS']
    compile_command += ['-c', str(source), '-o', str(obj)]
    original_main = [i for i, arg in enumerate(link) if arg.endswith(('/run_sim.cc.o', '/run_sim.cc.obj'))]
    if len(original_main) != 1:
        raise ValueError('Cannot identify exactly one run_sim main object in link command')
    link[original_main[0]] = str(obj)
    output_flags = [i for i, arg in enumerate(link) if arg == '-o']
    if len(output_flags) != 1 or output_flags[0]+1 >= len(link):
        raise ValueError('Cannot identify linker output')
    link[output_flags[0]+1] = str(exe)
    return compile_command, link, exe, dict(
        link_file=str(link_file), link_sha256=digest(link_file),
        flags_file=str(flags_file), flags_sha256=digest(flags_file),
        exporter_source_sha256=digest(source),
        main_object_replaced_only=True, original_build_modified=False)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--repo', required=True)
    p.add_argument('--parameters', default='tpcdrum/simulation/SimParameters.mac')
    p.add_argument('--build-dir', default='build')
    p.add_argument('--link-file')
    p.add_argument('--output', required=True, help='NEW directory only')
    p.add_argument('--path-segments', help='optional reconstructed polyline CSV in pad coordinates')
    p.add_argument('--dry-run', action='store_true')
    args = p.parse_args(argv)
    repo = Path(args.repo).expanduser().resolve(strict=True)
    build = Path(args.build_dir).expanduser()
    if not build.is_absolute(): build = repo/build
    build = build.resolve(strict=True)
    parameters = Path(args.parameters).expanduser()
    if not parameters.is_absolute(): parameters = repo/parameters
    parameters = parameters.resolve(strict=True)
    output = Path(args.output).expanduser().resolve()
    if output.exists(): raise FileExistsError('Refusing to overwrite '+str(output))
    output.mkdir(parents=True)
    before = digest(parameters)
    compile_command, link_command, exe, provenance = commands(
        repo, build, args.link_file, output)
    run_command = [str(exe), str(parameters), str(output)]
    path_input = None
    if args.path_segments:
        path_input = Path(args.path_segments).expanduser().resolve(strict=True)
        run_command.append(str(path_input))
    provenance.update(compile_command=compile_command, link_command=link_command,
                      run_command=run_command, parameter_sha256=before,
                      dry_run=args.dry_run, executed=False, physics_accuracy_validated=False)
    if path_input:
        provenance.update(reconstructed_path_csv=str(path_input),
                          reconstructed_path_sha256=digest(path_input),
                          path_origin='caller-supplied reconstructed polyline; verify observation-only provenance')
    receipt = output/'export_receipt.json'
    receipt.write_text(json.dumps(provenance, indent=2)+'\n')
    if args.dry_run:
        print('Prepared commands without execution:', receipt)
        return 0
    for label, command, cwd in [('compile', compile_command, build),
                                ('link', link_command, build),
                                ('export', run_command, repo/'tpcdrum/simulation')]:
        print(label.upper()+':', shlex.join(command), flush=True)
        with (output/(label+'.log')).open('w') as log:
            result = subprocess.run(command, cwd=cwd, stdout=log, stderr=subprocess.STDOUT)
        if result.returncode:
            provenance.update(failed_step=label, returncode=result.returncode)
            receipt.write_text(json.dumps(provenance, indent=2)+'\n')
            raise RuntimeError('%s failed (%d); read %s' %
                               (label, result.returncode, output/(label+'.log')))
    if digest(parameters) != before:
        raise RuntimeError('Parameter file changed during export; provenance is not stable')
    if path_input and digest(path_input) != provenance['reconstructed_path_sha256']:
        raise RuntimeError('Reconstructed path input changed during export')
    obj = json.loads((output/'geant4_materials.json').read_text())
    if not obj.get('all_material_tables_complete'):
        raise RuntimeError('Not all actual material tables were exported; inspect failed_tables')
    provenance.update(executed=True, generated_events=0,
                      material_manifest_sha256=digest(output/'geant4_materials.json'))
    if path_input:
        provenance['material_path_csv_sha256'] = digest(output/'reco_material_paths.csv')
    receipt.write_text(json.dumps(provenance, indent=2)+'\n')
    print('Material export complete:', output/'geant4_materials.json')
    return 0


if __name__ == '__main__':
    try:
        raise SystemExit(main())
    except Exception as exc:
        print('EXPORT ERROR:', exc, file=sys.stderr)
        raise SystemExit(1)
