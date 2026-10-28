#!/usr/bin/env python3
"""Install only new reconstruction paths. No git operation or existing-file edit."""
from pathlib import Path
import argparse
import shutil
import sys


def install(destination_repo):
    source=Path(__file__).resolve().parent
    repo=Path(destination_repo).expanduser().resolve(strict=True)
    simulation=repo/'tpcdrum/simulation'
    if not simulation.is_dir() or simulation.is_symlink() or (repo/'tpcdrum').is_symlink():
        raise ValueError('Need an existing real tpcdrum/simulation directory')
    payload=[(source,repo/'tpcdrum/reconstruction')]
    for name in ('C12_run_reconstruction.C','C12_view_reconstruction.C'):
        macro=source.parent/'simulation'/name
        if not macro.is_file() or macro.is_symlink():raise ValueError('Missing payload '+str(macro))
        payload.append((macro,simulation/name))
    # Check EVERY destination before creating any one of them.
    for src,dest in payload:
        if dest.exists() or dest.is_symlink():raise FileExistsError('Preserved existing path: '+str(dest))
    if any(p.is_symlink() for p in source.rglob('*')):
        raise ValueError('Payload must not contain symlinks')
    ignore=shutil.ignore_patterns('__pycache__','*.pyc','.native_build','_native_build','results','*.pt','*.root')
    shutil.copytree(source,payload[0][1],ignore=ignore)
    for src,dest in payload[1:]:
        # Exclusive creation also protects a file created after the preflight.
        with src.open('rb') as incoming,dest.open('xb') as outgoing:shutil.copyfileobj(incoming,outgoing)
    return [str(dest) for _,dest in payload]


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('destination_repo')
    args=p.parse_args()
    try:
        for path in install(args.destination_repo):print('Installed new path:',path)
    except Exception as error:
        print('INSTALL STOPPED:',error,file=sys.stderr);raise SystemExit(1)
