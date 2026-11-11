#!/usr/bin/env python3
"""Prepare a small equal-energy MC config by copying current server settings."""
import argparse
from pathlib import Path
import re
import sys


def replace_one(text, key, value):
    pattern = r'(?m)^([ \t]*)' + re.escape(key) + r'[ \t]+[^\n]*$'
    changed, count = re.subn(pattern, lambda match: match.group(1) + key + ' ' + value, text)
    if count != 1:
        raise ValueError('Expected one active %s line, found %d' % (key, count))
    return changed


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repo', required=True, type=Path)
    parser.add_argument('--events', type=int, default=1000)
    args = parser.parse_args()
    if args.events <= 0:
        raise ValueError('--events must be positive')
    repo = args.repo.expanduser().resolve()
    sim = repo / 'tpcdrum/simulation'
    source = sim / 'SimParameters.mac'
    config = sim / ('SimParameters_DDE_%d.mac' % args.events)
    output = sim / ('C12_72MeV_DDE_test%d.root' % args.events)
    if config.exists() or output.exists():
        raise ValueError('Test config/output already exists. Keep existing files and use a different --events value.')
    reaction = sim / 'C12_3alpha.reaction'
    reaction_text = reaction.read_text()
    uncommented = '\n'.join(line.split('%', 1)[0].split('#', 1)[0] for line in reaction_text.splitlines())
    required = [r'(?m)^\s*Particle\s*=\s*12C\s*$', r'(?m)^\s*Energy\s*=\s*72(?:\.0*)?\s+MeV\s*$',
                r'(?m)^\s*Heavy\s*=\s*12C\s*$', r'(?m)^\s*ShootHeavy\s*=\s*1\s*$']
    if not all(re.search(pattern, uncommented) for pattern in required):
        raise ValueError('C12_3alpha.reaction is not the expected 72-MeV C12 reaction with ShootHeavy=1; inspect it first.')
    ex = re.search(r'(?m)^\s*ExcitationEnergyHeavy\s*=\s*([0-9.]+)\s+MeV\s*$', uncommented)
    if not ex or abs(float(ex.group(1))-7.65)>.02:
        raise ValueError('Heavy C12 excitation is outside the adapter window 7.65 +/- 0.02 MeV.')
    text = source.read_text()
    text = replace_one(text, 'G4OutputFile', str(output))
    text = replace_one(text, 'run/beamOn', str(args.events))
    text = replace_one(text, 'ReactionFile', 'C12_3alpha.reaction')
    text = replace_one(text, 'VisMode', '0')
    # The run manager still submits G4 commands when VisMode is off.
    text = re.sub(r'(?m)^[ \t]*/?vis/[^\n]*(?:\n|$)', '', text)
    with config.open('x') as handle:
        handle.write(text)
    print('Config:', config)
    print('MC output:', output)
    print('Current gas, gain and other detector settings are copied from SimParameters.mac.')
    print('Run from tpcdrum/simulation:')
    print('HOYLE_EQUAL_ENERGY=1 ./run_sim.exe ' + config.name)


if __name__ == '__main__':
    try:
        main()
    except (OSError, ValueError) as error:
        print('PREPARATION STOPPED:', error, file=sys.stderr)
        sys.exit(1)
