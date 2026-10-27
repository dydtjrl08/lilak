#!/usr/bin/env python3
"""ROOT stage runner. Also launched by simulation/C12_run_reconstruction.C."""
from pathlib import Path
import argparse
import json
import os
import sys
from datetime import datetime

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage',default='inspect',choices=['inspect','ingest','prepare','spacepoints',
        'finding','fitting','refitting','vertex','association','material_request','material_import',
        'energy','kinematics','all','truth','export_paths','evaluate'])
    parser.add_argument('--config',default=str(Path(__file__).with_name('config.json')))
    parser.add_argument('--input',dest='input_root')
    parser.add_argument('--output',dest='output_root')
    parser.add_argument('--output-dir')
    parser.add_argument('--truth-root',help='Separate evaluation-only ROOT, never an inference input')
    args=parser.parse_args(argv)
    config_path=Path(args.config).expanduser().resolve(strict=True)
    config=json.loads(config_path.read_text())
    # Explicit paths are interpreted relative to the config, so ROOT's cwd does
    # not change which dataset/checkpoint/calibration a job reads.
    def resolve(value):
        path=Path(os.path.expandvars(value)).expanduser()
        return str((config_path.parent/path).resolve() if not path.is_absolute() else path.resolve())
    config['input_roots']=[resolve(p) for p in config.get('input_roots',[])]
    for group,key in [('finding','checkpoint'),('energy','stopping_tables'),
                      ('energy','material_path_csv'),('energy','material_path_request_csv'),('energy','material_path_receipt'),
                      ('navigator','repo'),('navigator','parameters'),('navigator','build_dir'),('navigator','link_file')]:
        if config.get(group,{}).get(key):config[group][key]=resolve(config[group][key])
    legacy=config.get('fitting',{}).get('legacy_sequence',{})
    if legacy.get('checkpoint'):legacy['checkpoint']=resolve(legacy['checkpoint'])
    if isinstance(config.get('si_geometry'),str):config['si_geometry']=resolve(config['si_geometry'])
    from reconstruction.raw_source import inspect
    from reconstruction.workflow import ingest,run_stage,run_all
    from reconstruction.stage_io import jsonable
    if args.stage=='inspect':
        result={'inputs':[inspect(p) for p in config['input_roots']],
                'drift_calibrated':config.get('drift',{}).get('calibrated_for_charge_mean') is True,
                'note':'Availability audit only. No physics accuracy is established.'}
    elif args.stage=='all':
        if args.input_root:config['input_roots']=[str(Path(args.input_root).expanduser().resolve(strict=True))]
        folder=args.output_dir or resolve(config.get('output_dir','results/run_'+datetime.now().strftime('%Y%m%d_%H%M%S')))
        result=run_all(config,folder)
    elif args.stage=='export_paths':
        if not args.input_root or not args.output_root:parser.error('--input stage.root and --output new_paths.csv are required')
        from reconstruction.physics_stages import write_material_requests
        result=write_material_requests(args.input_root,args.output_root,config)
    elif args.stage=='evaluate':
        if not args.input_root or not args.truth_root or not args.output_root:
            parser.error('--input observations.root --truth-root truth.root --output evaluation.root are required')
        from reconstruction.evaluation import evaluate_stage
        result=evaluate_stage(args.input_root,args.truth_root,args.output_root,config=config.get('evaluation'))
    elif args.stage=='truth':
        if not args.output_root:parser.error('--output is required')
        if config.get('entries') is None:parser.error('Truth export requires an explicit bounded entries list')
        from reconstruction.truth import export_truth
        export_truth(config['input_roots'],args.output_root,config['entries'],config.get('root_executable'),
                     k=config.get('n_tracks',3),tb_bin=config.get('finding',{}).get('tb_bin',4))
        result={'truth_root':args.output_root,'not_an_inference_input':True}
    else:
        if not args.output_root:parser.error('--output is required')
        if args.stage=='ingest':ingest(config,args.output_root)
        else:
            if not args.input_root:parser.error('--input previous_stage.root is required')
            run_stage(args.stage,args.input_root,args.output_root,config)
        result={'stage':args.stage,'output':args.output_root,'physics_accuracy_evaluated':False}
    print(json.dumps(jsonable(result),ensure_ascii=False,indent=2),flush=True)
    if args.stage=='all' and not result['all_events_have_physical_result']:
        print('Physical reconstruction is incomplete. Inspect per-event missing-input/ambiguity diagnostics.',file=sys.stderr)
        return 4
    return 0


if __name__=='__main__':
    try:sys.exit(main())
    except Exception as error:
        print('RECONSTRUCTION FAILED: %s: %s'%(type(error).__name__,error),file=sys.stderr,flush=True)
        raise
