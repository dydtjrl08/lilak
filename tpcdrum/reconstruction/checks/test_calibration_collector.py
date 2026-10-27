"""Bounded stdlib collector tests; fixture bytes are not physical calibration."""
import importlib.util
import json
from pathlib import Path
import tempfile
import types
import unittest
from unittest.mock import patch

CAL=Path(__file__).resolve().parents[1]/'calibration'
spec=importlib.util.spec_from_file_location('collector_under_test',CAL/'export_collect_inputs.py')
collect=importlib.util.module_from_spec(spec);spec.loader.exec_module(collect)


def native_document(folder,source=None):
    rows=[{'name':'TPCDrum/SimDataPath','value':str(folder),'type_flags':1}]
    rows.extend({'name':'TPCDrum/'+key,'value':filename,'type_flags':1} for key,filename in (
        ('STDGarfieldData','gas.root'),('FieldData','field.root'),('PulseShapeData','pulse.root'),('NoiseShapeData','noise.root')))
    return dict(schema='tpcdrum-persisted-parameters-v1',input=str(source or folder/'original.root'),uuid='STORED_UUID',
        containers=[dict(key='RunHeader',parameters=[]),dict(key='ParameterContainer',parameters=rows)])


def setup(folder):
    repo=folder/'repo';sim=repo/'tpcdrum/simulation';sim.mkdir(parents=True)
    current=repo/'current';historical=repo/'historical'
    current.mkdir();historical.mkdir()
    for directory,prefix in ((current,b'CURRENT'),(historical,b'HISTORICAL_COPY_TODAY')):
        for filename in ('gas.root','field.root','pulse.root','noise.root'):(directory/filename).write_bytes(prefix+filename.encode())
    lines=['TPCDrum/','  SimDataPath "'+str(current)+'"']
    lines+=['  '+key+' '+name for key,name in (('STDGarfieldData','gas.root'),('FieldData','field.root'),('PulseShapeData','pulse.root'),('NoiseShapeData','noise.root'))]
    (sim/'SimParameters.mac').write_text('\n'.join(lines)+'\n')
    metadata=folder/'stored.json';metadata.write_text(json.dumps(native_document(historical)))
    return repo,current,historical,metadata


class CollectorEvidenceTests(unittest.TestCase):
    def run_no_root(self,args):
        with patch.object(collect.shutil,'which',return_value=None):return collect.main(args)
    def test_current_and_stored_same_names_are_separate_and_never_certified_historical(self):
        with tempfile.TemporaryDirectory() as directory:
            folder=Path(directory);repo,current,historical,metadata=setup(folder);out=folder/'collection'
            before=(repo/'tpcdrum/simulation/SimParameters.mac').read_bytes()
            self.run_no_root(['--repo',str(repo),'--output',str(out),'--stored-parameter-json',str(metadata)])
            report=json.loads((out/'collection_manifest.json').read_text())
            self.assertEqual({g['origin'] for g in report['asset_collections']},{'current_config','stored_metadata'})
            self.assertEqual((out/'assets/gas.root').read_bytes(),(current/'gas.root').read_bytes())
            stored=next(g for g in report['asset_collections'] if g['origin']=='stored_metadata')
            gas=next(a for a in stored['assets'] if a['parameter']=='TPCDrum/STDGarfieldData')
            self.assertEqual((out/gas['copy']).read_bytes(),(historical/'gas.root').read_bytes())
            self.assertNotEqual((out/gas['copy']).read_bytes(),(out/'assets/gas.root').read_bytes())
            self.assertFalse(report['physical_calibration_validated']);self.assertFalse(report['historical_asset_identity_verified'])
            self.assertTrue(all(not a['historical_asset_identity_verified'] for g in report['asset_collections'] for a in g['assets']))
            self.assertEqual(before,(repo/'tpcdrum/simulation/SimParameters.mac').read_bytes())
    def test_exact_stored_path_missing_candidate_root_is_separate_not_substitution(self):
        with tempfile.TemporaryDirectory() as directory:
            folder=Path(directory);repo,current,historical,metadata=setup(folder);out=folder/'collection'
            doc=native_document(folder/'missing_old_server_folder');metadata.write_text(json.dumps(doc))
            self.run_no_root(['--repo',str(repo),'--output',str(out),'--stored-parameter-json',str(metadata),'--asset-root',str(current)])
            report=json.loads((out/'collection_manifest.json').read_text())
            stored=next(g for g in report['asset_collections'] if g['origin']=='stored_metadata')
            self.assertTrue(all('not_copied' in a['status'] for a in stored['assets']))
            candidates=[g for g in report['asset_collections'] if g['origin']=='unverified_local_candidate_root']
            self.assertEqual(len(candidates),2)
            self.assertTrue(any(a.get('copy') for g in candidates for a in g['assets']))
            self.assertTrue(all(g['metadata']['same_basename_does_not_prove_historical_identity'] for g in candidates))
            self.assertTrue(report['collection_has_missing_inputs'])
    def test_oversized_stored_asset_is_skipped_before_copy_and_hash(self):
        with tempfile.TemporaryDirectory() as directory:
            folder=Path(directory);repo,current,historical,metadata=setup(folder);out=folder/'collection'
            large=historical/'gas.root';large.write_bytes(b'x'*2048)
            self.run_no_root(['--repo',str(repo),'--output',str(out),'--stored-parameter-json',str(metadata),'--max-asset-mb','.001'])
            report=json.loads((out/'collection_manifest.json').read_text())
            self.assertTrue(any(x['path']==str(large) and 'size limit' in x['reason'] for x in report['skipped']))
            self.assertFalse(any(x['source']==str(large) for x in report['files']))
    def test_duplicate_stored_asset_setting_not_silently_picked_from_current(self):
        with tempfile.TemporaryDirectory() as directory:
            folder=Path(directory);repo,current,historical,metadata=setup(folder);out=folder/'collection'
            doc=native_document(historical);doc['containers'][1]['parameters'].append({'name':'TPCDrum/STDGarfieldData','value':'other.root'})
            metadata.write_text(json.dumps(doc))
            self.run_no_root(['--repo',str(repo),'--output',str(out),'--stored-parameter-json',str(metadata)])
            report=json.loads((out/'collection_manifest.json').read_text())
            stored=next(g for g in report['asset_collections'] if g['origin']=='stored_metadata')
            gas=next(a for a in stored['assets'] if a['parameter']=='TPCDrum/STDGarfieldData')
            self.assertEqual(gas['status'],'missing_or_ambiguous_reference')
            self.assertIn('TPCDrum/STDGarfieldData',stored['metadata']['duplicate_settings'])
    def test_native_root_effective_export_is_automatically_used_without_uproot(self):
        with tempfile.TemporaryDirectory() as directory:
            folder=Path(directory);repo,current,historical,metadata=setup(folder);out=folder/'collection'
            digi=folder/'actual_digi.root';digi.write_bytes(b'not copied')
            root=folder/'root';root.write_text('fixture executable pathname')
            def run(command,**kwargs):
                expression=command[-1];arguments=json.loads('['+expression.split('(',1)[1][:-1]+']')
                destination=Path(arguments[-1])
                if 'export_effective_parameters.C' in expression:doc=native_document(historical,digi)
                elif 'export_file_header.C' in expression:doc=dict(uuid='STORED_UUID',input=str(digi),waveforms_read=False)
                else:doc=dict(schema='fixture-transport',drift_time_calibrated_for_charge_mean=False)
                destination.write_text(json.dumps(doc));return types.SimpleNamespace(returncode=0,stdout='fixture native metadata export')
            with patch.object(collect.shutil,'which',return_value=str(root)),patch.object(collect.subprocess,'check_output',return_value='ROOT fixture'),patch.object(collect.subprocess,'run',side_effect=run):
                collect.main(['--repo',str(repo),'--output',str(out),'--digi',str(digi)])
            report=json.loads((out/'collection_manifest.json').read_text())
            self.assertFalse(report['raw_digi_copied'])
            self.assertTrue(report['stored_parameter_evidence'][0]['matches_requested_digi_uuid'])
            self.assertTrue(any(g['origin']=='stored_metadata' for g in report['asset_collections']))
            self.assertFalse(any(x['source']==str(digi) for x in report['files']))
    def test_known_digi_is_protected_even_with_unlimited_requested_asset_size(self):
        with tempfile.TemporaryDirectory() as directory:
            folder=Path(directory);repo,current,historical,metadata=setup(folder);out=folder/'collection'
            digi=current/'gas.root'
            self.run_no_root(['--repo',str(repo),'--output',str(out),'--digi',str(digi),'--max-asset-mb','10000'])
            report=json.loads((out/'collection_manifest.json').read_text())
            self.assertFalse(any(x['source']==str(digi) for x in report['files']))
            self.assertTrue(any('protected raw digi' in x['reason'] for x in report['skipped']))
    def test_shuffled_uproot_cycles_choose_numeric_latest(self):
        containers={}
        for cycle,name in [(10,'latest.root'),(2,'older.root')]:
            containers['ParameterContainer;'+str(cycle)]={'values_by_name':{'TPCDrum/STDGarfieldData':[name]}}
        key,values,duplicates=collect.stored_parameter_values({'containers':containers})
        self.assertEqual(key,'ParameterContainer;10');self.assertEqual(values['TPCDrum/STDGarfieldData'],'latest.root')
    def test_native_comment_out_wrong_asset_does_not_shadow_active_correct_asset(self):
        document=native_document(Path('/stored'))
        rows=document['containers'][1]['parameters']
        rows.insert(1,dict(name='TPCDrum/STDGarfieldData',value='wrong_commented.root',type_flags=513))
        rows.insert(2,dict(name='TPCDrum/PulseShapeData',value='wrong_line_comment.root',type_flags=257))
        rows.insert(3,dict(name='TPCDrum/FieldData',value='wrong_boolean_comment.root',is_line_comment=False,is_comment_out=True))
        key,settings,duplicates=collect.stored_parameter_values(document)
        self.assertEqual(settings['TPCDrum/STDGarfieldData'],'gas.root')
        self.assertEqual(settings['TPCDrum/PulseShapeData'],'pulse.root')
        self.assertEqual(settings['TPCDrum/FieldData'],'field.root')
        self.assertFalse(duplicates)
        self.assertTrue(collect.stored_activity_semantics(document)['stored_parameter_activity_verified'])
    def test_legacy_native_without_flags_is_explicitly_unverified_in_manifest(self):
        with tempfile.TemporaryDirectory() as directory:
            folder=Path(directory);repo,current,historical,metadata=setup(folder);out=folder/'collection'
            document=native_document(historical)
            for row in document['containers'][1]['parameters']:row.pop('type_flags')
            metadata.write_text(json.dumps(document))
            self.run_no_root(['--repo',str(repo),'--output',str(out),'--stored-parameter-json',str(metadata)])
            report=json.loads((out/'collection_manifest.json').read_text())
            evidence=report['stored_parameter_evidence'][0]
            self.assertFalse(evidence['stored_parameter_activity_verified'])
            self.assertTrue(evidence['references_may_include_commented_parameters'])
            self.assertIn('UNVERIFIED',evidence['parameter_activity_semantics'])
            group=next(group for group in report['asset_collections'] if group['origin']=='stored_metadata')
            self.assertTrue(all(item['reference_activity_verified'] is False for item in group['assets']))
    def test_permission_denied_during_source_probe_or_copy_preserves_manifest_and_archive(self):
        for phase in ('is_file','copy2'):
            with self.subTest(phase=phase),tempfile.TemporaryDirectory() as directory:
                folder=Path(directory);repo,current,historical,metadata=setup(folder);out=folder/'collection'
                forbidden=historical/'gas.root'
                original_probe=Path.is_file;original_copy=collect.shutil.copy2
                def probe(path):
                    if str(path)==str(forbidden):raise PermissionError('fixture denied source stat/probe')
                    return original_probe(path)
                def copying(source,destination,*args,**kwargs):
                    if str(source)==str(forbidden):raise PermissionError('fixture denied source read')
                    return original_copy(source,destination,*args,**kwargs)
                guard=patch.object(Path,'is_file',new=probe) if phase=='is_file' else patch.object(collect.shutil,'copy2',side_effect=copying)
                with guard:self.run_no_root(['--repo',str(repo),'--output',str(out),'--stored-parameter-json',str(metadata)])
                report=json.loads((out/'collection_manifest.json').read_text())
                self.assertTrue(out.with_suffix('.tar.gz').is_file())
                self.assertTrue(any('PermissionError' in error for error in report['errors']))
                self.assertTrue(any(item['path']==str(forbidden) and 'failed' in item['reason'] for item in report['skipped']))
                self.assertFalse(any(item['source']==str(forbidden) for item in report['files']))
                self.assertTrue(any(item['source']==str(historical/'pulse.root') for item in report['files']))
    def test_supplied_metadata_also_protects_its_raw_input_before_current_asset_copy(self):
        with tempfile.TemporaryDirectory() as directory:
            folder=Path(directory);repo,current,historical,metadata=setup(folder);out=folder/'collection'
            raw=current/'gas.root';metadata.write_text(json.dumps(native_document(historical,raw)))
            self.run_no_root(['--repo',str(repo),'--output',str(out),'--stored-parameter-json',str(metadata),'--max-asset-mb','10000'])
            report=json.loads((out/'collection_manifest.json').read_text())
            self.assertFalse(any(item['source']==str(raw) for item in report['files']))
            self.assertTrue(any('protected raw digi' in item['reason'] for item in report['skipped']))
    def test_nan_cap_and_wrong_metadata_file_do_not_allow_raw_reads(self):
        with tempfile.TemporaryDirectory() as directory:
            folder=Path(directory);repo,current,historical,metadata=setup(folder)
            with self.assertRaises(ValueError):self.run_no_root(['--repo',str(repo),'--output',str(folder/'badcap'),'--max-asset-mb','nan'])
            out=folder/'bounded';raw=folder/'should_not_read.root';raw.write_bytes(b'raw bytes')
            self.run_no_root(['--repo',str(repo),'--output',str(out),'--stored-parameter-json',str(raw)])
            report=json.loads((out/'collection_manifest.json').read_text())
            self.assertTrue(any('bounded JSON' in x['reason'] for x in report['skipped']))


if __name__=='__main__':unittest.main()
