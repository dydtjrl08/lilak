"""Metadata decoder checks. Actual-data check is optional when file is absent."""
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import hashlib
import numpy as np

from reconstruction.calibration.read_stored_parameters import decode_container,values_by_name,read_stored_parameters,_asset_record,select_container_key


class Model:
    def __init__(self,classname,members=None,bases=None,items=None):
        self.classname=classname;self.all_members=members or {};self.bases=bases or [];self.items=items or []
    def __iter__(self):return iter(self.items)


def parameter(name,raw,value,flags=1,comment=''):
    return Model('LKParameter',dict(fName=name,fTitle=raw,fValue=value,fType=flags,
        fComment=comment,fNumValues=len(value.split()),fValueArray=value.split()))


class StoredParameterTests(unittest.TestCase):
    def test_highest_numeric_cycle_is_selected_independent_of_iteration_order(self):
        for names in (['RunHeader;40','ParameterContainer;10','ParameterContainer;2'],
                      ['ParameterContainer;2','ParameterContainer;10','RunHeader;40']):
            self.assertEqual(select_container_key(names),'ParameterContainer;10')
        self.assertIsNone(select_container_key(['RunHeader;1']))
        with self.assertRaises(ValueError):select_container_key(['ParameterContainer;bad'])
    def test_streamed_array_preserves_order_duplicates_and_raw_vs_evaluated(self):
        items=[parameter('path','e{ROOT}/file.root','/server/data/file.root'),
            parameter('branch/input','GETChannel RawPad','GETChannel RawPad',17),None,
            parameter('branch/input','LKMCTrack MCTrack','LKMCTrack MCTrack',17),
            parameter('pressure','200','200',513,'commented legacy value'),
            parameter('','', '',257,'comment line')]
        container=Model('LKParameterContainer',bases=[Model('TObjArray',items=items)])
        rows=decode_container(container);settings=values_by_name(rows)
        self.assertEqual(rows[0]['raw'],'e{ROOT}/file.root')
        self.assertEqual(rows[0]['value'],'/server/data/file.root')
        self.assertTrue(rows[2]['null'])
        self.assertEqual(settings['branch/input'],['GETChannel RawPad','LKMCTrack MCTrack'])
        self.assertNotIn('pressure',settings);self.assertNotIn('',settings)
    def test_unknown_streamed_layout_is_refused_not_guessed(self):
        with self.assertRaises(ValueError):decode_container(Model('Unrecognized',items=[]))
        with self.assertRaises(ValueError):decode_container(Model('LKParameterContainer',bases=[Model('TObjArray',items=[Model('Wrong')])]))
    def test_same_basename_asset_hash_is_not_historical_identity_proof(self):
        with tempfile.TemporaryDirectory() as folder:
            p=Path(folder)/'transport.root';p.write_bytes(b'not the historical bytes')
            settings={'TPCDrum/STDGarfieldData':['transport.root'],'TPCDrum/SimDataPath':['/historical/common']}
            result=_asset_record(settings,'TPCDrum/STDGarfieldData',[folder])
            self.assertFalse(result['actual_asset_bytes_verified'])
            self.assertEqual(result['stored_joined_path'],'/historical/common/transport.root')
            candidate=result['local_candidates'][0]
            self.assertEqual(candidate['sha256'],hashlib.sha256(p.read_bytes()).hexdigest())
            self.assertFalse(candidate['historical_identity_verified'])
    def test_actual_3c3_metadata_decodes_without_any_ttree_read(self):
        path=Path(__file__).resolve().parents[3]/'received_data/hoyle_track_finding/data/C12_72MeV_3C3_digi.root'
        if not path.is_file():self.skipTest('User original ROOT not supplied in this checkout')
        import uproot
        # Opening any TTree object would invoke read_members. Fail immediately.
        patches=[patch.object(cls,'read_members',side_effect=AssertionError('A TTree was opened'))
                 for name,cls in vars(uproot.models.TTree).items()
                 if name.startswith('Model_TTree_v') and isinstance(cls,type)]
        for guard in patches:guard.start()
        try:result=read_stored_parameters(path)
        finally:
            for guard in reversed(patches):guard.stop()
        evidence=result['stored_setting_evidence']
        self.assertGreater(float(evidence['TPCDrum/GasPressure']),0.)
        self.assertGreater(float(evidence['TPCDrum/TimeBucketUnit']),0.)
        self.assertTrue(np.isfinite(float(evidence['TPCDrum/PulseDelay'])))
        self.assertTrue(result['input_stat_unchanged'])
        self.assertFalse(result['event_trees_opened']);self.assertFalse(result['waveforms_read'])
        self.assertFalse(result['mc_truth_read']);self.assertFalse(result['calibrated_for_charge_mean'])
        self.assertLess(result['uproot_source_requested_bytes'],1024*1024)
        self.assertIn('ParameterContainer;1',result['key_hashes'])
        self.assertEqual(len(result['key_hashes']['ParameterContainer;1']['object_payload_sha256']),64)
        self.assertTrue(all(not asset['actual_asset_bytes_verified'] for asset in result['external_assets']))


if __name__=='__main__':unittest.main()
