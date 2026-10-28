"""Evaluation-only step-reader tests; no timing calibration or resolution claim."""
import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest
import numpy as np

from reconstruction.calibration.empirical_reader import EmpiricalStepReader,step_provenance,export_step_truth
from reconstruction.stage_io import read_stage,manifest

ROOT=os.environ.get('ROOT_EXECUTABLE',shutil.which('root'))
CLASSES=r'''
#include "TObject.h"
#include "TClonesArray.h"
#include "TTree.h"
#include "TFile.h"
#include <vector>
class FeasibilityTrack : public TObject {
public:
 int fTrackID=7,fPDG=1000020040;
 std::vector<double> fVX,fVY,fVZ,fPX,fPY,fPZ,fEnergy;
 FeasibilityTrack(int id=7,int pdg=1000020040):fTrackID(id),fPDG(pdg),
  fVX{-1.},fVY{2.},fVZ{-66.},fPX{-3.},fPY{4.},fPZ{200.},fEnergy{6.}{}
 ClassDef(FeasibilityTrack,1);
};
class FeasibilityStep : public TObject {
public:
 int fTrackID=7;double fX=1.,fY=0.,fZ=118.,fTime=4.,fEdep=.01;
 FeasibilityStep(int id=7,double x=1.,double y=0.,double z=118.,double time=4.,double edep=.01):
  fTrackID(id),fX(x),fY(y),fZ(z),fTime(time),fEdep(edep){}
 ClassDef(FeasibilityStep,1);
};
ClassImp(FeasibilityTrack);ClassImp(FeasibilityStep);
void make_empirical_fixture(const char* output) {
 TFile file(output,"RECREATE");TTree event("event","synthetic truth fixture");
 TClonesArray* tracks=new TClonesArray("FeasibilityTrack");
 TClonesArray* steps=new TClonesArray("FeasibilityStep");
 event.Branch("MCTrack",&tracks,32000,0);event.Branch("MCStepTPCDrum",&steps,32000,0);
 int raw[512];for(int i=0;i<512;i++)raw[i]=123456;
 event.Branch("RawPad",raw,"RawPad[512]/I");
 Long64_t source,input;event.Branch("HoyleSourceEntry",&source);event.Branch("HoyleInputEntry",&input);
 TTree meta("HoyleMeta","source identities");int candidate=0,combination,multiplicity=1;
 std::vector<int> original,indices;
 meta.Branch("HoyleSourceEntry",&source);meta.Branch("HoyleCandidateIndex",&candidate);
 meta.Branch("HoyleCombinationIndex",&combination);meta.Branch("HoyleSubsetMultiplicity",&multiplicity);
 meta.Branch("HoyleOriginalTrackIDs",&original);meta.Branch("HoyleAlphaIndicesInTriplet",&indices);
 for(int e=0;e<6;e++){
  tracks->Delete();steps->Delete();source=e<3?1:2;input=100+e;combination=e%3;original={4+e%3};indices={e%3};
  if(e==0){
   new((*tracks)[0]) FeasibilityTrack(7);
   new((*steps)[0]) FeasibilityStep(7,3,24,120,6,.03);
   new((*steps)[1]) FeasibilityStep(9,99,99,99,99,.99);
   new((*steps)[2]) FeasibilityStep(7,1,0,118,4,.01);
   new((*steps)[3]) FeasibilityStep(7,2,12,119,5,.02);
  }else if(e==1){
   new((*tracks)[0]) FeasibilityTrack(7);new((*tracks)[1]) FeasibilityTrack(7);
   new((*steps)[0]) FeasibilityStep(7);
  }else if(e==2){
   new((*tracks)[0]) FeasibilityTrack(7);new((*tracks)[1]) FeasibilityTrack(9);
   new((*steps)[0]) FeasibilityStep(9,9,12,121,7,.05);
  }else if(e==3){
   new((*tracks)[0]) FeasibilityTrack(4,2212);new((*steps)[0]) FeasibilityStep(4);
  }else if(e==4){
   new((*tracks)[0]) FeasibilityTrack(7);new((*steps)[0]) FeasibilityStep(9);
  }else{
   new((*tracks)[0]) FeasibilityTrack(7);new((*steps)[0]) FeasibilityStep(7,1,0,118,4,-.01);
  }
  event.Fill();meta.Fill();
 }
 event.Write();meta.Write();file.Close();delete tracks;delete steps;
}
'''


@unittest.skipUnless(ROOT,'Set ROOT_EXECUTABLE for native step truth tests')
class EmpiricalReaderTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp=tempfile.TemporaryDirectory(prefix='empirical_steps_fixture_');cls.folder=Path(cls.temp.name)
        source=cls.folder/'steps_fixture.C';source.write_text(CLASSES);cls.input=cls.folder/'truth.root'
        code='gROOT->ProcessLine(%s);gROOT->ProcessLine(%s);'%(
            json.dumps('.L '+str(source)+'+'),json.dumps('make_empirical_fixture('+json.dumps(str(cls.input))+')'))
        done=subprocess.run([ROOT,'-n','-l','-b','-q','-e',code],text=True,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,timeout=90)
        if done.returncode or not cls.input.is_file():raise AssertionError('Native truth fixture failed:\n'+done.stdout[-8000:])
    @classmethod
    def tearDownClass(cls):cls.temp.cleanup()
    def test_selected_own_steps_keep_file_order_and_coordinates_no_transform(self):
        with EmpiricalStepReader(self.input,ROOT) as reader:
            steps=reader.steps(0)
            np.testing.assert_allclose(steps,[[7,3,24,120,.03,6],[7,1,0,118,.01,4],[7,2,12,119,.02,5]])
            initial=reader.track(0)
            np.testing.assert_array_equal(initial,[7,-1,2,-66,-3,4,200,6])
            np.testing.assert_array_equal(reader.steps(0,7),steps)
            with self.assertRaises(TypeError):reader.read(0)
            with self.assertRaises(IndexError):reader.steps(6)
            with self.assertRaises(ValueError):reader.steps(0,True)
    def test_duplicate_missing_nonalpha_and_multialpha_auto_selection_fail_explicitly(self):
        for entry,track,pattern in ((1,7,'Duplicate selected'),(2,None,'exactly one MC alpha'),
                                    (0,8,'ID absent'),(3,4,'not an alpha'),(4,7,'no saved'),(5,7,'Invalid selected')):
            with self.subTest(entry=entry,track=track),EmpiricalStepReader(self.input,ROOT) as reader:
                with self.assertRaisesRegex(RuntimeError,pattern):reader.steps(entry,track)
        with EmpiricalStepReader(self.input,ROOT) as reader:
            np.testing.assert_allclose(reader.steps(2,9),[[9,9,12,121,.05,7]])
    def test_siblings_share_source_event_and_candidate_details_are_retained(self):
        rows=step_provenance(self.input,[0,1,2,3])
        self.assertEqual(rows[0]['source_group'],rows[1]['source_group'])
        self.assertEqual(rows[1]['source_group'],rows[2]['source_group'])
        self.assertNotEqual(rows[2]['source_group'],rows[3]['source_group'])
        self.assertTrue(all(row['source_group_partition_verified'] for row in rows))
        self.assertFalse(any(row['cross_file_source_identity_verified'] for row in rows))
        self.assertEqual(rows[0]['candidate_metadata']['HoyleOriginalTrackIDs'],[4])
        self.assertEqual(rows[0]['input_entry'],100)
    def test_separate_truth_root_cannot_be_consumed_as_observations(self):
        target=self.folder/'separate_steps.root';export_step_truth(self.input,[0],target,ROOT)
        self.assertEqual(manifest(target)['role'],'truth')
        with self.assertRaises(ValueError):list(read_stage(target))
        rows=list(read_stage(target,allow_truth=True))
        self.assertEqual(len(rows),1);arrays,diag=rows[0]
        self.assertEqual(arrays['truth_steps_uvw_edep_time'].shape,(3,6))
        np.testing.assert_array_equal(arrays['truth_initial_track_g4'],[7,-1,2,-66,-3,4,200,6])
        self.assertNotIn('positive_waveform',arrays)


if __name__=='__main__':unittest.main()
