"""Native metadata round-trip using temporary API-compatible stand-in classes.

This checks ROOT compilation/streaming, exporter JSON and comment activity. The
stand-ins are NOT server LILAK dictionaries or evidence of physical calibration.
A deliberately damaged event TTree payload checks metadata export independence
from event deserialization. Only temporary synthetic ROOT files are modified.
"""
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest

from reconstruction.calibration.export_collect_inputs import stored_parameter_values

BASE = Path(__file__).resolve().parents[1]
ROOT = os.environ.get('ROOT_EXECUTABLE', shutil.which('root'))

CLASSES = r'''
#include "TNamed.h"
#include "TObjArray.h"
#include "TFile.h"
#include "TTree.h"
#include "TKey.h"
#include "TString.h"
#include <cstdio>

// Minimal verified public API; intentionally not the production class layout.
class LKParameter : public TNamed {
public:
  LKParameter() = default;
  LKParameter(const char* name,const char* raw,const char* value,
              const char* unit="",int flags=1)
    : TNamed(name,raw),fValue(value),fUnit(unit),fType(flags) {}
  TString GetRaw() const { return GetTitle(); }
  TString GetValue() const { return fValue; }
  TString GetUnit() const { return fUnit; }
  int GetType() const { return fType; }
  bool IsLineComment() const { return (fType & (1<<8)) != 0; }
  bool IsCommentOut() const { return (fType & (1<<9)) != 0; }
private:
  TString fValue,fUnit;
  int fType=1;
  ClassDef(LKParameter,1);
};
class LKParameterContainer : public TObjArray {
public:
  LKParameterContainer() { SetOwner(true); }
  ClassDef(LKParameterContainer,1);
};
ClassImp(LKParameter);
ClassImp(LKParameterContainer);

void make_parameter_fixture(const char* path,const char* key_info) {
  TFile file(path,"RECREATE","synthetic metadata fixture",9);
  LKParameterContainer header;
  header.Add(new LKParameter("ProjectVersion/0","synthetic-v1","synthetic-v1"));
  header.Write("RunHeader",TObject::kSingleKey);
  { LKParameterContainer old;
    old.Add(new LKParameter("TPCDrum/PulseShapeData","obsolete.root","obsolete.root"));
    old.Write("ParameterContainer",TObject::kSingleKey); }
  LKParameterContainer parameters;
  parameters.Add(new LKParameter("TPCDrum/SimDataPath","e{LILAK_PATH}/common","/actual/common"));
  parameters.Add(new LKParameter("TPCDrum/PulseShapeData","active.root","active.root"));
  parameters.Add(new LKParameter("TPCDrum/PulseShapeData","commented.root","commented.root","",1|(1<<9)));
  parameters.Add(new LKParameter("TPCDrum/FieldData","line-comment.root","line-comment.root","",1|(1<<8)));
  parameters.Add(new LKParameter("TPCDrum/TimeBucketUnit","40","40","ns",2));
  parameters.Add(new LKParameter("branch/input","GETChannel RawPad","GETChannel RawPad","",17));
  parameters.Add(new LKParameter("branch/input","LKMCTrack MCTrack","LKMCTrack MCTrack","",17));
  parameters.Add(new LKParameter("quoted","raw\\path\"quote\nnext\tcolumn", "value\\path\"quote\nnext\tcolumn"));
  parameters.Write("ParameterContainer",TObject::kSingleKey);
  TNamed unrelated("unrelated","not a parameter container");unrelated.Write();
  // Numerous branch metadata make the TTree key payload reliably compressed.
  TTree event("event","must not be deserialized by the parameter exporter");
  int number=5;
  for(int i=0;i<100;++i)event.Branch(Form("waveform_%04d",i),&number,Form("waveform_%04d/I",i));
  event.Fill();event.Write();
  auto* key=file.GetKey("event");
  FILE* out=fopen(key_info,"wx");
  fprintf(out,"{\"payload_offset\":%lld,\"payload_bytes\":%d,\"object_bytes\":%d}\n",
          key->GetSeekKey()+key->GetKeylen(),key->GetNbytes()-key->GetKeylen(),key->GetObjlen());
  fclose(out);file.Close();
}
'''


@unittest.skipUnless(ROOT, 'Set ROOT_EXECUTABLE for native ROOT parameter round-trip')
class RootParameterExportTests(unittest.TestCase):
    @classmethod
    def run_root(cls, code):
        return subprocess.run([ROOT, '-n', '-l', '-b', '-q', '-e', code],
                              text=True, stdout=subprocess.PIPE,
                              stderr=subprocess.STDOUT, timeout=90)

    @classmethod
    def load_classes(cls):
        return 'gROOT->ProcessLine(%s);' % json.dumps('.L '+str(cls.classes)+'+')

    @classmethod
    def setUpClass(cls):
        cls.directory = tempfile.TemporaryDirectory(prefix='reco_native_parameter_')
        cls.folder = Path(cls.directory.name)
        cls.classes = cls.folder/'parameter_fixture.C'
        cls.classes.write_text(CLASSES)
        cls.input = cls.folder/'parameters.root'
        cls.key_info = cls.folder/'event_key.json'
        expression = 'make_parameter_fixture(%s,%s)' % (json.dumps(str(cls.input)), json.dumps(str(cls.key_info)))
        done = cls.run_root(cls.load_classes()+'gROOT->ProcessLine(%s);' % json.dumps(expression))
        if done.returncode or not cls.input.is_file() or not cls.key_info.is_file():
            cls.directory.cleanup()
            raise AssertionError('ROOT fixture compilation/write failed:\n'+done.stdout[-8000:])

    @classmethod
    def tearDownClass(cls):
        cls.directory.cleanup()

    def export(self, source, output):
        macro = BASE/'calibration/export_effective_parameters.C'
        expression = 'export_effective_parameters(%s,%s)' % (json.dumps(str(source)),json.dumps(str(output)))
        code = self.load_classes()
        code += 'Int_t err=0;gROOT->ProcessLine(%s,&err);if(err)gSystem->Exit(91);' % json.dumps('.L '+str(macro))
        code += 'auto result=gROOT->ProcessLine(%s,&err);gSystem->Exit(err?92:int(result));' % json.dumps(expression)
        return self.run_root(code)

    def test_real_root_roundtrip_flags_duplicates_escaping_and_no_overwrite(self):
        before=hashlib.sha256(self.input.read_bytes()).hexdigest()
        output=self.folder/'parameters.json'
        done=self.export(self.input,output)
        self.assertEqual(done.returncode,0,done.stdout[-8000:])
        document=json.loads(output.read_text())
        self.assertEqual(document['schema'],'tpcdrum-persisted-parameters-v1')
        self.assertEqual(document['input'],str(self.input))
        self.assertTrue(document['uuid'])
        self.assertEqual({x['key'] for x in document['containers']},{'RunHeader','ParameterContainer'})
        rows=next(x['parameters'] for x in document['containers'] if x['key']=='ParameterContainer')
        self.assertEqual(len(rows),8)
        pulse=[x for x in rows if x['name']=='TPCDrum/PulseShapeData']
        self.assertEqual([x['value'] for x in pulse],['active.root','commented.root'])
        self.assertEqual(pulse[1]['type_flags'],513)
        self.assertTrue(pulse[1]['is_comment_out'])
        self.assertFalse(pulse[1]['is_line_comment'])
        line=next(x for x in rows if x['name']=='TPCDrum/FieldData')
        self.assertEqual(line['type_flags'],257)
        self.assertTrue(line['is_line_comment'])
        self.assertFalse(line['is_comment_out'])
        self.assertEqual(next(x for x in rows if x['name']=='TPCDrum/TimeBucketUnit')['unit'],'ns')
        quoted=next(x for x in rows if x['name']=='quoted')
        self.assertEqual(quoted['raw'],'raw\\path"quote\nnext\tcolumn')
        self.assertEqual(quoted['value'],'value\\path"quote\nnext\tcolumn')
        key,settings,duplicates=stored_parameter_values(document)
        self.assertEqual(key,'ParameterContainer')
        self.assertEqual(settings['TPCDrum/PulseShapeData'],'active.root')
        self.assertEqual(settings['TPCDrum/SimDataPath'],'/actual/common')
        self.assertNotIn('TPCDrum/FieldData',settings)
        self.assertEqual(duplicates['branch/input'],['GETChannel RawPad','LKMCTrack MCTrack'])
        output_before=output.read_bytes()
        again=self.export(self.input,output)
        self.assertNotEqual(again.returncode,0)
        self.assertEqual(output.read_bytes(),output_before)
        self.assertEqual(hashlib.sha256(self.input.read_bytes()).hexdigest(),before)

    def test_export_does_not_deserialize_damaged_event_tree_payload(self):
        source=self.folder/'damaged_event.root'
        shutil.copy2(self.input,source)
        key=json.loads(self.key_info.read_text())
        self.assertGreater(key['object_bytes'],key['payload_bytes'])
        self.assertGreater(key['payload_bytes'],16)
        # Damage only the event TTree compressed payload, retaining its readable
        # TKey header and the independent, valid parameter-container keys.
        with source.open('r+b') as stream:
            stream.seek(key['payload_offset'])
            stream.write(b'BAD-NOT-ZLIB!!!!')
        before=hashlib.sha256(source.read_bytes()).hexdigest()
        # Positive control: trying to deserialize the damaged event emits an
        # actual ROOT decompression error. This is separate from exporter run.
        probe=self.run_root('TFile f(%s,"READ");f.Get("event");' % json.dumps(str(source)))
        self.assertRegex(probe.stdout.lower(),r'error|unzip|decompress')
        output=self.folder/'damaged_event_parameters.json'
        done=self.export(source,output)
        self.assertEqual(done.returncode,0,done.stdout[-8000:])
        self.assertNotRegex(done.stdout.lower(),r'error in <.*(?:unzip|readbuffer)')
        document=json.loads(output.read_text())
        _,settings,_=stored_parameter_values(document)
        self.assertEqual(settings['TPCDrum/PulseShapeData'],'active.root')
        self.assertEqual(hashlib.sha256(source.read_bytes()).hexdigest(),before)


if __name__=='__main__':
    unittest.main(verbosity=2)
