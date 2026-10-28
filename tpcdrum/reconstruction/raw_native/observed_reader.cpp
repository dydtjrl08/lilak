// Standalone observed-only ROOT reader. No MC, Si, labels, or provenance branch
// is required or enabled. Reads existing streamed RawPad schema without LILAK.
#include "TFile.h"
#include "TTree.h"
#include "TTreeFormula.h"
#include "TROOT.h"
#include "TSystem.h"
#include "TInterpreter.h"
#include <iostream>
#include <vector>
#include <memory>
#include <string>
#include <stdexcept>
#include <cmath>
#include <iomanip>

int main(int argc,char** argv) {
  try {
    if(argc!=2) throw std::runtime_error("Usage: observed_reader input.root");
    static_assert(sizeof(float)==4,"Unsupported float size");
    gSystem->Load("libTreePlayer");
    gInterpreter->SetClassAutoLoading(false);
    TFile file(argv[1],"READ");
    auto tree=dynamic_cast<TTree*>(file.Get("event"));
    if(file.IsZombie() || !tree) throw std::runtime_error("Missing input/event tree");
    if(file.TestBit(TFile::kRecovered)) throw std::runtime_error("Recovered/unfinalized ROOT input: finish digitization first");
    tree->SetBranchStatus("*",0);
    tree->SetBranchStatus("RawPad*",1);
    auto formula=[&](const char* name,const std::string& expression) {
      auto q=std::unique_ptr<TTreeFormula>(new TTreeFormula(name,expression.c_str(),tree));
      if(q->GetNdim()<=0) throw std::runtime_error("Invalid observed branch: "+expression);
      return q;
    };
    auto pad=formula("obs_pad","RawPad.fPadID");
    auto adc=formula("obs_adc","RawPad.fBufferRawSig.fArray");
    std::unique_ptr<TTreeFormula> hw[3];
    auto total=tree->GetEntries();
    std::cout<<"RRREADY "<<total<<" "<<gROOT->GetVersion()<<"\n"<<std::flush;
    std::string command;
    while(std::cin>>command) {
      if(command=="Q") break;
      if(command!="W") throw std::runtime_error("Unknown reader command");
      Long64_t entry=-1;
      if(!(std::cin>>entry) || entry<0 || entry>=total) throw std::runtime_error("Invalid entry");
      tree->LoadTree(entry);
      if(tree->GetEntry(entry)<=0) throw std::runtime_error("Observed waveform read failed");
      int n=pad->GetNdata(),nw=adc->GetNdata();
      if((n!=768 && n!=1024) || nw!=n*512) throw std::runtime_error("Expected 768/1024 RawPad objects with 512 TB each");
      std::vector<float> values(768*512);
      std::vector<bool> seen(768,false);
      bool hardwareReady=false;
      // Existing digitizer saves the 768 TPC objects first; any trailing 256
      // objects are Si channels. Duplicate/missing TPC IDs fail explicitly.
      for(int p=0;p<768;p++) {
        double storedID=pad->EvalInstance(p);
        if(!std::isfinite(storedID) || std::floor(storedID)!=storedID)
          throw std::runtime_error("Noninteger/nonfinite pad ID");
        int id=static_cast<int>(storedID);
        if(id<0 || id>=768) {
          const char* names[]={"fAsad","fAget","fChan"};
          for(int k=0;k<3;k++) if(!hw[k]) hw[k]=formula(("obs_hw_"+std::to_string(k)).c_str(),std::string("RawPad.")+names[k]);
          if(!hardwareReady) {
            for(int k=0;k<3;k++) if(hw[k]->GetNdata()!=n) throw std::runtime_error("Hardware field length mismatch");
            hardwareReady=true;
          }
          int asad=hw[0]->EvalInstance(p),aget=hw[1]->EvalInstance(p),chan=hw[2]->EvalInstance(p);
          if(asad<0 || asad>=3 || aget<0 || aget>=4 || chan<0 || chan>=68 || chan==11 || chan==22 || chan==45 || chan==56)
            throw std::runtime_error("Invalid TPC hardware address");
          int connector=12-(asad*4+aget),baseRow=connector%4==0?0:(4-connector%4)*16;
          int baseLayer=4*std::abs((connector-1)/4-2),pin=chan+1-(chan>11)-(chan>22)-(chan>45)-(chan>56);
          int row=baseRow+15-(pin-pin%4)/4+(pin%4==0?1:0),offset[]={2,0,3,1};
          id=(baseLayer+offset[pin%4])*64+row;
        }
        if(id<0 || id>=768 || seen[id]) throw std::runtime_error("Duplicate/invalid TPC pad ID");
        seen[id]=true;
        for(int tb=0;tb<512;tb++) {
          double v=adc->EvalInstance(p*512+tb);
          if(!std::isfinite(v) || std::abs(v)>3.4e38) throw std::runtime_error("Nonfinite/overflow ADC");
          values[id*512+tb]=static_cast<float>(v);
        }
      }
      std::cout<<"RRW "<<entry<<" "<<values.size()<<"\n";
      std::cout.write(reinterpret_cast<const char*>(values.data()),values.size()*sizeof(float));
      std::cout.flush();
    }
    return 0;
  } catch(const std::exception& e) {
    std::cout<<"RRERROR "<<e.what()<<"\n"<<std::flush;
    return 2;
  }
}
