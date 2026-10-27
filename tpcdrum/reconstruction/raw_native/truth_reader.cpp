// Evaluation-only native reader. Never used by observed inference stages.

#include "TFile.h"
#include "TTree.h"
#include "TTreeFormula.h"
#include "TSystem.h"
#include "TInterpreter.h"
#include "TClonesArray.h"
#include "TClass.h"
#include "TRealData.h"
#include "TStreamerInfo.h"
#include "TStreamerElement.h"
#include "TDataMember.h"
#include "TVirtualCollectionProxy.h"
#include "TROOT.h"
#include <iostream>
#include <string>
#include <vector>
#include <memory>
#include <limits>
#include <cmath>
#include <cstdint>
#include <stdexcept>
int main(int argc, char** argv) {
  try {
    if(argc!=2) throw std::runtime_error("Usage: truth_reader input.root");
    static_assert(sizeof(float)==4 && sizeof(double)==8, "Unsupported numeric format");
    gSystem->Load("libTreePlayer");
    gInterpreter->SetClassAutoLoading(false);
    TFile f(argv[1], "READ");
    auto t=dynamic_cast<TTree*>(f.Get("event"));
    if(f.IsZombie() || !t) throw std::runtime_error("Cannot open event tree");
    const Long64_t total=t->GetEntries();
    auto formula=[&](const char* name, const std::string& expression, TTree* tree) {
      auto q=std::unique_ptr<TTreeFormula>(new TTreeFormula(name,expression.c_str(),tree));
      if(q->GetNdim()<=0) throw std::runtime_error("Invalid formula: "+expression);
      return q;
    };

    std::string raw="RawPad", truth="MCTrack", tag="MCTagTPCDrum";
    TClonesArray* tags=nullptr;
    if(!t->GetBranch(tag.c_str())) throw std::runtime_error("Missing MCTag branch");
    if(t->SetBranchAddress(tag.c_str(),&tags)<0) throw std::runtime_error("Cannot bind tag array");
    // ROOT's on-file class metadata supplies offsets and STL collection proxies.
    // No guessed object layout and no LILAK methods/dictionaries are used.
    auto member=[&](TObject* object,const char* name) {
      TClass* cls=tags->GetClass();
      auto elem=dynamic_cast<TStreamerElement*>(cls->GetStreamerInfo()->GetElements()->FindObject(name));
      if(!elem) throw std::runtime_error(std::string("Missing tag member ")+name);
      auto vc=TClass::GetClass(elem->GetTypeName());
      if(!vc || !vc->GetCollectionProxy()) throw std::runtime_error("Missing vector proxy");
      return std::make_pair(vc->GetCollectionProxy(),reinterpret_cast<char*>(object)+elem->GetOffset());
    };
    auto ints=[&](TObject* object,const char* name) {
      auto m=member(object,name);TVirtualCollectionProxy::TPushPop guard(m.first,m.second);
      if(m.first->GetType()!=kInt_t) throw std::runtime_error("Tag integer type mismatch");
      std::vector<int> out(m.first->Size());
      for(unsigned int i=0;i<out.size();i++) out[i]=*static_cast<int*>(m.first->At(i));
      return out;
    };
    auto doubles=[&](TObject* object,const char* name) {
      auto m=member(object,name);TVirtualCollectionProxy::TPushPop guard(m.first,m.second);
      if(m.first->GetType()!=kDouble_t) throw std::runtime_error("Tag double type mismatch");
      std::vector<double> out(m.first->Size());
      for(unsigned int i=0;i<out.size();i++) out[i]=*static_cast<double*>(m.first->At(i));
      return out;
    };

    t->SetBranchStatus("*",0);t->SetBranchStatus((raw+"*").c_str(),1);
    auto pad=formula("cn_pad",raw+".fPadID",t);
    auto adc=formula("cn_adc",raw+".fBufferRawSig.fArray",t);
    std::unique_ptr<TTreeFormula> hw[3];
    std::vector<std::unique_ptr<TTreeFormula>> tf;
    std::cout<<"CNNREADY "<<total<<" "<<gROOT->GetVersion()<<"\n"<<std::flush;
    std::string command;
    while(std::cin>>command) {
      if(command=="Q") break;
      if(command=="M") {
        if(!t->GetBranch("HoyleSourceEntry") || !t->GetBranch("HoyleInputEntry"))
          throw std::runtime_error("Missing source/input provenance branches");
        t->SetBranchStatus("*",0);t->SetBranchStatus("HoyleSourceEntry",1);t->SetBranchStatus("HoyleInputEntry",1);
        auto source=formula("meta_src","HoyleSourceEntry",t), input=formula("meta_in","HoyleInputEntry",t);
        std::vector<std::int64_t> values(total*2);
        for(Long64_t e=0;e<total;++e){
          t->LoadTree(e);if(t->GetEntry(e)<=0)throw std::runtime_error("Provenance read failed");
          if(source->GetNdata()!=1 || input->GetNdata()!=1)throw std::runtime_error("Non-scalar provenance");
          values[e*2]=input->EvalInstance64();values[e*2+1]=source->EvalInstance64();
          if(values[e*2]<0 || values[e*2+1]<0)throw std::runtime_error("Invalid negative provenance");
        }
        std::cout<<"CNNM "<<total<<"\n";
        std::cout.write((const char*)values.data(),values.size()*8);std::cout.flush();
      } else if(command=="E") {
        Long64_t e;std::cin>>e;
        if(e<0 || e>=total)throw std::runtime_error("Invalid truth entry");
        t->SetBranchStatus("*",0);t->SetBranchStatus((truth+"*").c_str(),1);
        const char* fields[]={"fTrackID","fVX[][0]","fVY[][0]","fVZ[][0]","fPX[][0]","fPY[][0]","fPZ[][0]","fEnergy[][0]"};
        auto pdg=formula("one_pdg",truth+".fPDG",t);
        std::vector<std::unique_ptr<TTreeFormula>> fs;
        for(int j=0;j<8;++j)fs.push_back(formula(("one_"+std::to_string(j)).c_str(),truth+"."+fields[j],t));
        t->LoadTree(e);if(t->GetEntry(e)<=0)throw std::runtime_error("Truth read failed");
        int nt=pdg->GetNdata();for(auto& q:fs)if(q->GetNdata()!=nt)throw std::runtime_error("Track alignment failure");
        std::vector<double> v;
        for(int j=0;j<nt;++j)if(std::abs((int)pdg->EvalInstance(j))==1000020040)
          for(auto& q:fs)v.push_back(q->EvalInstance(j));
        std::cout<<"CNNE "<<e<<" "<<v.size()/8<<"\n";
        std::cout.write((const char*)v.data(),v.size()*8);std::cout.flush();
      } else if(command=="L") {
        Long64_t e;std::cin>>e;
        if(e<0||e>=total)throw std::runtime_error("Invalid label entry");
        t->SetBranchStatus("*",0);t->SetBranchStatus((tag+"*").c_str(),1);
        t->LoadTree(e);if(t->GetEntry(e)<=0 || !tags || tags->GetEntriesFast()!=768)
          throw std::runtime_error("Expected 768 TPC tags");
        std::vector<std::int32_t> vox,tid;std::vector<float> weights;
        for(int p=0;p<768;p++) {
          auto obj=tags->At(p);if(!obj)throw std::runtime_error("Null tag object");
          auto index=ints(obj,"fIndex"), ids=ints(obj,"fMCID");auto w=doubles(obj,"fWeight");
          if(index.size()!=ids.size()||index.size()!=w.size())throw std::runtime_error("Tag parallel array length mismatch");
          for(unsigned int j=0;j<index.size();j++) {
            if(!std::isfinite(w[j])||w[j]<0)throw std::runtime_error("Invalid stored tag weight");
            if(w[j]==0)continue;
            if(index[j]<0||index[j]>=512)throw std::runtime_error("Non-TB tag index");
            // Uploaded digitizer stores MCTagTPCDrum array index == physical padID.
            vox.push_back(p*512+index[j]);tid.push_back(ids[j]);weights.push_back((float)w[j]);
          }
        }
        std::cout<<"CNNL "<<e<<" "<<weights.size()<<"\n";
        std::cout.write((const char*)vox.data(),vox.size()*4);
        std::cout.write((const char*)tid.data(),tid.size()*4);
        std::cout.write((const char*)weights.data(),weights.size()*4);std::cout.flush();
      } else if(command=="G") {
        auto meta=dynamic_cast<TTree*>(f.Get("HoyleMeta"));
        std::vector<std::int64_t> groups(total);
        if(meta) {
          if(meta->GetEntries()!=total || !meta->GetBranch("HoyleSourceEntry"))
            throw std::runtime_error("HoyleMeta not entry-aligned or missing HoyleSourceEntry");
          meta->SetBranchStatus("*",0);meta->SetBranchStatus("HoyleSourceEntry",1);
          auto group=formula("cn_group","HoyleSourceEntry",meta);
          for(Long64_t e=0;e<total;e++) {
            meta->LoadTree(e);meta->GetEntry(e);
            if(group->GetNdata()!=1) throw std::runtime_error("Non-scalar HoyleSourceEntry");
            groups[e]=(std::int64_t)group->EvalInstance64(0);
          }
        } else for(Long64_t e=0;e<total;e++) groups[e]=e;
        std::cout<<"CNNG "<<total<<" "<<(meta?1:0)<<"\n";
        std::cout.write((const char*)groups.data(),groups.size()*sizeof(std::int64_t));std::cout.flush();
      } else throw std::runtime_error("Unknown request: "+command);
    }
    return 0;
  } catch(const std::exception& e) {
    std::cout<<"CNNERROR "<<e.what()<<"\n"<<std::flush;return 2;
  }
}
