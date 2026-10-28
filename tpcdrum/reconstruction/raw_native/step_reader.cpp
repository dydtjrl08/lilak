// Calibration/evaluation truth ONLY. This executable never reads RawPad/MCTag.
#include "TFile.h"
#include "TTree.h"
#include "TTreeFormula.h"
#include "TROOT.h"
#include "TSystem.h"
#include "TInterpreter.h"
#include <cmath>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

int main(int argc,char** argv) {
 try {
  if(argc!=2)throw std::runtime_error("Usage: step_reader input.root");
  static_assert(sizeof(double)==8,"Unsupported numeric representation");
  gSystem->Load("libTreePlayer");gInterpreter->SetClassAutoLoading(false);
  TFile file(argv[1],"READ");auto tree=dynamic_cast<TTree*>(file.Get("event"));
  if(file.IsZombie()||!tree)throw std::runtime_error("Missing input/event tree");
  if(file.TestBit(TFile::kRecovered))throw std::runtime_error("Recovered/unfinalized input rejected");
  if(!tree->GetBranch("MCTrack")||!tree->GetBranch("MCStepTPCDrum"))throw std::runtime_error("Missing MCTrack or MCStepTPCDrum truth branch");
  tree->SetBranchStatus("*",0);tree->SetBranchStatus("MCTrack*",1);tree->SetBranchStatus("MCStepTPCDrum*",1);
  auto formula=[&](const char* name,const std::string& expression) {
   auto result=std::unique_ptr<TTreeFormula>(new TTreeFormula(name,expression.c_str(),tree));
   if(result->GetNdim()<=0)throw std::runtime_error("Invalid truth schema: "+expression);
   return result;
  };
  auto id=formula("ep_track_id","MCTrack.fTrackID"),pdg=formula("ep_pdg","MCTrack.fPDG");
  const char* fields[]={"fTrackID","fX","fY","fZ","fEdep","fTime"};
  std::vector<std::unique_ptr<TTreeFormula>> steps;
  std::vector<std::unique_ptr<TTreeFormula>> initial;
  for(int j=0;j<6;j++)steps.push_back(formula(("ep_step_"+std::to_string(j)).c_str(),std::string("MCStepTPCDrum.")+fields[j]));
  const auto total=tree->GetEntries();
  std::cout<<"EPREADY "<<total<<" "<<gROOT->GetVersion()<<"\n"<<std::flush;
  std::string command;
  while(std::cin>>command) {
   if(command=="Q")break;
   if(command!="S"&&command!="T")throw std::runtime_error("This truth helper accepts only MC steps/initial tracks, never waveform requests");
   Long64_t entry=-1,requested=-1;
   if(!(std::cin>>entry>>requested)||entry<0||entry>=total||requested==0||requested < -1)
    throw std::runtime_error("Invalid selected truth entry/track ID");
   tree->SetBranchStatus("MCStepTPCDrum*",command=="S");
   if(command=="T"&&initial.empty()) {
    const char* names[]={"fVX[][0]","fVY[][0]","fVZ[][0]","fPX[][0]","fPY[][0]","fPZ[][0]","fEnergy[][0]"};
    for(int j=0;j<7;j++)initial.push_back(formula(("ep_initial_"+std::to_string(j)).c_str(),std::string("MCTrack.")+names[j]));
   }
   tree->LoadTree(entry);if(tree->GetEntry(entry)<=0)throw std::runtime_error("Truth entry read failed");
   int nt=id->GetNdata();if(pdg->GetNdata()!=nt)throw std::runtime_error("MCTrack member alignment failure");
   std::vector<Long64_t> alpha_ids;
   for(int i=0;i<nt;i++) {
    const double a=id->EvalInstance(i),b=pdg->EvalInstance(i);
    if(!std::isfinite(a)||a<=0||std::floor(a)!=a||!std::isfinite(b)||std::floor(b)!=b)
     throw std::runtime_error("Invalid track identity/PDG");
    if(std::abs(b)==1000020040.)alpha_ids.push_back(static_cast<Long64_t>(a));
   }
   if(requested==-1) {
    if(alpha_ids.size()!=1)throw std::runtime_error("Auto alpha selection requires exactly one MC alpha track; supply explicit own track ID");
    requested=alpha_ids.front();
   }
   int matches=0,selected_index=-1;double selected_pdg=0;
   for(int i=0;i<nt;i++)if(id->EvalInstance(i)==requested){matches++;selected_pdg=pdg->EvalInstance(i);selected_index=i;}
   if(matches!=1)throw std::runtime_error(matches?"Duplicate selected MCTrack ID":"Selected MCTrack ID absent");
   if(std::abs(selected_pdg)!=1000020040.)throw std::runtime_error("Selected MC track is not an alpha");
   if(command=="T") {
    std::vector<double> row(1,static_cast<double>(requested));
    for(auto& field:initial) {
     if(field->GetNdata()!=nt)throw std::runtime_error("Initial track member alignment failure");
     const double value=field->EvalInstance(selected_index);
     if(!std::isfinite(value))throw std::runtime_error("Invalid initial track value");
     row.push_back(value);
    }
    if(row.back()<0)throw std::runtime_error("Negative initial kinetic energy");
    std::cout<<"EPTRACK "<<entry<<" "<<requested<<" 8\n";
    std::cout.write(reinterpret_cast<const char*>(row.data()),8*sizeof(double));std::cout.flush();
    continue;
   }
   const int ns=steps[0]->GetNdata();
   if(ns<0||ns>2000000)throw std::runtime_error("MC step count exceeds bounded event limit");
   for(auto& field:steps)if(field->GetNdata()!=ns)throw std::runtime_error("MCStep member alignment failure");
   std::vector<double> rows;
   for(int i=0;i<ns;i++) {
    const double own=steps[0]->EvalInstance(i);
    if(!std::isfinite(own)||own<=0||std::floor(own)!=own)throw std::runtime_error("Invalid MCStep track ID");
    if(own!=requested)continue;
    for(int j=0;j<6;j++) {
     const double value=steps[j]->EvalInstance(i);
     if(!std::isfinite(value)||(j==4&&value<0))throw std::runtime_error("Invalid selected MCStep value");
     rows.push_back(value);
    }
   }
   if(rows.empty())throw std::runtime_error("Selected alpha has no saved MCStepTPCDrum rows");
   if(rows.size()/6>1000000)throw std::runtime_error("Selected alpha steps exceed bounded output limit");
   // Exact on-file ordering and coordinate values: NO transform or sorting.
   std::cout<<"EPSTEP "<<entry<<" "<<requested<<" "<<rows.size()/6<<"\n";
   std::cout.write(reinterpret_cast<const char*>(rows.data()),rows.size()*sizeof(double));std::cout.flush();
  }
  return 0;
 } catch(const std::exception& error) {
  std::cout<<"CNNERROR "<<error.what()<<"\n"<<std::flush;return 2;
 }
}
