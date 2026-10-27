// Standalone analysis of ORIGINAL transport MC. No digitization or task changes.
// No #include directives; load the existing LILAK dictionaries via rootlogon.
// Run interpreted (no +/++):
// root -l -b -q ../../macros/rootlogon.C \
//   'hoyle_si_vertex_20261027/C12_study_hoyle_si_by_vertex.C("C12_72MeV_DDE_test1000000.root")'
// N_alpha,Si counts UNIQUE selected decay alpha IDs, not hit detector modules.
// Primary result: stored-step arrival multiplicity, including N=0 decays.
// A second result reports own-alpha Si Edep > threshold (default 0.1 MeV).
// Output: unique ROOT + CSV + 2 SVG + two-page PDF in the current directory.
// Reference box/region labels are plotting definitions, not active-volume cuts.
// Adapted from the project's existing study_hoyle_si_by_vertex.C.

// Original MC only; run in a ROOT session with the LILAK dictionaries loaded.
// root -l 'C12_study_hoyle_si_by_vertex.C("C12_72MeV.root")'
// Batch: root -l -b -q 'C12_study_hoyle_si_by_vertex.C("C12_72MeV.root")'
// Arguments: input, Si deposited-energy threshold [MeV], maxEvents (-1=all).
// No existing files/tasks/classes are modified. Output names are unique.
//
// Direct 12C -> 3 alpha only, as in C12_72MeV.reaction. A carbon parent and
// exactly three direct alpha daughters with a common initial vertex must exist.
// The recoil alpha is NOT counted. Sequential 12C -> alpha+8Be is not selected.
// This identifies the configured Hoyle decay; it does not independently measure
// the parent's excitation energy from its mass.
//
// Two definitions, each with ALL selected decays as denominator:
//   arrival: alpha has at least one STORED step in any of the eight Si volumes.
//   edep:    that alpha's summed Si Edep > threshold (default 0.1 MeV).
// Steps of secondary electrons are not added. Neither is a digitized/trigger
// efficiency. If zero-Edep steps are not stored, arrival is only a proxy for
// geometric entry, not an exact count of all volume-boundary crossings.
//
// Coordinates follow the old vtx_sim.C / TPCDrum convention:
//   u=-X_G4, v=Z_G4+66, w=Y_G4+128.8 [mm]. Nominal pad v=[-6,138), Si centre plane v=199.
// Do NOT mix the old log plot's +72 offset with these boundaries.
// No TPC activity, alpha energy, angle, or transverse vertex cut is applied.


namespace HoyleSiVertex20261027 {
// Change only these constants to change the displayed vertex bins.
const int N_BINS = 25;
const double V_MIN = -290., V_MAX = 210.; // 20 mm per bin; flow bins retained
const double V_OFFSET = 66., W_OFFSET = 128.8;
const double PAD_START = -6., PAD_END = 138., SI_PLANE = 199.;
const double VERTEX_TOLERANCE = 1.e-3; // mm, same as STDecaySubsetTask
const double CONFIDENCE = 0.682689492137;
const char* const SI_BRANCHES[8] = {
    "MCStepSiDet_RU", "MCStepSiDet_RM", "MCStepSiDet_RD", "MCStepSiDet_LU",
    "MCStepSiDet_LM", "MCStepSiDet_LD", "MCStepSiDet_U", "MCStepSiDet_D"};
const char* const METRIC[2] = {"arrival", "edep"};
const char* const REGIONS[4] = {"before_pad", "pad_v_interval", "pad_to_Si", "at_or_after_Si"};

void Require(bool ok, const std::string& message) {
    if (!ok) throw std::runtime_error("C12_study_hoyle_si_by_vertex: " + message);
}
bool Finite(const TVector3& p) {
    return std::isfinite(p.X()) && std::isfinite(p.Y()) && std::isfinite(p.Z());
}
struct Decay {
    int parent;
    TVector3 vertex;
    std::array<int,3> ids;
};
std::vector<Decay> FindDecays(TClonesArray* tracks, bool& malformed) {
    std::map<int,LKMCTrack*> byID;
    std::map<int,std::vector<LKMCTrack*>> daughters;
    malformed = false;
    for (int i=0; i<tracks->GetEntriesFast(); ++i) {
        auto* t = dynamic_cast<LKMCTrack*>(tracks->At(i));
        if (!t || t->GetTrackID()<=0 || t->GetNumVertices()<1 ||
            !Finite(t->GetPrimaryPosition()) || !byID.emplace(t->GetTrackID(),t).second) {
            malformed = true;
            return {};
        }
        if (t->GetPDG()==1000020040) daughters[t->GetParentID()].push_back(t);
    }
    std::vector<Decay> result;
    for (auto& item : daughters) {
        const auto p = byID.find(item.first);
        // Accept the excitation/isomer digit, but require an actual carbon parent.
        if (p==byID.end() || p->second->GetPDG()/10!=100006012 || item.second.size()!=3)
            continue;
        auto& a = item.second;
        bool shared = true;
        for (int i=0; i<3; ++i) for (int j=i+1; j<3; ++j)
            shared &= (a[i]->GetPrimaryPosition()-a[j]->GetPrimaryPosition()).Mag()<=VERTEX_TOLERANCE;
        if (!shared) continue;
        std::sort(a.begin(),a.end(),[](LKMCTrack* x,LKMCTrack* y) {return x->GetTrackID()<y->GetTrackID();});
        // Use the daughters' production vertex, not the carbon's beam-entry vertex.
        result.push_back({item.first,a[0]->GetPrimaryPosition(),
                          {{a[0]->GetTrackID(),a[1]->GetTrackID(),a[2]->GetTrackID()}}});
    }
    return result;
}
struct Counts {
    Long64_t n[4] = {0,0,0,0};
    Long64_t N() const {return n[0]+n[1]+n[2]+n[3];}
    Long64_t AtLeast(int k) const {Long64_t s=0; for (int j=k;j<4;++j) s+=n[j]; return s;}
};
int Region(double v) {return v<PAD_START ? 0 : v<PAD_END ? 1 : v<SI_PLANE ? 2 : 3;}

// Event binomial probabilities: each original event is one trial.
void Point(TGraphAsymmErrors* g, double x, double dx, Long64_t pass, Long64_t total) {
    if (!total) return; // empty bins are missing, never fake zero-efficiency points
    const double p = double(pass)/total;
    const double lo = TEfficiency::ClopperPearson(total,pass,CONFIDENCE,false);
    const double hi = TEfficiency::ClopperPearson(total,pass,CONFIDENCE,true);
    const int i = g->GetN();
    g->SetPoint(i,x,p);
    g->SetPointError(i,dx,dx,p-lo,hi-p);
}
void Boundaries(double ymax) {
    for (double x : {PAD_START,PAD_END,SI_PLANE}) {
        auto* line = new TLine(x,0.,x,ymax);
        line->SetLineStyle(2); line->SetLineColor(kGray+2); line->Draw();
    }
}
void CsvRow(std::ofstream& csv, const char* metric, const char* scope,
            const std::string& label, double lo, double hi, const Counts& c) {
    const double nan = std::numeric_limits<double>::quiet_NaN();
    const Long64_t n = c.N();
    csv << metric << ',' << scope << ',' << label << ',' << lo << ',' << hi << ',' << n;
    for (int k=0;k<4;++k) csv << ',' << c.n[k];
    for (int k=0;k<4;++k) {
        csv << ',' << (n ? double(c.n[k])/n : nan)
            << ',' << (n ? TEfficiency::ClopperPearson(n,c.n[k],CONFIDENCE,false) : nan)
            << ',' << (n ? TEfficiency::ClopperPearson(n,c.n[k],CONFIDENCE,true) : nan);
    }
    for (int k=1;k<=3;++k) csv << ',' << (n ? double(c.AtLeast(k))/n : nan);
    double mean=0., square=0.;
    for (int k=0;k<4;++k) {mean+=k*c.n[k]; square+=k*k*c.n[k];}
    mean = n ? mean/n : nan;
    // Three alphas in the same event are correlated: SEM uses EVENTS, not 3*N trials.
    const double sem = n>1 ? std::sqrt(std::max(0.,square-n*mean*mean)/(n-1)/n)/3. : nan;
    csv << ',' << mean << ',' << mean/3. << ',' << sem << '\n';
}

TCanvas* Draw(const char* metric, const std::vector<Counts>& bins, const Counts* regions,
              TH1D* hVertex, TFile* output, const TString& unique, double threshold) {
    output->cd();
    const int colors[4] = {kGray+1,kAzure+1,kOrange+7,kGreen+2};
    const double width = (V_MAX-V_MIN)/N_BINS;
    auto* c = new TCanvas("c_"+TString(metric)+"_"+unique,
        Form("Hoyle decay alpha Si %s (Edep threshold %.3g MeV)",metric,threshold),1300,850);
    c->Divide(2,2);
    c->cd(1);
    auto* stack = new THStack("stack_"+TString(metric)+"_"+unique,
        Form("Selected direct decays: %s;v_{pad} [mm];Events",metric));
    auto* legend = new TLegend(.67,.67,.89,.89);
    for (int k=0;k<4;++k) {
        auto* h = new TH1D(Form("h_%s_n%d_%s",metric,k,unique.Data()),"",N_BINS,V_MIN,V_MAX);
        h->SetDirectory(nullptr); h->SetFillColor(colors[k]); h->SetLineColor(colors[k]);
        for (int b=0;b<=N_BINS+1;++b) h->SetBinContent(b,bins[b].n[k]);
        stack->Add(h); legend->AddEntry(h,Form("%d decay #alpha in Si",k),"f");
        h->Write(Form("h_%s_n%d_vs_v",metric,k));
    }
    stack->Draw("HIST"); legend->Draw();
    Boundaries(std::max(1.,hVertex->GetMaximum()*1.05));

    for (int panel=0;panel<2;++panel) {
        c->cd(panel+2);
        auto* frame = gPad->DrawFrame(V_MIN,0.,V_MAX,1.05,
            panel==0 ? "Exact multiplicity;v_{pad} [mm];P(N_{#alpha,Si}=k | decay in bin)"
                     : "Event acceptance;v_{pad} [mm];P(N_{#alpha,Si} #geq k | decay in bin)");
        frame->GetYaxis()->SetTitleSize(.035);
        Boundaries(1.05);
        auto* lg = new TLegend(.65,.68,.89,.89);
        for (int k=(panel ? 1:0);k<4;++k) {
            auto* g = new TGraphAsymmErrors;
            g->SetName(Form("g_%s_%s%d_vs_v",metric,panel?"ge":"eq",k));
            g->SetLineColor(colors[k]); g->SetMarkerColor(colors[k]); g->SetMarkerStyle(20+k);
            for (int b=1;b<=N_BINS;++b)
                Point(g,V_MIN+(b-.5)*width,width/2.,panel?bins[b].AtLeast(k):bins[b].n[k],bins[b].N());
            g->Draw("P SAME");
            lg->AddEntry(g,Form(panel?"N #geq %d":"N = %d",k),"lp");
            g->Write();
        }
        lg->Draw();
    }
    c->cd(4);
    auto* frame = new TH1D("regions_frame_"+TString(metric)+"_"+unique,
        "Reference regions (beam-axis only);Region;P(N_{#alpha,Si}=k | decay in region)",4,-.5,3.5);
    frame->SetDirectory(nullptr); frame->SetMinimum(0.); frame->SetMaximum(1.05);
    const char* labels[4]={"Before pad","Pad interval","Pad to Si","After Si"};
    for (int r=0;r<4;++r) frame->GetXaxis()->SetBinLabel(r+1,labels[r]);
    frame->GetXaxis()->SetLabelSize(.04); frame->GetYaxis()->SetTitleSize(.035);
    frame->Draw("AXIS");
    auto* lg = new TLegend(.71,.68,.89,.89);
    for (int k=0;k<4;++k) {
        auto* g = new TGraphAsymmErrors;
        g->SetName(Form("g_%s_eq%d_regions",metric,k));
        g->SetMarkerColor(colors[k]); g->SetLineColor(colors[k]); g->SetMarkerStyle(20+k);
        for (int r=0;r<4;++r) Point(g,r+(k-1.5)*.15,0.,regions[r].n[k],regions[r].N());
        g->Draw("P SAME"); lg->AddEntry(g,Form("N = %d",k),"lp"); g->Write();
    }
    lg->Draw(); c->Modified(); c->Update();
    c->Write("canvas_"+TString(metric));
    return c;
}
} // namespace HoyleSiVertex20261027

void C12_study_hoyle_si_by_vertex(const char* input="C12_72MeV.root",
                              double siThresholdMeV=0.1, Long64_t maxEvents=-1)
{
    using namespace HoyleSiVertex20261027;
    Require(input && *input,"input filename is empty");
    Require(std::isfinite(siThresholdMeV) && siThresholdMeV>=0.,"invalid Si threshold");
    Require(maxEvents==-1 || maxEvents>0,"maxEvents must be -1 or positive");
    TString inputPath(input); gSystem->ExpandPathName(inputPath);
    if (!gSystem->IsAbsoluteFileName(inputPath))
        gSystem->PrependPathName(gSystem->WorkingDirectory(),inputPath);
    std::unique_ptr<TFile> source(TFile::Open(inputPath,"READ"));
    Require(source && !source->IsZombie(),std::string("cannot open ")+inputPath.Data());
    Require(!source->TestBit(TFile::kRecovered),"original MC was recovered; confirm simulation completed first");
    auto* tree = dynamic_cast<TTree*>(source->Get("event"));
    Require(tree,"missing event tree");
    Require(!tree->GetBranch("RawPad"),"use original transport MC before digitization, not RawPad data");
    Require(!source->Get("HoyleMeta"),"use original full MC, not a 3C1/3C2/3C3 subset");
    TClonesArray* tracks=nullptr;
    std::array<TClonesArray*,8> si{{nullptr,nullptr,nullptr,nullptr,nullptr,nullptr,nullptr,nullptr}};
    tree->SetBranchStatus("*",0);
    Require(tree->GetBranch("MCTrack"),"missing MCTrack");
    tree->SetBranchStatus("MCTrack*",1);
    Require(tree->SetBranchAddress("MCTrack",&tracks)>=0,"cannot bind MCTrack");
    for (int d=0;d<8;++d) {
        // Missing detectors must NOT silently look like zero efficiency.
        Require(tree->GetBranch(SI_BRANCHES[d]),std::string("missing Si branch: ")+SI_BRANCHES[d]);
        tree->SetBranchStatus(TString(SI_BRANCHES[d])+"*",1);
        Require(tree->SetBranchAddress(SI_BRANCHES[d],&si[d])>=0,std::string("cannot bind ")+SI_BRANCHES[d]);
    }
    const Long64_t n = maxEvents<0 ? tree->GetEntries() : std::min(maxEvents,tree->GetEntries());
    Require(n>0,"empty input");

    TUUID uuid; TString unique=uuid.AsString(); unique.ReplaceAll("-","");
    TString stem=gSystem->BaseName(inputPath); if (stem.EndsWith(".root")) stem.Resize(stem.Length()-5);
    const TString prefix=stem+"_hoyle_si_vtx_"+unique;
    std::unique_ptr<TFile> output(TFile::Open(prefix+".root","CREATE"));
    Require(output && !output->IsZombie(),"cannot create output ROOT file");
    std::ofstream csv((prefix+".csv").Data()); Require(bool(csv),"cannot create CSV");
    csv << std::setprecision(12);
    csv << "metric,scope,label,v_low_mm,v_high_mm,N_decay,N0,N1,N2,N3";
    for (int k=0;k<4;++k) csv << ",P"<<k<<",P"<<k<<"_lo68,P"<<k<<"_hi68";
    csv << ",P_ge1,P_ge2,P_ge3,mean_n_alpha,alpha_fraction,alpha_fraction_event_SEM\n";

    output->cd();
    TNamed("input_file",inputPath.Data()).Write();
    TNamed("input_uuid",source->GetUUID().AsString()).Write();
    TNamed("analysis_schema","c12-direct-si-multiplicity-vs-vertex-v1").Write();
    TNamed("reference_box_definition",
        "in_pad_reference_box is only a legacy plotting reference: u=[-36.95,36.95), "
        "v=[-6,138), w=[0,128.8) mm. It is NOT TPCDrum::IsInBoundary, "
        "is NOT an active-volume acceptance flag, and is NOT used to select events.").Write();
    TNamed("definitions",
        "Direct C12->3alpha genealogy + common daughter vertex within 0.001 mm; recoil excluded. "
        "status: 0=no identified direct triplet (not proof of no physical decay), 1=selected, "
        "2=multiple triplets, 3=malformed MCTrack. Denominator: status=1, INCLUDING nSi=0. "
        "arrival=own stored Si step; edep=sum of own-track Si Edep strictly above threshold. "
        "No TPC or transverse cuts. Regions use v only, half-open intervals. "
        "No cross-section reweighting. Individual event-probability intervals: 68.27% Clopper-Pearson. "
        "alpha_fraction SEM treats events as independent, not the three siblings.").Write();
    TNamed("coordinates","u=-X_G4; v=Z_G4+66; w=Y_G4+128.8 mm; pad v=[-6,138), Si v=199").Write();
    TParameter<double>("si_threshold_MeV",siThresholdMeV).Write();
    TParameter<double>("vertex_tolerance_mm",VERTEX_TOLERANCE).Write();
    TParameter<int>("analysis_complete",0).Write();

    Long64_t entry=0, statusCounts[4]={0,0,0,0};
    Int_t status=0,parentID=0,alphaID[3]={0,0,0},detectorMask[3]={0,0,0};
    Int_t nArrival=-1,nEdep=-1,region=-1;
    Double_t u=0.,v=0.,w=0.,siEdep[3]={0.,0.,0.};
    Bool_t inPadReferenceBox=false;
    auto* audit = new TTree("event_audit","One row per inspected original MC event");
    audit->Branch("entry",&entry,"entry/L"); audit->Branch("status",&status,"status/I");
    audit->Branch("parent_id",&parentID,"parent_id/I"); audit->Branch("alpha_id",alphaID,"alpha_id[3]/I");
    audit->Branch("u_pad_mm",&u,"u_pad_mm/D"); audit->Branch("v_pad_mm",&v,"v_pad_mm/D");
    audit->Branch("w_pad_mm",&w,"w_pad_mm/D"); audit->Branch("region",&region,"region/I");
    audit->Branch("in_pad_reference_box",&inPadReferenceBox,"in_pad_reference_box/O");
    audit->Branch("si_detector_mask",detectorMask,"si_detector_mask[3]/I");
    audit->Branch("si_edep_MeV",siEdep,"si_edep_MeV[3]/D");
    audit->Branch("n_si_arrival",&nArrival,"n_si_arrival/I");
    audit->Branch("n_si_edep",&nEdep,"n_si_edep/I");
    gROOT->cd();
    auto* hVertex = new TH1D("h_vertex_"+unique,"Selected direct decay vertices;v_{pad} [mm];Events",N_BINS,V_MIN,V_MAX);
    hVertex->SetDirectory(nullptr);
    Counts total[2], regions[2][4];
    std::vector<Counts> bins[2] = {std::vector<Counts>(N_BINS+2),std::vector<Counts>(N_BINS+2)};
    const double nan=std::numeric_limits<double>::quiet_NaN();
    for (entry=0;entry<n;++entry) {
        Require(tree->GetEntry(entry)>0,"read failure at entry "+std::to_string(entry));
        Require(tracks,"null MCTrack array at entry "+std::to_string(entry));
        status=0; parentID=0; nArrival=nEdep=region=-1; u=v=w=nan; inPadReferenceBox=false;
        std::fill(alphaID,alphaID+3,0); std::fill(detectorMask,detectorMask+3,0);
        std::fill(siEdep,siEdep+3,0.);
        bool malformed=false;
        const auto decay=FindDecays(tracks,malformed);
        status=malformed ? 3 : decay.size()>1 ? 2 : decay.empty() ? 0 : 1;
        ++statusCounts[status];
        if (status==1) {
            parentID=decay[0].parent;
            std::copy(decay[0].ids.begin(),decay[0].ids.end(),alphaID);
            u=-decay[0].vertex.X(); v=decay[0].vertex.Z()+V_OFFSET; w=decay[0].vertex.Y()+W_OFFSET;
            region=Region(v);
            inPadReferenceBox=(u>=-36.95 && u<36.95 && region==1 && w>=0. && w<W_OFFSET);
            for (int d=0;d<8;++d) {
                Require(si[d],"null Si array at entry "+std::to_string(entry));
                for (int j=0;j<si[d]->GetEntriesFast();++j) {
                    auto* step=dynamic_cast<LKMCStep*>(si[d]->At(j));
                    Require(step,"invalid Si step object");
                    for (int a=0;a<3;++a) if (step->GetTrackID()==alphaID[a]) {
                        Require(std::isfinite(step->GetEdep()) && step->GetEdep()>=0.,"invalid Si Edep");
                        detectorMask[a] |= (1<<d);
                        siEdep[a] += step->GetEdep();
                        Require(std::isfinite(siEdep[a]),"Si Edep sum overflow");
                    }
                }
            }
            nArrival=nEdep=0;
            for (int a=0;a<3;++a) {nArrival+=(detectorMask[a]!=0); nEdep+=(siEdep[a]>siThresholdMeV);}
            hVertex->Fill(v);
            const int b=hVertex->FindFixBin(v);
            for (int m=0;m<2;++m) {
                const int count=m ? nEdep:nArrival;
                ++total[m].n[count]; ++regions[m][region].n[count]; ++bins[m][b].n[count];
            }
        }
        Require(audit->Fill()>=0,"cannot write audit row");
        if (entry%20000==0) printf("entry %lld/%lld; selected decays=%lld\n",entry,n,statusCounts[1]);
    }
    const double inf=std::numeric_limits<double>::infinity();
    const double edges[5]={-inf,PAD_START,PAD_END,SI_PLANE,inf};
    printf("\nInspected=%lld; selected direct decays=%lld; no identified triplet=%lld; ambiguous=%lld; malformed=%lld\n",
           n,statusCounts[1],statusCounts[0],statusCounts[2],statusCounts[3]);
    printf("v_pad=Z_G4+66 mm; pad=[-6,138); Si plane=199. No TPC/angle/transverse selection.\n");
    for (int m=0;m<2;++m) {
        printf("\n[%s] threshold for edep: > %.6g MeV\n",METRIC[m],siThresholdMeV);
        printf("region                 N_decay       N0       N1       N2       N3     P(>=1)    P(3)\n");
        for (int r=-1;r<4;++r) {
            const Counts& c=r<0 ? total[m]:regions[m][r];
            const char* label=r<0 ? "all":REGIONS[r];
            printf("%-22s %8lld %8lld %8lld %8lld %8lld %9.3f%% %8.3f%%\n",label,c.N(),
                   c.n[0],c.n[1],c.n[2],c.n[3],c.N()?100.*c.AtLeast(1)/c.N():nan,c.N()?100.*c.n[3]/c.N():nan);
            CsvRow(csv,METRIC[m],r<0?"all":"region",label,r<0?-inf:edges[r],r<0?inf:edges[r+1],c);
        }
        for (int b=0;b<=N_BINS+1;++b) {
            const double lo=b==0 ? -inf : V_MIN+(b-1)*(V_MAX-V_MIN)/N_BINS;
            const double hi=b==N_BINS+1 ? inf : V_MIN+b*(V_MAX-V_MIN)/N_BINS;
            CsvRow(csv,METRIC[m],"v_bin",std::to_string(b),lo,hi,bins[m][b]);
        }
    }
    csv.flush(); Require(bool(csv),"CSV write failure"); csv.close();
    output->cd();
    Require(audit->Write()>0,"cannot save event audit");
    hVertex->Write("h_decay_vertex");
    for (int s=0;s<4;++s) TParameter<Long64_t>(Form("status_%d_count",s),statusCounts[s]).Write();
    TParameter<Long64_t>("inspected_events",n).Write();
    TParameter<Long64_t>("vertex_underflow",bins[0][0].N()).Write();
    TParameter<Long64_t>("vertex_overflow",bins[0][N_BINS+1].N()).Write();
    gStyle->SetOptStat(0);
    for (int m=0;m<2;++m) {
        auto* canvas=Draw(METRIC[m],bins[m],regions[m],hVertex,output.get(),unique,siThresholdMeV);
        canvas->SaveAs(prefix+"_"+METRIC[m]+".svg");
        canvas->Print(prefix+".pdf"+(m==0 ? "(" : ")"));
    }
    output->cd();
    TParameter<Long64_t>("selected_decays",statusCounts[1]).Write();
    output->cd(); TParameter<int>("analysis_complete",1).Write("analysis_complete",TObject::kOverwrite);
    output->Flush(); Require(!output->TestBit(TFile::kWriteError),"ROOT output write failure");
    output->Close();
    tree->ResetBranchAddresses(); source->Close();
    delete tracks; for (auto* a:si) delete a;
    gROOT->cd();
    printf("\nSaved: %s.root\n       %s.csv\n",prefix.Data(),prefix.Data());
    printf("Images: %s_arrival.svg, %s_edep.svg; two-page PDF: %s.pdf\n",prefix.Data(),prefix.Data(),prefix.Data());
    printf("ROOT canvases: canvas_arrival, canvas_edep. Empty-bin efficiencies are omitted / CSV nan.\n");
    printf("Vertex plot flow bins: below range=%lld; above range=%lld (kept in CSV and totals).\n",
           bins[0][0].N(),bins[0][N_BINS+1].N());
    if (!statusCounts[1]) printf("WARNING: no selected direct decays; inspect event_audit/genealogy before interpreting efficiencies.\n");
}
