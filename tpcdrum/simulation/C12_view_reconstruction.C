// Dictionary-free viewer for reconstruction stage TTrees. No +/++.
// root -l 'C12_view_reconstruction.C("03_finding.root",0)'
// PDF base is optional: saves <base>_event.pdf and <base>_waveforms.pdf.
int C12_view_reconstruction(const char* inputRoot, Long64_t eventRow=0,
                            const char* pdfBase="")
{
    auto* file=TFile::Open(inputRoot,"READ");
    if (!file || file->IsZombie()) {
        ::Error("C12_view_reconstruction","Cannot open stage ROOT");
        if (gROOT->IsBatch()) gSystem->Exit(1);
        return 1;
    }
    auto* tree=dynamic_cast<TTree*>(file->Get("events"));
    if (!tree || eventRow<0 || eventRow>=tree->GetEntries()) {
        ::Error("C12_view_reconstruction","No requested events row");
        if (gROOT->IsBatch()) gSystem->Exit(1);
        return 1;
    }
    tree->GetEntry(eventRow);
    auto leaf=[&](const char* name)->TLeaf* {return tree->GetLeaf(name);};
    auto* wave=leaf("positive_waveform");
    if (!wave) wave=leaf("raw_waveform");
    if (!wave) {
        // Separate truth-evaluation ROOTs intentionally contain no observations.
        auto* metrics=dynamic_cast<TTree*>(file->Get("metrics"));
        auto* directory=dynamic_cast<TDirectory*>(file->Get("performance"));
        if (!metrics || !directory) {
            ::Error("C12_view_reconstruction","Neither embedded waveform nor performance objects exist");
            if (gROOT->IsBatch()) gSystem->Exit(1);
            return 1;
        }
        metrics->Scan("name:value","","colsize=45");
        auto* canvas=new TCanvas("evaluation_performance","Evaluation metrics (not an inference input)",1300,800);
        canvas->Divide(3,2);TIter next(directory->GetListOfKeys());int panel=0;
        while (auto* key=dynamic_cast<TKey*>(next())) {
            auto* hist=dynamic_cast<TH1*>(directory->Get(key->GetName()));
            if (hist && panel<6) {canvas->cd(++panel);hist->Draw("HIST");}
        }
        if (pdfBase && *pdfBase) canvas->Print(TString(pdfBase)+"_performance.pdf");
        return 0;
    }
    if (wave->GetLen()!=12*64*512) {
        ::Error("C12_view_reconstruction","Expected embedded native (12,64,512) waveform");
        if (gROOT->IsBatch()) gSystem->Exit(1);
        return 1;
    }
    auto* components=leaf("components");
    auto* shape=leaf("components__shape");
    const int k=shape ? int(shape->GetValue(0)) : 0;
    auto* fit=leaf("fit_points_uvw_mm");
    auto* valid=leaf("fit_valid_layer");
    auto* vertex=leaf("vertex_uvw_mm");
    auto* vertexOK=leaf("vertex_valid");
    auto* entry=leaf("entry");
    const Long64_t identity=entry ? Long64_t(entry->GetValue()) : eventRow;
    const int colors[]={kBlue+1,kRed+1,kGreen+2,kMagenta+1,kOrange+1};
    auto* map=new TH2D(Form("peak_%lld",eventRow),
        Form("Entry %lld: observed peak;native pad index;layer index",identity),64,-.5,63.5,12,-.5,11.5);
    int chosen[12]={0};
    for (int l=0;l<12;++l) {
        double largest=-1.;
        for (int p=0;p<64;++p) {
            double peak=0.,sum=0.;
            for (int t=0;t<512;++t) {
                const double q=wave->GetValue((l*64+p)*512+t);
                peak=std::max(peak,q);sum+=std::max(0.,q);
            }
            map->SetBinContent(p+1,l+1,peak);
            if (sum>largest) {largest=sum;chosen[l]=p;}
        }
    }
    auto* overview=new TCanvas(Form("overview_%lld",eventRow),"Observed signal and fitted paths",1500,550);
    overview->Divide(3,1);
    overview->cd(1);gPad->SetRightMargin(.16);map->SetStats(false);map->Draw("COLZ");
    for (int projection=0;projection<2;++projection) {
        overview->cd(projection+2);
        const double low=projection ? -20. : -50.;
        const double high=projection ? 200. : 50.;
        auto* frame=gPad->DrawFrame(-100.,low,210.,high,
            projection ? "Fitted paths;v [mm];w [mm]" : "Fitted paths;v [mm];u [mm]");
        if (fit && valid) for (int a=0;a<k;++a) {
            auto* line=new TGraph();
            for (int l=0;l<12;++l) if (valid->GetValue(a*12+l)) {
                const int index=(a*12+l)*3;
                const double v=fit->GetValue(index+1),x=fit->GetValue(index+(projection ? 2 : 0));
                if (std::isfinite(v) && std::isfinite(x)) line->SetPoint(line->GetN(),v,x);
            }
            line->SetLineColor(colors[a%5]);line->SetMarkerColor(colors[a%5]);
            line->SetMarkerStyle(20);line->SetLineWidth(2);line->Draw("LP SAME");
        }
        if (vertex && vertexOK && vertexOK->GetValue()) {
            auto* point=new TGraph(1);
            point->SetPoint(0,vertex->GetValue(1),vertex->GetValue(projection ? 2 : 0));
            point->SetMarkerStyle(29);point->SetMarkerSize(2);point->Draw("P SAME");
        }
        if (!fit) {auto* text=new TLatex();text->SetNDC();text->SetTextSize(.04);text->DrawLatex(.17,.85,"No fitted trajectory in this stage");}
        (void)frame;
    }
    auto* pulses=new TCanvas(Form("waveforms_%lld",eventRow),"Selected observed pad and inferred contributions",1550,1000);
    pulses->Divide(3,4,.002,.002);
    for (int l=0;l<12;++l) {
        pulses->cd(l+1);const int p=chosen[l];
        auto* observed=new TH1D(Form("wave_%lld_%d",eventRow,l),
            Form("L%02d pad %02d: largest observed charge;TB;ADC",l,p),512,-.5,511.5);
        for (int t=0;t<512;++t) observed->SetBinContent(t+1,wave->GetValue((l*64+p)*512+t));
        observed->SetStats(false);observed->SetLineColor(kBlack);observed->SetLineWidth(2);
        observed->Draw("HIST");
        if (components) for (int a=0;a<k;++a) {
            auto* part=new TH1D(Form("component_%lld_%d_%d",eventRow,l,a),"",512,-.5,511.5);
            for (int t=0;t<512;++t) part->SetBinContent(t+1,components->GetValue(((a*12+l)*64+p)*512+t));
            part->SetLineColor(colors[a%5]);part->SetLineWidth(2);part->Draw("HIST SAME");
        }
        if (l==0) {
            auto* text=new TLatex();text->SetNDC();text->SetTextSize(.045);
            text->DrawLatex(.16,.84,"Black: observed; colors: inferred slots");
            text->DrawLatex(.16,.77,"Slots are arbitrary; not MC track IDs");
        }
    }
    if (pdfBase && *pdfBase) {
        overview->Print(TString(pdfBase)+"_event.pdf");
        pulses->Print(TString(pdfBase)+"_waveforms.pdf");
    }
    auto* metrics=dynamic_cast<TTree*>(file->Get("metrics"));
    if (metrics) metrics->Scan("name:value","","colsize=45");
    auto* directory=dynamic_cast<TDirectory*>(file->Get("performance"));
    if (directory) {
        auto* performance=new TCanvas(Form("performance_%lld",eventRow),"Stage diagnostics; truth accuracy only when explicitly evaluated",1300,800);
        performance->Divide(3,2);TIter next(directory->GetListOfKeys());int panel=0;
        while (auto* key=dynamic_cast<TKey*>(next())) {
            auto* hist=dynamic_cast<TH1*>(directory->Get(key->GetName()));
            if (hist && panel<6) {performance->cd(++panel);hist->Draw("HIST");}
        }
        if (pdfBase && *pdfBase) performance->Print(TString(pdfBase)+"_performance.pdf");
    }
    // Keep the input open while the interactive ROOT canvases refer to its objects.
    return 0;
}
