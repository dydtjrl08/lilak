
void count_si_multiplicity(const char* filename = "~/workspace/lilak_seunghwan/data/HIMAC_2702/Ne20_3alphaDirectDecay.root",
                            double edepThresholdMeV = 0.0)
{
    // rootlogon.C가 이미 libLILAK/libNPCore/libNPPhysics를 로드해 놨다면
    // 아래 줄은 그냥 무해하게 넘어감 (이미 로드된 라이브러리 재로드 시도).
    gSystem->Load("libLILAK");

    TFile *f = TFile::Open(filename);
    if (!f || f->IsZombie()) {
        std::cout << "파일 열기 실패: " << filename << std::endl;
        return;
    }

    TTree *t = (TTree*)f->Get("event");
    if (!t) {
        std::cout << "'event' 트리를 찾을 수 없습니다." << std::endl;
        f->ls();
        return;
    }

    // 8개 Si 검출기 branch 이름
    const std::vector<std::string> siNames = {
        "MCStepSiDet_RU", "MCStepSiDet_RM", "MCStepSiDet_RD",
        "MCStepSiDet_LU", "MCStepSiDet_LM", "MCStepSiDet_LD",
        "MCStepSiDet_U",  "MCStepSiDet_D"
    };

    TTreeReader reader(t);

    std::vector<TTreeReaderArray<Double_t>*> edepArrays;
    for (const auto& name : siNames) {
        edepArrays.push_back(
            new TTreeReaderArray<Double_t>(reader, (name + ".fEdep").c_str()));
    }

    const int maxMult = (int)siNames.size(); // 8
    std::vector<Long64_t> countByMultiplicity(maxMult + 1, 0);

    Long64_t nEntries = t->GetEntries();

    while (reader.Next()) {
        int nHitDet = 0;

        for (auto* arrPtr : edepArrays) {
            TTreeReaderArray<Double_t>& edeps = *arrPtr;

            double totalEdep = 0.0;
            for (int j = 0; j < (int)edeps.GetSize(); ++j) totalEdep += edeps[j];

            bool hasHit = (edepThresholdMeV > 0.0)
                            ? (totalEdep > edepThresholdMeV)
                            : (edeps.GetSize() > 0);

            if (hasHit) nHitDet++;
        }

        countByMultiplicity[nHitDet]++;
    }

    std::cout << "\n=== 총 이벤트 수: " << nEntries << " ===\n";
    if (edepThresholdMeV > 0.0)
        std::cout << "(에너지 문턱값 " << edepThresholdMeV << " MeV 적용)\n";
    std::cout << "Si 검출기 hit multiplicity 별 이벤트 개수:\n";
    for (int m = 0; m <= 3 && m <= maxMult; ++m) {
        std::cout << "  Si " << m << "개 hit : "
                   << countByMultiplicity[m] << " events" << std::endl;
    }
    Long64_t over3 = 0;
    for (int m = 4; m <= maxMult; ++m) over3 += countByMultiplicity[m];
    if (over3 > 0)
        std::cout << "  Si 4개 이상 hit : " << over3 << " events (참고용)\n";

    // SetBinContent()로 채우면 fEntries/Mean/StdDev가 잘못 계산되므로
    // (호출 횟수만큼만 "entries"로 잡힘), 실제 이벤트 개수만큼 Fill()을
    // 반복 호출해서 통계박스도 올바르게 나오도록 함.
    TH1D *h = new TH1D("hSiMult", "Si detector hit multiplicity;# of Si detectors hit;events",
                        maxMult + 1, -0.5, maxMult + 0.5);
    for (int m = 0; m <= maxMult; ++m)
        for (Long64_t k = 0; k < countByMultiplicity[m]; ++k) h->Fill(m);

    TCanvas *c = new TCanvas("cSiMult", "Si hit multiplicity", 800, 600);
    h->Draw("hist text0");
    c->SaveAs("si_multiplicity.png");
    std::cout << "\n히스토그램 저장: si_multiplicity.png\n";

    for (auto* p : edepArrays) delete p;
}

