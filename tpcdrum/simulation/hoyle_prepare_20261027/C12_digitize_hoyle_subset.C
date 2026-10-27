// DIGITIZATION ONLY. First create MC subsets with C12_make_hoyle_subsets.C.
// Load rebuilt LILAK with the project's rootlogon. No +/++ or #include needed.
// One input subset per fresh ROOT process. K is read from HoyleMeta (1/2/3).
// TPC waveform + Si deposited energy in MeV; no Si waveform response.

namespace c12_hoyle_digi_20261027 {
void Require(bool ok, const TString& message)
{
    if (!ok) throw std::runtime_error(message.Data());
}

TString Absolute(const char* value)
{
    Require(value && *value, "Empty input/output path.");
    TString path(value);
    gSystem->ExpandPathName(path);
    if (!gSystem->IsAbsoluteFileName(path))
        gSystem->PrependPathName(gSystem->WorkingDirectory(), path);
    return path;
}

Long64_t Validate(const TString& path)
{
    std::unique_ptr<TFile> file(TFile::Open(path, "READ"));
    Require(file && !file->IsZombie() && !file->TestBit(TFile::kRecovered),
            "Subset is missing, corrupt or recovered: " + path);
    auto* tree = dynamic_cast<TTree*>(file->Get("event"));
    auto* meta = dynamic_cast<TTree*>(file->Get("HoyleMeta"));
    auto* complete = dynamic_cast<TParameter<int>*>(file->Get("SubsetComplete"));
    auto* schema = dynamic_cast<TNamed*>(file->Get("C12SubsetSchema"));
    auto* accepted = dynamic_cast<TParameter<Long64_t>*>(file->Get("SubsetAccepted"));
    Require(tree && meta && tree->GetEntries() > 0 &&
            tree->GetEntries() == meta->GetEntries() && complete && complete->GetVal() == 1 &&
            schema && TString(schema->GetTitle()) == "c12-direct-all-combinations-v1" &&
            accepted && accepted->GetVal() > 0,
            "Use a completed, nonempty MC subset from C12_make_hoyle_subsets.C.");
    Require(!tree->GetBranch("RawPad") && tree->GetBranch("MCTrack") &&
            tree->GetBranch("MCStepTPCDrum"), "Input must be subset MC before digitization.");
    const char* detectors[] = {"RU", "RM", "RD", "LU", "LM", "LD", "U", "D"};
    for (const char* detector : detectors) {
        const TString branch = TString("MCStepSiDet_") + detector;
        Require(tree->GetBranch(branch), "Missing Si MCStep branch: " + branch);
    }
    auto* uuid = dynamic_cast<TNamed*>(file->Get("HoyleSourceUUID"));
    Require(uuid && TString(uuid->GetTitle()).Length(), "Missing original source UUID.");
    const char* fields[] = {"HoyleSourceEntry", "HoyleCombinationIndex", "HoyleSubsetMultiplicity",
        "HoyleSelectionMode", "HoyleOriginalCarbon12TrackID", "HoyleOriginalTrackIDs",
        "HoyleOriginalParentIDs", "HoyleTripletOriginalTrackIDs", "HoyleAlphaIndicesInTriplet"};
    for (const char* name : fields)
        Require(meta->GetBranch(name), TString("Missing metadata: ") + name);

    Long64_t source = -1, groupSource = -1;
    int k = 0, fixedK = 0, combination = -1, mode = -1, carbon = -1;
    std::vector<int> *ids = nullptr, *parents = nullptr, *triplet = nullptr, *indices = nullptr;
    TClonesArray* tracks = nullptr;
    tree->SetBranchStatus("*", 0);
    tree->SetBranchStatus("MCTrack*", 1);
    Require(tree->SetBranchAddress("MCTrack", &tracks) >= 0, "Cannot bind MCTrack.");
    Require(meta->SetBranchAddress(fields[0], &source) >= 0 &&
            meta->SetBranchAddress(fields[1], &combination) >= 0 &&
            meta->SetBranchAddress(fields[2], &k) >= 0 &&
            meta->SetBranchAddress(fields[3], &mode) >= 0 &&
            meta->SetBranchAddress(fields[4], &carbon) >= 0 &&
            meta->SetBranchAddress(fields[5], &ids) >= 0 &&
            meta->SetBranchAddress(fields[6], &parents) >= 0 &&
            meta->SetBranchAddress(fields[7], &triplet) >= 0 &&
            meta->SetBranchAddress(fields[8], &indices) >= 0, "Cannot bind subset metadata.");
    const std::array<std::vector<std::vector<int>>, 3> patterns = {{
        {{0}, {1}, {2}}, {{0, 1}, {0, 2}, {1, 2}}, {{0, 1, 2}}
    }};
    const Long64_t entries = tree->GetEntries();
    for (Long64_t e = 0; e < entries; ++e) {
        Require(meta->GetEntry(e) > 0 && tree->GetEntry(e) > 0, "Subset read failure.");
        if (e == 0) fixedK = k;
        Require(k >= 1 && k <= 3 && k == fixedK && mode == 0 && carbon > 0 &&
                ids && parents && triplet && indices &&
                ids->size() == static_cast<std::size_t>(k) &&
                parents->size() == static_cast<std::size_t>(k) && triplet->size() == 3 &&
                tracks && tracks->GetEntriesFast() == k && source >= 0,
                Form("Invalid track/metadata count at entry %lld", e));
        const int ncomb = k == 3 ? 1 : 3;
        Require(combination == e % ncomb && *indices == patterns[k-1][combination],
                "Combination order does not match all-combinations MC output.");
        if (combination == 0) {
            Require(source > groupSource, "Original source groups are not increasing.");
            groupSource = source;
        } else Require(source == groupSource, "Sibling source group mismatch.");
        for (int a = 0; a < 3; ++a)
            Require((*triplet)[a] > 0 && (a == 0 || (*triplet)[a] > (*triplet)[a-1]),
                    "Invalid original triplet IDs.");
        for (int a = 0; a < k; ++a) {
            auto* track = dynamic_cast<LKMCTrack*>(tracks->At(a));
            Require(track && track->GetPDG() == 1000020040 && track->GetTrackID() == a+1 &&
                    track->GetParentID() == 0 && track->GetNumVertices() > 0 &&
                    (*parents)[a] == carbon && (*ids)[a] == (*triplet)[(*indices)[a]],
                    Form("Invalid selected alpha at entry %lld", e));
        }
        if ((e+1) % 10000 == 0)
            std::cout << "[C12 digi preflight] " << e+1 << "/" << entries << std::endl;
    }
    Require(entries == accepted->GetVal() * (fixedK == 3 ? 1 : 3),
            "Subset output count does not match the accepted direct decays.");
    tree->ResetBranchAddresses();
    meta->ResetBranchAddresses();
    delete tracks; delete ids; delete parents; delete triplet; delete indices;
    std::cout << "[C12 digi preflight] completed 3C" << fixedK
              << " MC: " << entries << " rows; original-parent proof present." << std::endl;
    return entries;
}
} // namespace c12_hoyle_digi_20261027

bool C12_digitize_hoyle_subset(const char* inputSubset,
                               const char* outputDigi = "",
                               Long64_t numEvents = -1,
                               Long64_t startEntry = 0,
                               const char* parameterFile = "SimParameters.mac")
{
    using namespace c12_hoyle_digi_20261027;
    try {
        Require(numEvents == -1 || numEvents > 0, "numEvents must be positive or -1 for all remaining entries.");
        Require(startEntry >= 0, "startEntry must be nonnegative.");
        const TString input = Absolute(inputSubset), parameters = Absolute(parameterFile);
        TString output(outputDigi ? outputDigi : "");
        if (output.IsNull()) {
            output = input;
            if (output.EndsWith(".root")) output.Remove(output.Length()-5);
            output += "_digi_SiEdep.root";
        }
        output = Absolute(output);
        Require(input != output && output != parameters, "Input/output paths must differ.");
        Require(gSystem->AccessPathName(output), "Refusing to overwrite: " + output);
        Require(!gSystem->AccessPathName(gSystem->DirName(output), kWritePermission),
                "Output directory is missing or not writable: " + output);
        Require(!gSystem->AccessPathName(parameters), "Parameter file is missing: " + parameters);
        const Long64_t total = Validate(input);
        Require(startEntry < total, "startEntry is beyond the end of the input subset.");
        const Long64_t available = total-startEntry;
        const Long64_t requested = numEvents < 0 ? available : std::min(numEvents, available);
        // Metadata and event must stay in one physical output file. For explicit
        // batches use numEvents/startEntry with a NEW output name for each batch.
        TTree::SetMaxTreeSize(1000000000000000LL);
        auto* run = new LKRun();
        run->SetAutoTermination(false);
        run->SetEventCountForMessage(1000);
        run->AddPar(parameters);
        run->AddDetector(new TPCDrum());
        run->SetInputFile(input);
        run->SetOutputFile(output);
        run->Add(new STDSiEdepSumTask());
        run->Add(new STDDriftElectronMaker());
        run->Add(new STDElectronicsMaker());
        run->Add(new STHoyleMetaCopyTask());
        Require(run->Init(), "LKRun digitization initialization failed.");
        run->SetAutoTermination(false);
        auto* file = run->GetOutputFile();
        Require(file && file->IsOpen() && file->IsWritable() &&
                TString(file->GetName()) == output, "LKRun did not open the requested output path.");
        Require(run->GetNumEvents() >= startEntry+requested,
                "LKRun parameters limited the requested input range.");
        file->cd();
        TParameter<int>("C12DigiComplete", 0).Write();
        file->Flush();
        std::cout << "[C12 digi] input: " << input << "\noutput: " << output
                  << "\nparameters: " << parameters << "\nentries: " << startEntry
                  << ".." << startEntry+requested-1 << " of " << total
                  << "\nTPC waveforms + Si deposited energy; no Si waveform response." << std::endl;
        if (startEntry > 0) run->Run(startEntry, startEntry+requested-1);
        else run->Run(requested);
        auto* tree = run->GetOutputTree();
        auto* meta = dynamic_cast<TTree*>(file->Get("HoyleMeta"));
        Require(file->IsOpen() && tree && tree->GetCurrentFile() == file &&
                tree->GetEntries() == requested && meta && meta->GetEntries() == requested,
                "Early stop, file rollover or metadata alignment failure; output is incomplete.");
        const char* required[] = {"RawPad", "MCTrack", "MCTagTPCDrum", "EdepSumSiInputEntry",
            "EdepSumSiTotalMeV", "EdepSumSiTrackID", "EdepSumSiTrackDetID", "EdepSumSiTrackMeV",
            "HoyleInputEntry", "HoyleSourceEntry"};
        for (const char* name : required)
            Require(tree->GetBranch(name), TString("Digi is missing branch ") + name);
        std::unique_ptr<TFile> source(TFile::Open(input, "READ"));
        Require(source && !source->IsZombie(), "Cannot reopen original subset metadata.");
        file->cd();
        const char* objects[] = {"HoyleSourceFile", "HoyleSourceUUID", "SubsetSelection",
            "C12SubsetSchema", "SubsetStepPolicy", "SubsetVertexToleranceMM", "SubsetComplete",
            "SubsetInputEntries", "SubsetRequestedEntries", "SubsetProcessedAll", "SubsetInspected",
            "SubsetAccepted", "SubsetNoTriplet", "SubsetAmbiguous", "SubsetMalformed"};
        for (const char* name : objects) {
            auto* object = source->Get(name);
            Require(object && object->Write(name, TObject::kOverwrite) > 0,
                    TString("Cannot preserve source proof: ") + name);
        }
        TNamed("C12DigiInputSubset", input.Data()).Write();
        TNamed("C12DigiParameterFile", parameters.Data()).Write();
        TNamed("C12DigiSiEnergy", "Own alpha MCStep deposited energy in MeV; no Si ADC/smearing/threshold").Write();
        TParameter<Long64_t>("C12DigiInputEntries", total).Write();
        TParameter<Long64_t>("C12DigiStartEntry", startEntry).Write();
        TParameter<Long64_t>("C12DigiRequestedEvents", requested).Write();
        file->Flush();
        Require(!file->TestBit(TFile::kWriteError), "Output disk write failure.");
        Require(TParameter<int>("C12DigiComplete", 1).Write("", TObject::kOverwrite) > 0,
                "Cannot write digi completion marker.");
        file->Flush();
        Require(!file->TestBit(TFile::kWriteError), "Output disk write failure.");
        file->Close();
        std::cout << "[C12 digi] COMPLETED " << requested << " entries: " << output << std::endl;
        return true;
    } catch (const std::exception& error) {
        ::Error("C12_digitize_hoyle_subset", "%s", error.what());
        return false;
    }
}
