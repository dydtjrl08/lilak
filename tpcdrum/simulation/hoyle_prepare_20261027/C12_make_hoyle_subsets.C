// MC SELECTION ONLY. No drift, GEM, electronics or Si response is run.
// Interpreted macro: load LILAK via rootlogon; do not append + or ++.
// No #include directives and no new task/dictionary/project rebuild required.
// Adapted from the existing, tested standalone STDecaySubsetBuilder.
// 3C3 = all three decay alphas; 3C2/3C1 = all 3 pair/single combinations.
// These names are NOT Si-hit multiplicity selections.
// Original MC is read-only. Existing output files are never overwritten.



// Standalone MC preprocessing tool, NOT an LKRun digitization task.
// No LILAK class changes or ROOT dictionary for this class are required.
class C12HoyleSubsetBuilder20261027 {
public:
    struct Summary {
        Long64_t inputEntries = 0;
        Long64_t requestedEntries = 0;
        Long64_t inspected = 0;
        Long64_t accepted = 0;
        Long64_t noTriplet = 0;
        Long64_t ambiguous = 0;
        Long64_t malformed = 0;
    };

    // Writes prefix_3C1.root, prefix_3C2.root, prefix_3C3.root in one pass.
    // Requires stored parentage: C12 -> alpha + alpha + alpha at one vertex.
    // Existing output files are never overwritten. Errors throw exceptions.
    Summary Run(const char* inputFile, const char* outputPrefix,
                Long64_t maxEntries = -1, double vertexToleranceMM = 1.e-3);
};

namespace c12_hoyle_subset_20261027 {
void Check(bool ok, const std::string& message) {
    if (!ok) throw std::runtime_error("C12HoyleSubsetBuilder20261027: " + message);
}
TString Path(const char* value) {
    Check(value && *value, "empty path");
    TString path(value);
    gSystem->ExpandPathName(path);
    if (!gSystem->IsAbsoluteFileName(path))
        gSystem->PrependPathName(gSystem->WorkingDirectory(), path);
    return path;
}
bool Finite(const TVector3& v) {
    return std::isfinite(v.X()) && std::isfinite(v.Y()) && std::isfinite(v.Z());
}
bool Carbon12(int pdg) {
    // Nuclear PDG 10LZZZAAAI: accept C12 excitation/isomer digit I as well.
    return pdg / 10 == 100006012;
}
bool Alpha(int pdg) { return pdg == 1000020040; }

struct Triplet {
    int parent;
    TVector3 vertex;
    std::array<LKMCTrack*, 3> tracks;
};

std::vector<Triplet> Find(TClonesArray* tracks, double tolerance, bool& malformed) {
    std::map<int, LKMCTrack*> byID;
    std::map<int, std::vector<LKMCTrack*>> daughters;
    malformed = false;
    for (int i = 0; i < tracks->GetEntriesFast(); ++i) {
        auto* t = dynamic_cast<LKMCTrack*>(tracks->At(i));
        if (!t || t->GetTrackID() <= 0 || t->GetNumVertices() < 1 ||
            !Finite(t->GetPrimaryPosition()) || !byID.emplace(t->GetTrackID(), t).second) {
            malformed = true;
            return {};
        }
        if (Alpha(t->GetPDG())) daughters[t->GetParentID()].push_back(t);
    }
    std::vector<Triplet> found;
    for (auto& item : daughters) {
        const auto parent = byID.find(item.first);
        // No vertex-only fallback: recoil alpha can share the same vertex.
        if (parent == byID.end() || !Carbon12(parent->second->GetPDG()) ||
            item.second.size() != 3) continue;
        auto& a = item.second;
        bool sameVertex = true;
        for (int i = 0; i < 3; ++i)
            for (int j = i + 1; j < 3; ++j)
                sameVertex &= (a[i]->GetPrimaryPosition() - a[j]->GetPrimaryPosition()).Mag() <= tolerance;
        if (!sameVertex) continue;
        std::sort(a.begin(), a.end(), [](LKMCTrack* x, LKMCTrack* y) {
            return x->GetTrackID() < y->GetTrackID();
        });
        found.push_back({item.first, a[0]->GetPrimaryPosition(), {{a[0], a[1], a[2]}}});
    }
    return found;
}

void CopyTrack(const LKMCTrack* from, LKMCTrack* to, int id) {
    for (int v = 0; v < from->GetNumVertices(); ++v) {
        Check(std::isfinite(from->GetVX(v)) && std::isfinite(from->GetVY(v)) &&
              std::isfinite(from->GetVZ(v)) && std::isfinite(from->GetPX(v)) &&
              std::isfinite(from->GetPY(v)) && std::isfinite(from->GetPZ(v)) &&
              std::isfinite(from->GetEnergy(v)) && from->GetEnergy(v) >= 0.,
              "selected alpha has invalid stored position/momentum/kinetic energy");
    }
    // All stored trajectory vertices, initial momentum/energy and volume/status
    // values are retained. Only track/parent IDs change for a standalone subset.
    to->SetMCTrack(id, 0, from->GetPDG(), from->GetVolumeID(0), from->GetStatusID(0),
                  from->GetVX(0), from->GetVY(0), from->GetVZ(0),
                  from->GetPX(0), from->GetPY(0), from->GetPZ(0), from->GetEnergy(0));
    for (int i = 1; i < from->GetNumVertices(); ++i)
        to->AddVertex(from->GetVolumeID(i), from->GetStatusID(i),
                      from->GetVX(i), from->GetVY(i), from->GetVZ(i),
                      from->GetPX(i), from->GetPY(i), from->GetPZ(i), from->GetEnergy(i));
}

struct Input {
    std::unique_ptr<TFile> file;
    TTree* tree = nullptr;
    TClonesArray* tracks = nullptr;
    std::vector<TString> names;
    std::vector<TClonesArray*> steps;
    ~Input() {
        if (tree) tree->ResetBranchAddresses();
        delete tracks;
        for (auto* step : steps) delete step;
    }
};

struct Output {
    std::unique_ptr<TFile> file;
    TTree *event = nullptr, *meta = nullptr;
    TClonesArray* tracks = nullptr;
    std::vector<TClonesArray*> steps;
    Long64_t source = -1;
    int candidate = 0, combination = 0, multiplicity, carbon = -1, mode = 0;
    double x = 0, y = 0, z = 0;
    std::vector<int> selected, originalIDs, originalParents, tripletIDs;

    Output(const TString& path, int k, Input& input, double tolerance) : multiplicity(k) {
        file.reset(TFile::Open(path, "CREATE"));
        Check(file && !file->IsZombie(), "cannot create " + std::string(path.Data()));
        file->cd();
        TParameter<int>("SubsetComplete", 0).Write();
        TNamed("C12SubsetSchema", "c12-direct-all-combinations-v1").Write();
        TNamed("SubsetStepPolicy", "Selected alpha own-track steps only; no recoil, parent carbon or secondary daughters").Write();
        TNamed("HoyleSourceFile", input.file->GetName()).Write();
        TNamed("HoyleSourceUUID", input.file->GetUUID().AsString()).Write();
        TNamed("SubsetSelection", "Exactly one stored direct C12->3alpha triplet; common vertex; no acceptance cut; alpha-own steps only").Write();
        TParameter<double>("SubsetVertexToleranceMM", tolerance).Write();
        // Preserve run parameters/geometry objects; do not copy unrelated trees.
        std::set<std::string> copied;
        TIter next(input.file->GetListOfKeys());
        while (auto* key = dynamic_cast<TKey*>(next())) {
            if (!copied.insert(key->GetName()).second) continue;
            std::unique_ptr<TObject> object(key->ReadObj());
            if (object && !object->InheritsFrom(TTree::Class()) &&
                !file->GetKey(key->GetName())) object->Write(key->GetName());
        }
        file->cd();
        event = new TTree("event", "Selected direct-decay alpha subset MC");
        meta = new TTree("HoyleMeta", "One provenance row per output event");
        event->SetAutoFlush(-16000000);
        meta->SetAutoFlush(-2000000);
        tracks = new TClonesArray("LKMCTrack");
        steps.resize(input.names.size(), nullptr);
        event->Branch("MCTrack", &tracks, 32000, 99);
        for (std::size_t b = 0; b < steps.size(); ++b) {
            steps[b] = new TClonesArray("LKMCStep");
            event->Branch(input.names[b], &steps[b], 32000, 99);
        }
        // Only TClonesArray branches in event: compatible with LKRun input.
        meta->Branch("HoyleSourceEntry", &source, "HoyleSourceEntry/L");
        meta->Branch("HoyleCandidateIndex", &candidate, "HoyleCandidateIndex/I");
        meta->Branch("HoyleCombinationIndex", &combination, "HoyleCombinationIndex/I");
        meta->Branch("HoyleSubsetMultiplicity", &multiplicity, "HoyleSubsetMultiplicity/I");
        meta->Branch("HoyleOriginalCarbon12TrackID", &carbon, "HoyleOriginalCarbon12TrackID/I");
        meta->Branch("HoyleSelectionMode", &mode, "HoyleSelectionMode/I");
        meta->Branch("HoyleDecayVertexX", &x, "HoyleDecayVertexX/D");
        meta->Branch("HoyleDecayVertexY", &y, "HoyleDecayVertexY/D");
        meta->Branch("HoyleDecayVertexZ", &z, "HoyleDecayVertexZ/D");
        meta->Branch("HoyleAlphaIndicesInTriplet", &selected);
        meta->Branch("HoyleOriginalTrackIDs", &originalIDs);
        meta->Branch("HoyleOriginalParentIDs", &originalParents);
        meta->Branch("HoyleTripletOriginalTrackIDs", &tripletIDs);
    }
    ~Output() {
        if (event) event->ResetBranchAddresses();
        if (meta) meta->ResetBranchAddresses();
        delete tracks;
        for (auto* step : steps) delete step;
    }
    void Fill(Long64_t entry, const Triplet& trio, int comb,
              const std::vector<int>& indices, const Input& input) {
        source = entry; combination = comb; carbon = trio.parent;
        x = trio.vertex.X(); y = trio.vertex.Y(); z = trio.vertex.Z();
        selected = indices;
        tracks->Clear("C");
        for (auto* s : steps) s->Clear("C");
        originalIDs.clear(); originalParents.clear(); tripletIDs.clear();
        for (auto* t : trio.tracks) tripletIDs.push_back(t->GetTrackID());
        std::map<int, int> remap;
        for (std::size_t i = 0; i < indices.size(); ++i) {
            auto* from = trio.tracks[indices[i]];
            CopyTrack(from, static_cast<LKMCTrack*>(tracks->ConstructedAt(i)), i + 1);
            remap[from->GetTrackID()] = i + 1;
            originalIDs.push_back(from->GetTrackID());
            originalParents.push_back(from->GetParentID());
        }
        for (std::size_t b = 0; b < steps.size(); ++b) {
            for (int i = 0; i < input.steps[b]->GetEntriesFast(); ++i) {
                auto* from = dynamic_cast<LKMCStep*>(input.steps[b]->At(i));
                Check(from != nullptr, "invalid MCStep element");
                auto mapped = remap.find(from->GetTrackID());
                if (mapped == remap.end()) continue;
                Check(std::isfinite(from->GetX()) && std::isfinite(from->GetY()) &&
                      std::isfinite(from->GetZ()) && std::isfinite(from->GetTime()) &&
                      std::isfinite(from->GetEdep()) && from->GetEdep() >= 0.,
                      "selected alpha has an invalid MCStep");
                auto* to = static_cast<LKMCStep*>(steps[b]->ConstructedAt(steps[b]->GetEntriesFast()));
                to->SetMCStep(mapped->second, from->GetX(), from->GetY(), from->GetZ(),
                              from->GetTime(), from->GetEdep());
            }
        }
        Check(event->Fill() >= 0 && meta->Fill() >= 0, "output Fill failed");
    }
    void Finish(const C12HoyleSubsetBuilder20261027::Summary& summary) {
        file->cd();
        Check(event->GetEntries() == meta->GetEntries(), "metadata alignment failed");
        const Long64_t expected = summary.accepted * (multiplicity == 3 ? 1 : 3);
        Check(event->GetEntries() == expected, "all-combinations output count mismatch");
        Check(event->Write("", TObject::kOverwrite) > 0 &&
              meta->Write("", TObject::kOverwrite) > 0, "output tree write failed");
        TParameter<Long64_t>("SubsetInputEntries", summary.inputEntries).Write();
        TParameter<Long64_t>("SubsetRequestedEntries", summary.requestedEntries).Write();
        TParameter<int>("SubsetProcessedAll", summary.inspected == summary.inputEntries).Write();
        TParameter<Long64_t>("SubsetInspected", summary.inspected).Write();
        TParameter<Long64_t>("SubsetAccepted", summary.accepted).Write();
        TParameter<Long64_t>("SubsetNoTriplet", summary.noTriplet).Write();
        TParameter<Long64_t>("SubsetAmbiguous", summary.ambiguous).Write();
        TParameter<Long64_t>("SubsetMalformed", summary.malformed).Write();
        file->Flush();
        Check(!file->TestBit(TFile::kWriteError), "output disk write failed");
        Check(TParameter<int>("SubsetComplete", 1).Write("", TObject::kOverwrite) > 0,
              "completion marker write failed");
        file->Flush();
        Check(!file->TestBit(TFile::kWriteError), "output disk write failed");
        std::cout << file->GetName() << " : " << event->GetEntries() << " entries\n";
    }
};
} // namespace c12_hoyle_subset_20261027

C12HoyleSubsetBuilder20261027::Summary C12HoyleSubsetBuilder20261027::Run(
    const char* inputFile, const char* outputPrefix, Long64_t maxEntries, double tolerance) {
    using namespace c12_hoyle_subset_20261027;
    Check(maxEntries == -1 || maxEntries > 0, "maxEntries must be -1 or positive");
    Check(std::isfinite(tolerance) && tolerance >= 0, "invalid vertex tolerance");
    const TString inputPath = Path(inputFile), prefix = Path(outputPrefix);
    Input input;
    input.file.reset(TFile::Open(inputPath, "READ"));
    Check(input.file && !input.file->IsZombie(), "cannot open MC input");
    Check(!input.file->TestBit(TFile::kRecovered), "input required ROOT recovery; confirm the simulation finished cleanly first");
    Check(!input.file->Get("HoyleMeta"), "input is already a subset; use the original simulation MC file");
    input.tree = dynamic_cast<TTree*>(input.file->Get("event"));
    Check(input.tree && input.tree->GetBranch("MCTrack"), "missing event/MCTrack");
    Check(input.tree->GetBranch("MCStepTPCDrum"), "missing MCStepTPCDrum; use transport MC with saved steps");
    Check(!input.tree->GetBranch("RawPad"), "input must be MC before digitization, not RawPad digi");
    auto* branches = input.tree->GetListOfBranches();
    for (int b = 0; b < branches->GetEntriesFast(); ++b) {
        TString name = branches->At(b)->GetName();
        if (name.BeginsWith("MCStep")) input.names.push_back(name);
    }
    Check(!input.names.empty(), "no MCStep* branches: cannot digitize this input");
    input.steps.resize(input.names.size(), nullptr);
    input.tree->SetBranchStatus("*", 0);
    input.tree->SetBranchStatus("MCTrack*", 1);
    Check(input.tree->SetBranchAddress("MCTrack", &input.tracks) >= 0, "MCTrack binding failed");
    for (std::size_t b = 0; b < input.names.size(); ++b) {
        input.tree->SetBranchStatus(input.names[b] + "*", 1);
        Check(input.tree->SetBranchAddress(input.names[b], &input.steps[b]) >= 0,
              "MCStep binding failed");
    }
    std::array<TString, 3> paths;
    for (int k = 1; k <= 3; ++k) {
        paths[k - 1] = prefix + Form("_3C%d.root", k);
        Check(gSystem->AccessPathName(paths[k - 1]),
              "refusing to overwrite " + std::string(paths[k - 1].Data()));
    }
    const TString directory = gSystem->DirName(prefix);
    Check(!gSystem->AccessPathName(directory, kWritePermission), "output directory missing/not writable");
    std::array<std::unique_ptr<Output>, 3> outputs;
    for (int k = 1; k <= 3; ++k) outputs[k - 1].reset(new Output(paths[k - 1], k, input, tolerance));
    Summary summary;
    summary.inputEntries = input.tree->GetEntries();
    const Long64_t n = maxEntries < 0 ? input.tree->GetEntries() :
        std::min(maxEntries, input.tree->GetEntries());
    summary.requestedEntries = n;
    // Keep event and HoyleMeta in the same file. Explicit count windows are
    // available in the separate digitizer rather than implicit ROOT rollover.
    TTree::SetMaxTreeSize(1000000000000000LL);
    const std::array<std::vector<std::vector<int>>, 3> patterns = {{
        {{0}, {1}, {2}}, {{0, 1}, {0, 2}, {1, 2}}, {{0, 1, 2}}
    }};
    for (Long64_t entry = 0; entry < n; ++entry) {
        Check(input.tree->GetEntry(entry) > 0 && input.tracks, "input read failed at " + std::to_string(entry));
        for (auto* s : input.steps) Check(s != nullptr, "null MCStep array");
        ++summary.inspected;
        bool malformed;
        auto candidates = Find(input.tracks, tolerance, malformed);
        if (malformed) ++summary.malformed;
        else if (candidates.empty()) ++summary.noTriplet;
        else if (candidates.size() != 1) ++summary.ambiguous;
        else {
            ++summary.accepted;
            for (int k = 0; k < 3; ++k)
                for (std::size_t c = 0; c < patterns[k].size(); ++c)
                    outputs[k]->Fill(entry, candidates[0], c, patterns[k][c], input);
        }
        if ((entry + 1) % 1000 == 0 || entry + 1 == n)
            std::cout << "[C12HoyleSubsetBuilder20261027] " << entry + 1 << "/" << n
                      << "; accepted " << summary.accepted << std::endl;
    }
    for (auto& output : outputs) output->Finish(summary);
    std::cout << "Inspected=" << summary.inspected << "; accepted=" << summary.accepted
              << "; no confirmed triplet=" << summary.noTriplet
              << "; ambiguous=" << summary.ambiguous << "; malformed=" << summary.malformed << '\n';
    if (!summary.accepted)
        std::cout << "WARNING: no confirmed direct C12->3alpha. Check stored MCTrack parentage; do not digitize empty files.\n";
    return summary;
}

bool C12_make_hoyle_subsets(const char* inputMC,
                           const char* outputPrefix = "",
                           Long64_t maxEntries = -1,
                           double vertexToleranceMM = 1.e-3)
{
    try {
        TString prefix(outputPrefix ? outputPrefix : "");
        if (prefix.IsNull()) {
            prefix = c12_hoyle_subset_20261027::Path(inputMC);
            if (prefix.EndsWith(".root")) prefix.Remove(prefix.Length() - 5);
            prefix += "_decay";
        }
        C12HoyleSubsetBuilder20261027 builder;
        const auto summary = builder.Run(inputMC, prefix, maxEntries, vertexToleranceMM);
        std::cout << "[C12 MC subsets] direct decays=" << summary.accepted
                  << "; 3C3=" << summary.accepted
                  << "; 3C2=" << summary.accepted * 3
                  << "; 3C1=" << summary.accepted * 3
                  << "; processed all original entries="
                  << (summary.inspected == summary.inputEntries ? "YES" : "NO (limited test)")
                  << "\nMC separation only: no digitization was started." << std::endl;
        if (summary.accepted == 0) {
            ::Error("C12_make_hoyle_subsets", "No confirmed direct C12->3alpha was found. Empty output is not a usable digi sample.");
            return false;
        }
        return true;
    } catch (const std::exception& error) {
        ::Error("C12_make_hoyle_subsets", "%s", error.what());
        return false;
    }
}
