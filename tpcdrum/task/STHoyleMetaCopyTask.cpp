#include "STHoyleMetaCopyTask.h"
#include "LKRun.h"
#include "TFile.h"
#include "TTree.h"
#include "TChain.h"
#include "TBranch.h"
#include "TLeaf.h"
#include "TDirectory.h"
#include "TNamed.h"
#include "TError.h"
#include <stdexcept>
#include <iostream>

ClassImp(STHoyleMetaCopyTask);

STHoyleMetaCopyTask::STHoyleMetaCopyTask()
    : LKTask("STHoyleMetaCopyTask", "Preserve Hoyle source-event provenance") {}

STHoyleMetaCopyTask::~STHoyleMetaCopyTask()
{
    delete fMetadataFile;
}

bool STHoyleMetaCopyTask::Init()
{
    TDirectory::TContext context;
    if (fReady || fMetadataFile) {
        ::Error(GetName(), "Use a new task instance for each run.");
        return false;
    }
    if (!fRun || !fRun->GetInputFile() || !fRun->GetInputTree() ||
        !fRun->GetOutputFile() || !fRun->GetOutputTree()) {
        ::Error(GetName(), "Input/output file and event tree are required.");
        return false;
    }
    auto chain = dynamic_cast<TChain*>(fRun->GetInputTree());
    if (chain && chain->GetListOfFiles()->GetEntries() != 1) {
        ::Error(GetName(), "Only ONE input file is supported; do not chain subset files.");
        return false;
    }
    auto output = fRun->GetOutputFile();
    fEvents = fRun->GetOutputTree();
    if (!output->IsWritable() || fEvents->GetEntries() != 0 ||
        output->Get("HoyleMeta") || fEvents->GetBranch("HoyleInputEntry") ||
        fEvents->GetBranch("HoyleSourceEntry")) {
        ::Error(GetName(), "Use a NEW output: existing entries or provenance would conflict.");
        return false;
    }
    // Separate read-only handle: never change addresses/status of LKRun's input.
    fMetadataFile = TFile::Open(fRun->GetInputFile()->GetName(), "READ");
    if (!fMetadataFile || fMetadataFile->IsZombie()) {
        ::Error(GetName(), "Cannot open input metadata.");
        return false;
    }
    fMetadata = dynamic_cast<TTree*>(fMetadataFile->Get("HoyleMeta"));
    if (!fMetadata || fMetadata->GetEntries() != fRun->GetInputTree()->GetEntries()) {
        ::Error(GetName(), "Input needs HoyleMeta with exactly one row per input event. No IDs will be fabricated.");
        return false;
    }
    auto leaf = fMetadata->GetLeaf("HoyleSourceEntry");
    if (!leaf || TString(leaf->GetTypeName()) != "Long64_t" ||
        leaf->GetLeafCount() || leaf->GetLenStatic() != 1) {
        ::Error(GetName(), "HoyleSourceEntry must be a scalar Long64_t in input HoyleMeta.");
        return false;
    }
    if (fMetadata->SetBranchAddress("HoyleSourceEntry", &fSourceEntry, &fSourceBranch) < 0)
        return false;
    // These values are committed ONLY when LKRun fills its event tree.
    fEntryBranch = fEvents->Branch("HoyleInputEntry", &fInputEntry, "HoyleInputEntry/L");
    auto source = fEvents->Branch("HoyleSourceEntry", &fSourceEntry, "HoyleSourceEntry/L");
    fReady = fEntryBranch && source;
    if (fReady) std::cout << "[STHoyleMetaCopyTask] input HoyleMeta entries: "
                          << fMetadata->GetEntries() << std::endl;
    return fReady;
}

void STHoyleMetaCopyTask::Exec(Option_t*)
{
    if (!fReady || fFinished) throw std::runtime_error("STHoyleMetaCopyTask: invalid task lifecycle");
    fInputEntry = fRun->GetCurrentEventID();
    if (fInputEntry < 0 || fInputEntry >= fMetadata->GetEntries() ||
        fSourceBranch->GetEntry(fInputEntry) <= 0 || fSourceEntry < 0)
        throw std::runtime_error("STHoyleMetaCopyTask: cannot read source ID for current input entry");
}

bool STHoyleMetaCopyTask::EndOfRun()
{
    if (fFinished) return true;
    if (!fReady) return false;
    TDirectory::TContext context;
    auto output = fRun->GetOutputFile();
    if (!output || !output->IsOpen() || !output->IsWritable() || output->Get("HoyleMeta"))
        throw std::runtime_error("STHoyleMetaCopyTask: output closed or HoyleMeta already exists");
    output->cd();
    // Clone ALL metadata fields, including vectors of original track/parent IDs.
    auto copy = fMetadata->CloneTree(0);
    if (!copy) throw std::runtime_error("STHoyleMetaCopyTask: CloneTree failed");
    copy->SetDirectory(output);
    const Long64_t saved = fEvents->GetEntries();
    for (Long64_t i = 0; i < saved; ++i) {
        // Read the branch that was actually committed, not a sequential counter.
        if (fEntryBranch->GetEntry(i) <= 0 || fInputEntry < 0 ||
            fInputEntry >= fMetadata->GetEntries() || fMetadata->GetEntry(fInputEntry) <= 0 ||
            copy->Fill() < 0)
            throw std::runtime_error("STHoyleMetaCopyTask: metadata copy failed; do not use this output");
    }
    if (copy->GetEntries() != saved || copy->Write("HoyleMeta", TObject::kOverwrite) <= 0)
        throw std::runtime_error("STHoyleMetaCopyTask: metadata write failed");
    // Disconnect shared addresses before the read-only metadata file is destroyed.
    copy->ResetBranchAddresses();
    TNamed provenance("HoyleMetaCopyInputFile", fMetadataFile->GetName());
    provenance.Write();
    fFinished = true;
    std::cout << "[STHoyleMetaCopyTask] saved event=" << saved
              << ", HoyleMeta=" << copy->GetEntries() << " (entry-aligned)" << std::endl;
    return true;
}
