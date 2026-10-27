#include "STDSiEdepSumTask.h"
#include "LKRun.h"
#include "LKMCStep.h"
#include "TPCDrum.h"
#include "STDSiArray.h"
#include "TClonesArray.h"
#include "TClass.h"
#include "TFile.h"
#include "TTree.h"
#include "TError.h"
#include <algorithm>
#include <cmath>
#include <iostream>
#include <map>
#include <set>
#include <stdexcept>
#include <utility>

ClassImp(STDSiEdepSumTask);

namespace {
void RequireSiEdep(bool condition, const std::string& message)
{
    if (!condition) throw std::runtime_error("STDSiEdepSumTask: " + message);
}
}

STDSiEdepSumTask::STDSiEdepSumTask()
    : LKTask("STDSiEdepSumTask", "Sum Si MC energy deposits in MeV without waveform response")
{
}

bool STDSiEdepSumTask::Init()
{
    try {
        RequireSiEdep(!fStarted, "use a new task instance for each run");
        fStarted = true;
        RequireSiEdep(fRun && fRun->GetInputTree() && fRun->GetOutputFile() &&
                      fRun->GetOutputTree(), "input/output file and event tree are required");
        fEvents = fRun->GetOutputTree();
        RequireSiEdep(fRun->GetOutputFile()->IsWritable() && fEvents->GetEntries() == 0,
                      "use a NEW writable output with no saved events");

        auto* detector = dynamic_cast<TPCDrum*>(fRun->GetDetector());
        RequireSiEdep(detector != nullptr, "AddDetector(new TPCDrum()) is required");
        auto* si = dynamic_cast<STDSiArray*>(detector->GetDetectorPlane(1));
        RequireSiEdep(si && si->GetSiNum() > 0, "Si detector plane is missing");

        // Resolve the exact detector branch names from the initialized geometry.
        // A present, empty MCStep array means no deposit; a MISSING array is an
        // input error and must never be silently interpreted as zero energy.
        std::set<std::string> seen;
        for (int det = 0; det < si->GetSiNum(); ++det) {
            const TString name = si->GetSiName(det);
            const TString branchName = "MCStep" + name;
            RequireSiEdep(!name.IsNull() && seen.insert(name.Data()).second,
                          "empty or duplicate Si detector name");
            RequireSiEdep(fRun->GetInputTree()->GetBranch(branchName.Data()) != nullptr,
                          "missing input " + std::string(branchName.Data()) +
                          "; use the original subset MC, not a digi without Si MCStep branches");
            auto* steps = fRun->GetBranchA(branchName, false);
            RequireSiEdep(steps && steps->GetClass() &&
                          steps->GetClass()->InheritsFrom(LKMCStep::Class()),
                          "input " + std::string(branchName.Data()) + " is not LKMCStep[]");
            fSiSteps.push_back(steps);
            fDetectorNames.emplace_back(name.Data());
        }
        fDetectorMeV.assign(fSiSteps.size(), 0.);
        fPositiveStepCount.assign(fSiSteps.size(), 0);

        const char* branches[] = {
            "EdepSumSiInputEntry", "EdepSumSiTotalMeV", "EdepSumSiDetectorNames",
            "EdepSumSiMeV", "EdepSumSiStepCount", "EdepSumSiTrackDetID",
            "EdepSumSiTrackID", "EdepSumSiTrackMeV"
        };
        for (const char* name : branches)
            RequireSiEdep(!fEvents->GetBranch(name) && !fRun->GetInputTree()->GetBranch(name),
                          std::string("branch already exists: ") + name);

        RequireSiEdep(fEvents->Branch(branches[0], &fInputEntry, "EdepSumSiInputEntry/L") != nullptr,
                      "cannot create input-entry branch");
        RequireSiEdep(fEvents->Branch(branches[1], &fTotalMeV, "EdepSumSiTotalMeV/D") != nullptr,
                      "cannot create total-energy branch");
        RequireSiEdep(fEvents->Branch(branches[2], &fDetectorNames) != nullptr,
                      "cannot create detector-name branch");
        RequireSiEdep(fEvents->Branch(branches[3], &fDetectorMeV) != nullptr,
                      "cannot create detector-energy branch");
        RequireSiEdep(fEvents->Branch(branches[4], &fPositiveStepCount) != nullptr,
                      "cannot create positive-step-count branch");
        RequireSiEdep(fEvents->Branch(branches[5], &fTrackDetID) != nullptr &&
                      fEvents->Branch(branches[6], &fTrackID) != nullptr &&
                      fEvents->Branch(branches[7], &fTrackMeV) != nullptr,
                      "cannot create per-detector/per-track energy branches");

        fReady = true;
        std::cout << "[STDSiEdepSumTask] " << fSiSteps.size()
                  << " Si detectors; raw MCStep Edep in MeV, no ADC/smearing/threshold\n";
        return true;
    } catch (const std::exception& error) {
        ::Error(GetName(), "%s", error.what());
        return false;
    }
}

void STDSiEdepSumTask::Exec(Option_t*)
{
    RequireSiEdep(fReady && !fFinished, "invalid task lifecycle");
    fInputEntry = fRun->GetCurrentEventID();
    RequireSiEdep(fInputEntry >= 0, "invalid input event ID");
    fTotalMeV = 0.;
    std::fill(fDetectorMeV.begin(), fDetectorMeV.end(), 0.);
    std::fill(fPositiveStepCount.begin(), fPositiveStepCount.end(), 0);
    fTrackDetID.clear();
    fTrackID.clear();
    fTrackMeV.clear();

    std::map<std::pair<int, int>, double> byTrack;
    for (std::size_t det = 0; det < fSiSteps.size(); ++det) {
        const auto* steps = fSiSteps[det];
        for (int index = 0; index < steps->GetEntriesFast(); ++index) {
            const auto* step = dynamic_cast<const LKMCStep*>(steps->At(index));
            const std::string where = " at input entry " + std::to_string(fInputEntry) +
                                      ", " + fDetectorNames[det] + ", step " + std::to_string(index);
            RequireSiEdep(step != nullptr, "null/wrong MCStep element" + where);
            const double energy = step->GetEdep(); // Geant4/LILAK numerical MeV
            RequireSiEdep(std::isfinite(energy) && energy >= 0., "invalid deposited energy" + where);
            if (energy == 0.) continue;
            const int trackID = step->GetTrackID();
            RequireSiEdep(trackID > 0, "nonpositive MC track ID for positive energy" + where);

            // One physical deposit is counted exactly once. No ohmic/junction
            // sharing, *1e6 conversion, ADC scaling, clipping, or Gaussian noise.
            fDetectorMeV[det] += energy;
            fTotalMeV += energy;
            double& trackEnergy = byTrack[{static_cast<int>(det), trackID}];
            trackEnergy += energy;
            RequireSiEdep(std::isfinite(fDetectorMeV[det]) && std::isfinite(fTotalMeV) &&
                          std::isfinite(trackEnergy), "energy-sum overflow" + where);
            ++fPositiveStepCount[det];
        }
    }
    for (const auto& value : byTrack) {
        fTrackDetID.push_back(value.first.first);
        fTrackID.push_back(value.first.second);
        fTrackMeV.push_back(value.second);
    }
    // Deliberately NO TTree::Fill(): LKRun commits all event branches together.
    // Therefore an event omitted by another task is also omitted here.
}

bool STDSiEdepSumTask::EndOfRun()
{
    if (fFinished) return true;
    if (!fReady || !fEvents) return false;
    fFinished = true;
    std::cout << "[STDSiEdepSumTask] event rows with Si Edep: " << fEvents->GetEntries()
              << "; file writing is handled by LKRun\n";
    return true;
}
