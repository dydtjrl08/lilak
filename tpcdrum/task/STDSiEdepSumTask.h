#ifndef STD_SI_EDEP_SUM_TASK_H
#define STD_SI_EDEP_SUM_TASK_H

#include "LKTask.h"
#include <string>
#include <vector>

class TClonesArray;
class TTree;

// MC deposited energy only. Does not create Si channels or touch ADC, tags,
// MCStep coordinates, or random-number state. Units are numerical MeV.
class STDSiEdepSumTask : public LKTask
{
public:
    STDSiEdepSumTask();
    ~STDSiEdepSumTask() override = default;
    bool Init() override;
    void Exec(Option_t*) override;
    bool EndOfRun() override;

private:
    std::vector<TClonesArray*> fSiSteps; //!
    TTree* fEvents = nullptr; //!
    bool fStarted = false; //!
    bool fReady = false; //!
    bool fFinished = false; //!

    // These values are persisted by branches in LKRun's event tree, not by
    // streaming the task object. All branch names start with EdepSum because
    // older LKRun readers explicitly skip that prefix for non-TClonesArray data.
    Long64_t fInputEntry = -1; //!
    Double_t fTotalMeV = 0.; //!
    std::vector<std::string> fDetectorNames; //!
    std::vector<double> fDetectorMeV; //!
    std::vector<int> fPositiveStepCount; //!
    std::vector<int> fTrackDetID; //!
    std::vector<int> fTrackID; //!
    std::vector<double> fTrackMeV; //!

    ClassDefOverride(STDSiEdepSumTask, 1);
};

#endif
