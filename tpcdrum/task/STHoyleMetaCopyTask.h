#ifndef ST_HOYLE_META_COPY_TASK_H
#define ST_HOYLE_META_COPY_TASK_H

#include "LKTask.h"
class TFile;
class TTree;
class TBranch;

// Copies provenance, not detector response. One input ROOT file per run.
// Input must already contain an entry-aligned HoyleMeta tree.
class STHoyleMetaCopyTask : public LKTask
{
public:
    STHoyleMetaCopyTask();
    ~STHoyleMetaCopyTask() override;
    bool Init() override;
    void Exec(Option_t*) override;
    bool EndOfRun() override;

private:
    TFile* fMetadataFile = nullptr; //!
    TTree* fMetadata = nullptr; //!
    TTree* fEvents = nullptr; //!
    TBranch* fSourceBranch = nullptr; //!
    TBranch* fEntryBranch = nullptr; //!
    Long64_t fInputEntry = -1; //!
    Long64_t fSourceEntry = -1; //!
    bool fReady = false; //!
    bool fFinished = false; //!

    ClassDefOverride(STHoyleMetaCopyTask, 1);
};
#endif
