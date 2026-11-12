// Run with libLILAK loaded on original simulation MC.
#include "C12DirectDecayMC.h"
#include "TNamed.h"

int C12_filter_direct_decay_mc(const char* filename, const char* outputName,
                              const char* selection = "direct")
{
    using namespace C12DirectDecayMC;
    const TString mode(selection);
    Status wanted = Direct;
    if (mode == "no_reaction")
        wanted = NoReactionProducts;
    else if (mode == "no_daughters")
        wanted = CarbonWithoutDaughters;
    else if (mode != "direct") {
        std::fprintf(stderr, "Selection must be direct, no_reaction, or no_daughters.\n");
        return 2;
    }

    Input input(filename, true);
    if (!input.event)
        return 2;
    if (wanted != Direct && !input.secondariesConfirmed) {
        std::fprintf(stderr, "This selection requires persistency/MCSecondary true in the input.\n");
        return 2;
    }
    TFile output(outputName, "CREATE");
    if (output.IsZombie())
        return 2;

    auto* selected = input.event->CloneTree(0);
    if (!selected)
        return 2;
    Long64_t sourceEntry = 0, outputEntry = -1;
    int statusCode = Invalid;
    TTree audit("decay_selection", "One selection decision per original event entry");
    audit.Branch("source_entry", &sourceEntry, "source_entry/L");
    audit.Branch("output_entry", &outputEntry, "output_entry/L");
    audit.Branch("status", &statusCode, "status/I");

    std::array<Long64_t, StatusCount> counts{};
    for (; sourceEntry < input.event->GetEntries(); ++sourceEntry) {
        if (!input.Read(sourceEntry))
            return 2;
        Decay decay;
        const Status status = Classify(*input.tracks, decay);
        ++counts[status];
        statusCode = status;
        outputEntry = -1;
        if (status == wanted) {
            outputEntry = selected->GetEntries();
            if (selected->Fill() < 0)
                return 2;
        }
        if (audit.Fill() < 0)
            return 2;
    }
    output.cd();
    for (const char* name : {"ParameterContainer", "ProcessTable", "SensitiveDetectors", "Volumes"}) {
        if (auto* object = input.file.Get(name))
            if (object->Write(name, TObject::kSingleKey) <= 0)
                return 2;
    }
    TNamed("decay_selection_status",
           "0=direct; 1=no saved reaction products; 2=C12 product without saved alpha/Be8 daughters; "
           "3=other or incomplete; 4=ambiguous; 5=invalid. output_entry=-1 means not copied.").Write();
    TNamed("decay_selection_mode", mode.Data()).Write();
    if (selected->Write() <= 0 || audit.Write() <= 0)
        return 2;
    output.Flush();
    if (output.TestBit(TFile::kWriteError))
        return 2;

    PrintCounts(counts);
    std::printf("Copied %lld / %lld events to %s\n", selected->GetEntries(), sourceEntry, outputName);
    return 0;
}
