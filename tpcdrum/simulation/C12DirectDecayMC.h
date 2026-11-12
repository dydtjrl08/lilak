#ifndef C12_DIRECT_DECAY_MC_H
#define C12_DIRECT_DECAY_MC_H

// LKMCTrack and LKParameterContainer are provided by the loaded LILAK dictionary.
#include "TClonesArray.h"
#include "TFile.h"
#include "TMath.h"
#include "TTree.h"
#include "TVector3.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>
#include <map>
#include <vector>

namespace C12DirectDecayMC {

constexpr int alphaPDG = 1000020040;
constexpr int carbon12PDG = 100006012;
constexpr int beryllium8PDG = 100004008;
constexpr double vertexToleranceMm = 1.e-3;

enum Status { Direct, NoReactionProducts, CarbonWithoutDaughters, OtherOrIncomplete,
              Ambiguous, Invalid, StatusCount };

constexpr const char* statusLabels[StatusCount] = {
    "direct", "no saved reaction products", "C12 product without saved alpha/Be8 daughters",
    "other or incomplete", "ambiguous", "invalid"
};

inline bool Finite(const TVector3& v)
{
    return std::isfinite(v.X()) && std::isfinite(v.Y()) && std::isfinite(v.Z());
}

struct Decay {
    int parentId = 0;
    double zMm = 0.;
    std::array<int, 3> trackId{};
    std::array<double, 3> energyMeV{}, thetaDeg{}, pzMeVc{};
};

inline Status Classify(const TClonesArray& tracks, Decay& decay)
{
    std::map<int, const LKMCTrack*> byId;
    std::map<int, std::vector<const LKMCTrack*>> alphasByParent;
    int beamTracks = 0;
    bool carbonProduct = false, otherNuclearTrack = false;
    for (int i = 0; i < tracks.GetEntriesFast(); ++i) {
        const auto* track = dynamic_cast<const LKMCTrack*>(tracks.At(i));
        if (!track || track->GetTrackID() <= 0 || track->GetNumVertices() < 1 ||
            !byId.emplace(track->GetTrackID(), track).second)
            return Invalid;
        const int pdg = track->GetPDG();
        if (pdg / 10 == carbon12PDG) {
            if (track->GetParentID() == 0)
                ++beamTracks;
            else
                carbonProduct = true;
        } else if (std::abs(static_cast<long long>(pdg)) >= 100) {
            otherNuclearTrack = true;
        }
        if (pdg == alphaPDG)
            alphasByParent[track->GetParentID()].push_back(track);
    }

    const std::vector<const LKMCTrack*>* triplet = nullptr;
    for (const auto& family : alphasByParent) {
        const auto parent = byId.find(family.first);
        if (parent == byId.end() || parent->second->GetPDG() / 10 != carbon12PDG ||
            family.second.size() != 3)
            continue;
        if (triplet)
            return Ambiguous;
        triplet = &family.second;
    }

    if (!triplet) {
        if (carbonProduct) {
            for (const auto& item : byId) {
                const auto* track = item.second;
                if (track->GetPDG() != alphaPDG && track->GetPDG() / 10 != beryllium8PDG)
                    continue;
                const auto parent = byId.find(track->GetParentID());
                if (parent == byId.end() ||
                    (parent->second->GetPDG() / 10 == carbon12PDG &&
                     parent->second->GetParentID() != 0))
                    return OtherOrIncomplete;
            }
            return CarbonWithoutDaughters;
        }
        return beamTracks == 1 && !otherNuclearTrack ? NoReactionProducts : OtherOrIncomplete;
    }

    auto alphas = *triplet;
    std::sort(alphas.begin(), alphas.end(), [](const LKMCTrack* a, const LKMCTrack* b) {
        return a->GetTrackID() < b->GetTrackID();
    });
    const auto vertex = alphas.front()->GetPrimaryPosition();
    if (!Finite(vertex))
        return Invalid;
    decay.zMm = vertex.Z();
    decay.parentId = alphas.front()->GetParentID();
    for (const auto& item : byId) {
        const auto* track = item.second;
        const int pdg = track->GetPDG();
        if (track->GetParentID() == decay.parentId && pdg != alphaPDG &&
            std::abs(static_cast<long long>(pdg)) >= 100 &&
            (track->GetPrimaryPosition() - vertex).Mag() <= vertexToleranceMm)
            return OtherOrIncomplete;
    }
    for (int i = 0; i < 3; ++i) {
        const auto& alpha = *alphas[i];
        const auto position = alpha.GetPrimaryPosition();
        if (!Finite(position))
            return Invalid;
        for (int j = 0; j < i; ++j)
            if ((position - alphas[j]->GetPrimaryPosition()).Mag() > vertexToleranceMm)
                return OtherOrIncomplete;

        const TVector3 p(alpha.GetPX(0), alpha.GetPY(0), alpha.GetPZ(0));
        const double kinetic = alpha.GetEnergy(0);
        if (!Finite(p) || !(p.Mag2() > 0.) || !std::isfinite(kinetic) || kinetic <= 0.)
            return Invalid;
        decay.trackId[i] = alpha.GetTrackID();
        decay.energyMeV[i] = kinetic;
        decay.pzMeVc[i] = p.Z();
        // The beamline is the undirected Geant4 world Z axis.
        decay.thetaDeg[i] = std::atan2(std::hypot(p.X(), p.Y()), std::abs(p.Z())) *
                            TMath::RadToDeg();
    }
    return Direct;
}

struct Input {
    TFile file;
    TTree* event = nullptr;
    TClonesArray* tracks = nullptr;
    bool secondariesConfirmed = false;

    Input(const char* filename, bool allBranches) : file(filename, "READ")
    {
        if (file.IsZombie() || file.TestBit(TFile::kRecovered) || file.GetKey("HoyleMeta")) {
            std::fprintf(stderr, "Use a complete original simulation MC ROOT file.\n");
            return;
        }
        const auto* parameters = dynamic_cast<const LKParameterContainer*>(file.Get("ParameterContainer"));
        if (parameters && parameters->CheckPar("persistency/MCSecondary")) {
            secondariesConfirmed = parameters->GetParBool("persistency/MCSecondary");
            if (!secondariesConfirmed) {
                std::fprintf(stderr, "The MC file was written with persistency/MCSecondary false.\n");
                return;
            }
        } else {
            std::fprintf(stderr, "MCSecondary setting unavailable. Only recorded decays can be selected.\n");
        }

        auto* tree = dynamic_cast<TTree*>(file.Get("event"));
        if (!tree || !tree->GetBranch("MCTrack")) {
            std::fprintf(stderr, "Missing event/MCTrack in the MC ROOT file.\n");
            return;
        }
        if (!allBranches) {
            tree->SetBranchStatus("*", 0);
            tree->SetBranchStatus("MCTrack*", 1);
        }
        if (tree->SetBranchAddress("MCTrack", &tracks) < 0) {
            std::fprintf(stderr, "Cannot read MCTrack. Check that libLILAK is loaded.\n");
            return;
        }
        event = tree;
    }

    ~Input()
    {
        if (event)
            event->ResetBranchAddresses();
        delete tracks;
    }

    Input(const Input&) = delete;
    Input& operator=(const Input&) = delete;

    bool Read(Long64_t entry)
    {
        if (event->GetEntry(entry) > 0 && tracks)
            return true;
        std::fprintf(stderr, "Could not read MC entry %lld.\n", entry);
        return false;
    }
};

inline void PrintCounts(const std::array<Long64_t, StatusCount>& counts)
{
    for (int i = 0; i < StatusCount; ++i)
        std::printf("  %s: %lld\n", statusLabels[i], counts[i]);
}

}

#endif
