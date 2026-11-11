// Load libLILAK in ROOT before running this macro on the original simulation MC.
#include "TCanvas.h"
#include "TClonesArray.h"
#include "TFile.h"
#include "TH1D.h"
#include "TH2D.h"
#include "TLorentzVector.h"
#include "TMath.h"
#include "TTree.h"
#include "TUUID.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>
#include <map>
#include <vector>

namespace {
bool FiniteVector(const TVector3& p)
{
    return std::isfinite(p.X()) && std::isfinite(p.Y()) && std::isfinite(p.Z());
}
}

// Returns 0 for PASS, 1 for unequal geometry, and 2 for an incomplete check.
int C12_check_equal_energy_mc(const char* filename = "C12_HoyleDirect_100.root")
{
    constexpr int alphaPDG = 1000020040;
    constexpr int carbon12PDGWithoutIsomer = 100006012;
    constexpr double vertexToleranceMm = 1.e-3;
    constexpr double angleToleranceDeg = 1.e-3;
    constexpr double fractionTolerance = 1.e-6;

    TFile file(filename, "READ");
    auto* event = dynamic_cast<TTree*>(file.Get("event"));
    TClonesArray* tracks = nullptr;
    if (file.IsZombie() || file.TestBit(TFile::kRecovered) || file.Get("HoyleMeta") ||
        !event || !event->GetBranch("MCTrack")) {
        std::fprintf(stderr, "INCONCLUSIVE: use a complete, original MC file with event/MCTrack.\n");
        return 2;
    }
    event->SetBranchStatus("*", 0);
    event->SetBranchStatus("MCTrack*", 1);
    if (event->SetBranchAddress("MCTrack", &tracks) < 0) {
        std::fprintf(stderr, "INCONCLUSIVE: cannot read MCTrack. Check the loaded LILAK library.\n");
        return 2;
    }

    // Odd bins place the equal-energy point at the center of a Dalitz bin.
    TH2D dalitz("h_hoyle_dalitz",
                "Dalitz plot (COM);(#epsilon_{2}-#epsilon_{1})/#sqrt{3};#epsilon_{3}-1/3;Triplets",
                141, -0.7, 0.7, 141, -0.7, 0.7);
    TH1D openingAngles("h_hoyle_com_opening",
                       "Alpha-pair opening angles (COM);Opening angle [deg];Alpha pairs",
                       181, -0.5, 180.5);
    dalitz.SetDirectory(nullptr);
    dalitz.SetStats(false);
    openingAngles.SetDirectory(nullptr);
    openingAngles.SetStats(false);

    Long64_t selected = 0, noTriplet = 0, ambiguous = 0, invalid = 0, outsideTolerance = 0;
    double maxAngleError = 0., maxFractionError = 0.;
    for (Long64_t entry = 0; entry < event->GetEntries(); ++entry) {
        if (event->GetEntry(entry) <= 0 || !tracks) {
            ++invalid;
            continue;
        }

        std::map<int, const LKMCTrack*> byId;
        std::map<int, std::vector<const LKMCTrack*>> alphasByParent;
        bool bad = false;
        for (int i = 0; i < tracks->GetEntriesFast(); ++i) {
            const auto* track = dynamic_cast<const LKMCTrack*>(tracks->At(i));
            if (!track || track->GetTrackID() <= 0 || track->GetNumVertices() < 1 ||
                !byId.emplace(track->GetTrackID(), track).second) {
                bad = true;
                break;
            }
            if (track->GetPDG() == alphaPDG)
                alphasByParent[track->GetParentID()].push_back(track);
        }
        if (bad) {
            ++invalid;
            continue;
        }

        // A recoil alpha has a different parent and cannot enter the triplet.
        const std::vector<const LKMCTrack*>* triplet = nullptr;
        for (const auto& family : alphasByParent) {
            const auto parent = byId.find(family.first);
            if (parent == byId.end() ||
                parent->second->GetPDG() / 10 != carbon12PDGWithoutIsomer ||
                family.second.size() != 3)
                continue;

            if (triplet) {
                ++ambiguous;
                bad = true;
                break;
            }
            triplet = &family.second;
        }
        if (bad)
            continue;
        if (!triplet) {
            ++noTriplet;
            continue;
        }

        // Use the same fixed track-ID labeling as the Dalitz analysis.
        auto alphas = *triplet;
        std::sort(alphas.begin(), alphas.end(), [](const LKMCTrack* a, const LKMCTrack* b) {
            return a->GetTrackID() < b->GetTrackID();
        });
        std::array<TVector3, 3> momentum;
        std::array<double, 3> kinetic;
        std::vector<double> masses;
        const auto vertex = alphas.front()->GetPrimaryPosition();
        for (int a = 0; a < 3; ++a) {
            const auto& alpha = *alphas[a];
            const auto position = alpha.GetPrimaryPosition();
            momentum[a].SetXYZ(alpha.GetPX(0), alpha.GetPY(0), alpha.GetPZ(0));
            kinetic[a] = alpha.GetEnergy(0);
            if (!FiniteVector(position) || (position - vertex).Mag() > vertexToleranceMm ||
                !FiniteVector(momentum[a]) || !std::isfinite(kinetic[a]) || kinetic[a] < 0.) {
                bad = true;
                break;
            }
            // Infer the simulated nuclear mass from the saved momentum and kinetic energy.
            if (kinetic[a] > 1.e-9)
                masses.push_back(0.5 * (momentum[a].Mag2() / kinetic[a] - kinetic[a]));
        }
        if (bad || masses.empty()) {
            ++invalid;
            continue;
        }
        std::sort(masses.begin(), masses.end());
        const double mass = masses[masses.size() / 2];
        if (!std::isfinite(mass) || std::abs(mass - 3727.3794) > 0.1) {
            ++invalid;
            continue;
        }

        std::array<TLorentzVector, 3> four;
        TLorentzVector total;
        for (int a = 0; a < 3; ++a) {
            four[a].SetPxPyPzE(momentum[a].X(), momentum[a].Y(), momentum[a].Z(), mass + kinetic[a]);
            total += four[a];
        }
        const auto beta = total.BoostVector();
        if (!FiniteVector(beta) || beta.Mag2() >= 1.) {
            ++invalid;
            continue;
        }

        double sumKinetic = 0.;
        for (int a = 0; a < 3; ++a) {
            four[a].Boost(-beta);
            kinetic[a] = four[a].E() - mass;
            sumKinetic += kinetic[a];
            if (!FiniteVector(four[a].Vect()) || four[a].P() <= 0. ||
                !std::isfinite(kinetic[a]) || kinetic[a] < -1.e-8)
                bad = true;
        }
        if (bad || !std::isfinite(sumKinetic) || sumKinetic <= 0.) {
            ++invalid;
            continue;
        }

        const std::array<double, 3> fraction{
            kinetic[0] / sumKinetic, kinetic[1] / sumKinetic, kinetic[2] / sumKinetic
        };
        dalitz.Fill((fraction[1] - fraction[0]) / std::sqrt(3.), fraction[2] - 1. / 3.);

        double angleError = 0., fractionError = 0.;
        for (int a = 0; a < 3; ++a) {
            const auto p = four[a].Vect();
            const auto q = four[(a + 1) % 3].Vect();
            const double angle = std::atan2(p.Cross(q).Mag(), p.Dot(q)) * TMath::RadToDeg();
            openingAngles.Fill(angle);
            angleError = std::max(angleError, std::abs(angle - 120.));
            fractionError = std::max(fractionError, std::abs(fraction[a] - 1. / 3.));
        }
        ++selected;
        if (angleError > angleToleranceDeg || fractionError > fractionTolerance)
            ++outsideTolerance;
        maxAngleError = std::max(maxAngleError, angleError);
        maxFractionError = std::max(maxFractionError, fractionError);
    }
    event->ResetBranchAddresses();
    delete tracks;

    std::printf("Entries=%lld | triplets=%lld | no triplet=%lld | ambiguous=%lld | invalid=%lld\n",
                event->GetEntries(), selected, noTriplet, ambiguous, invalid);
    if (selected > 0) {
        const TString canvasName = TString("c_hoyle_dde_") + TUUID().AsString();
        auto* canvas = new TCanvas(canvasName, "Hoyle DDE check", 1150, 520);
        canvas->Divide(2, 1);
        canvas->cd(1)->SetRightMargin(0.16);
        dalitz.DrawCopy("COLZ");
        canvas->cd(2);
        openingAngles.SetLineColor(kBlue + 1);
        openingAngles.SetLineWidth(2);
        openingAngles.DrawCopy("HIST");
        canvas->Modified();
        canvas->Update();

        std::printf("Max COM |angle-120|=%.6g deg | max |T_i/sum(T)-1/3|=%.6g\n",
                    maxAngleError, maxFractionError);
    }
    if (outsideTolerance > 0) {
        std::printf("FAIL: %lld triplets outside the DDE tolerances.\n", outsideTolerance);
        return 1;
    }
    if (selected == 0 || ambiguous > 0 || invalid > 0) {
        std::printf("INCONCLUSIVE: no evaluable triplets or incomplete event records.\n");
        return 2;
    }
    std::printf("PASS: all %lld checked triplets have equal COM energies and 120-degree angles.\n", selected);
    return 0;
}
