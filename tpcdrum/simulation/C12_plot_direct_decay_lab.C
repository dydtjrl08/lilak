// Run with libLILAK loaded on simulation MC (event/MCTrack).
#include "C12DirectDecayMC.h"
#include "TCanvas.h"
#include "TH2D.h"
#include "TNamed.h"
#include "TSystem.h"
#include "TUUID.h"

#include <memory>

int C12_plot_direct_decay_lab(const char* filename, const char* outputPrefix = "C12_direct_lab")
{
    using namespace C12DirectDecayMC;
    Input input(filename, false);
    if (!input.event)
        return 2;

    const TString prefix(outputPrefix);
    const TString rootName = prefix + ".root";
    const TString zPlotName = prefix + "_energy_vs_z.pdf";
    const TString anglePlotName = prefix + "_theta_vs_energy.pdf";
    if (prefix.IsNull() || !gSystem->AccessPathName(rootName) ||
        !gSystem->AccessPathName(zPlotName) || !gSystem->AccessPathName(anglePlotName)) {
        std::fprintf(stderr, "Choose a nonempty output prefix whose ROOT and PDF files do not exist.\n");
        return 2;
    }

    TH2D energyVsZ("h_alpha_energy_vs_z",
                   "Alpha energy at production;Geant4 decay z [mm];Alpha lab kinetic energy [MeV];Alphas / bin",
                   200, -400., 400., 160, 0., 80.);
    TH2D thetaVsEnergy("h_alpha_theta_vs_energy",
                       "Alpha angle to beamline;Alpha lab kinetic energy [MeV];Beamline angle [deg];Alphas / bin",
                       160, 0., 80., 181, 0., 90.5);
    energyVsZ.SetDirectory(nullptr);
    thetaVsEnergy.SetDirectory(nullptr);
    energyVsZ.SetStats(false);
    thetaVsEnergy.SetStats(false);

    TFile output(rootName, "CREATE");
    if (output.IsZombie())
        return 2;
    Decay decay;
    Long64_t entry = 0;
    TTree decays("decays", "One row per recorded direct decay, alpha arrays ordered by track ID");
    decays.Branch("source_entry", &entry, "source_entry/L");
    decays.Branch("parent_id", &decay.parentId, "parent_id/I");
    decays.Branch("vertex_z_mm", &decay.zMm, "vertex_z_mm/D");
    decays.Branch("track_id", decay.trackId.data(), "track_id[3]/I");
    decays.Branch("energy_mev", decay.energyMeV.data(), "energy_mev[3]/D");
    decays.Branch("theta_deg", decay.thetaDeg.data(), "theta_deg[3]/D");
    decays.Branch("pz_mev_c", decay.pzMeVc.data(), "pz_mev_c[3]/D");

    std::array<Long64_t, StatusCount> counts{};
    Long64_t backwards = 0, outsideZ = 0, outsideEnergy = 0;
    double maxEnergy = 0., maxTheta = 0.;
    for (; entry < input.event->GetEntries(); ++entry) {
        if (!input.Read(entry))
            return 2;
        const Status status = Classify(*input.tracks, decay);
        ++counts[status];
        if (status != Direct)
            continue;
        if (decays.Fill() < 0)
            return 2;
        for (int i = 0; i < 3; ++i) {
            const double energy = decay.energyMeV[i];
            const double theta = decay.thetaDeg[i];
            energyVsZ.Fill(decay.zMm, energy);
            thetaVsEnergy.Fill(energy, theta);
            backwards += decay.pzMeVc[i] < 0.;
            outsideZ += decay.zMm < -400. || decay.zMm >= 400.;
            outsideEnergy += energy >= 80.;
            maxEnergy = std::max(maxEnergy, energy);
            maxTheta = std::max(maxTheta, theta);
        }
    }

    std::printf("Read %lld MC entries\n", entry);
    PrintCounts(counts);
    std::printf("%lld alphas in each histogram | pz < 0: %lld | z overflow/underflow: %lld | energy overflow: %lld\n",
                3 * counts[Direct], backwards, outsideZ, outsideEnergy);
    output.cd();
    TNamed("beamline_angle", "Unsigned angle to Geant4 world Z: atan2(hypot(px,py),abs(pz)).").Write();
    if (energyVsZ.Write() <= 0 || thetaVsEnergy.Write() <= 0 || decays.Write() <= 0)
        return 2;
    decays.SetDirectory(nullptr);
    output.Flush();
    if (output.TestBit(TFile::kWriteError))
        return 2;
    output.Close();
    if (!counts[Direct]) {
        std::fprintf(stderr, "No direct decays selected. Empty histograms saved in %s.\n", rootName.Data());
        return 1;
    }

    // Zoom only the PDF views; the saved histograms keep their full ranges.
    const double energyLimit = std::min(80., 5. * (1. + std::ceil(maxEnergy / 5.)));
    const double thetaLimit = std::min(90.5, 5. * (1. + std::ceil(maxTheta / 5.)));
    energyVsZ.GetYaxis()->SetRangeUser(0., energyLimit);
    thetaVsEnergy.GetXaxis()->SetRangeUser(0., energyLimit);
    thetaVsEnergy.GetYaxis()->SetRangeUser(0., thetaLimit);

    const TString canvasTag(TUUID().AsString());
    auto zCanvas = std::make_unique<TCanvas>(TString("c_c12_lab_z_") + canvasTag,
                                             "Alpha energy versus decay z", 960, 640);
    zCanvas->SetRightMargin(0.17);
    zCanvas->SetLogz();
    energyVsZ.DrawCopy("COLZ");
    zCanvas->SaveAs(zPlotName);

    auto angleCanvas = std::make_unique<TCanvas>(TString("c_c12_lab_theta_") + canvasTag,
                                                 "Alpha angle versus energy", 960, 640);
    angleCanvas->SetRightMargin(0.17);
    thetaVsEnergy.DrawCopy("COLZ");
    angleCanvas->SaveAs(anglePlotName);
    if (gSystem->AccessPathName(zPlotName) || gSystem->AccessPathName(anglePlotName)) {
        std::fprintf(stderr, "Could not write both PDF plots.\n");
        return 2;
    }
    if (!zCanvas->IsBatch()) {
        zCanvas.release();
        angleCanvas.release();
    }
    std::printf("Saved %s, %s, %s\n", rootName.Data(), zPlotName.Data(), anglePlotName.Data());
    return 0;
}
