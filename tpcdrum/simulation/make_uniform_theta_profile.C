
void make_uniform_theta_profile(
    const char* outputName = "uniform_theta_profile.root",
    double thetaMinDeg = -24.0,
    double thetaMaxDeg = +24.0,
    double phiMinDeg = -24.0,
    double phiMaxDeg = +24.0,
    double energyMinMeV = 1.0,
    double energyMaxMeV = 45.0,
    double meanPadXMM = 0.0,
    double sigmaPadXMM = 1.0,
    double meanPadZMM = 128.8,
    double sigmaPadZMM = 5.0)
{
    const double thetaMinRad = thetaMinDeg * TMath::DegToRad();
    const double thetaMaxRad = thetaMaxDeg * TMath::DegToRad();
    const double phiMinRad = phiMinDeg * TMath::DegToRad();
    const double phiMaxRad = phiMaxDeg * TMath::DegToRad();

    // TPC-Drum coordinate conversion:
    //   x_pad = -x_G4
    //   z_pad =  y_G4 + 128.8 mm
    const double meanG4XMM = -meanPadXMM;
    const double meanG4YMM = meanPadZMM - 128.8;

    const int nEnergyBins = 440;
    const int nPositionBins = 200;
    const int nAngleBins = 480;
    const double nSigmaRange = 5.0;

    TFile output(outputName, "RECREATE");
    if (output.IsZombie()) {
        std::cerr << "ERROR: cannot create " << outputName << std::endl;
        return;
    }

    TH1F energy(
        "Energy",
        "Uniform alpha kinetic energy;E_{#alpha} [MeV];weight",
        nEnergyBins, energyMinMeV, energyMaxMeV);
    for (int iEnergy = 1; iEnergy <= nEnergyBins; ++iEnergy)
        energy.SetBinContent(iEnergy, 1.0);

    TH2F xThetaX(
        "XThetaX",
        "Beam x and #theta_{X};x_{G4} [mm];#theta_{X}^{NPTool} [rad]",
        nPositionBins,
        meanG4XMM - nSigmaRange * sigmaPadXMM,
        meanG4XMM + nSigmaRange * sigmaPadXMM,
        nAngleBins, thetaMinRad, thetaMaxRad);

    for (int iX = 1; iX <= nPositionBins; ++iX) {
        const double x = xThetaX.GetXaxis()->GetBinCenter(iX);
        const double dx = (x - meanG4XMM) / sigmaPadXMM;
        const double gaussianWeight = std::exp(-0.5 * dx * dx);
        for (int iTheta = 1; iTheta <= nAngleBins; ++iTheta)
            xThetaX.SetBinContent(iX, iTheta, gaussianWeight);
    }

    TH2F yPhiY(
        "YPhiY",
        "Beam y and #phi_{Y};y_{G4} [mm];#phi_{Y}^{NPTool} [rad]",
        nPositionBins,
        meanG4YMM - nSigmaRange * sigmaPadZMM,
        meanG4YMM + nSigmaRange * sigmaPadZMM,
        nAngleBins, phiMinRad, phiMaxRad);

    for (int iY = 1; iY <= nPositionBins; ++iY) {
        const double y = yPhiY.GetXaxis()->GetBinCenter(iY);
        const double dy = (y - meanG4YMM) / sigmaPadZMM;
        const double gaussianWeight = std::exp(-0.5 * dy * dy);
        for (int iPhi = 1; iPhi <= nAngleBins; ++iPhi)
            yPhiY.SetBinContent(iY, iPhi, gaussianWeight);
    }

    energy.Write();
    xThetaX.Write();
    yPhiY.Write();
    output.Close();

    std::cout << "Created: " << outputName << '\n'
              << "Energy: uniform [" << energyMinMeV << ", "
              << energyMaxMeV << "] MeV\n"
              << "ThetaX: uniform [" << thetaMinDeg << ", "
              << thetaMaxDeg << "] deg\n"
              << "PhiY: uniform [" << phiMinDeg << ", "
              << phiMaxDeg << "] deg\n"
              << "pad x: Gaussian mean=" << meanPadXMM
              << " mm, sigma=" << sigmaPadXMM << " mm\n"
              << "pad z: Gaussian mean=" << meanPadZMM
              << " mm, sigma=" << sigmaPadZMM << " mm\n"
              << "pad y: fixed at 0 mm by the Beam generator" << std::endl;
}

