#include "TPCDrumConstruction.h"
#include "../simulation/hoyle_equal_energy_20261026/G4HoyleEqualEnergyModel.hh"
#include "NPInputParser.h"
#include "NPOptionManager.h"

#include <stdexcept>
#include <string>
#include <vector>

TPCDrumConstruction::TPCDrumConstruction()
{
}

G4VPhysicalVolume* TPCDrumConstruction::Construct()
{
    auto runManager = (LKG4RunManager *) G4RunManager::GetRunManager();
    fPar = runManager -> GetParameterContainer();

    const bool hoyleEqualEnergy = fPar->CheckPar("TPCDrum/HoyleEqualEnergy") &&
                                  fPar->GetParBool("TPCDrum/HoyleEqualEnergy");
    if (hoyleEqualEnergy) {
        NPL::InputParser reaction(NPOptionManager::getInstance()->GetReactionFile());
        const auto decays = reaction.GetAllBlocksWithToken("Decay");
        const std::vector<std::string> alphaDaughters{"4He", "4He", "4He"};
        if (decays.size() != 1 || decays.front()->GetMainValue() != "12C" ||
            !decays.front()->HasToken("Daughter") ||
            decays.front()->GetVectorString("Daughter") != alphaDaughters) {
            throw std::runtime_error(
                "TPCDrum/HoyleEqualEnergy true requires exactly one active Decay block: "
                "Decay 12C with Daughter = 4He 4He 4He. "
                "Set HoyleEqualEnergy false to use the NPTool decay channels in the reaction file.");
        }
    }

    fTPCDrum = new TPCDrum();

    // =========================== world ================================ 
    G4LogicalVolume *logicWorld = GetWorld();
    G4PVPlacement *physWorld = new G4PVPlacement(0, G4ThreeVector(0., 0., 0.), logicWorld, "World", 0, false, 0, true);


    // ========================== chamber ================================
    G4LogicalVolume* logicChamber = GetTPCDrumChamber();
    logicChamber -> SetVisAttributes(GetColor("WHITE"));
    auto pvpChamber = new G4PVPlacement(0, G4ThreeVector(0.,0.,0.), logicChamber, "TPCDrum", logicWorld, false, 1, true);

    // ========================== Chamber window material =============================
    double windowThickness = fPar -> GetParDouble("TPCDrum/WindowThickness") *mm;
    const double window_shiftBeamLine = (fTPCDrum->fBeamLineLength+windowThickness)*fHalfUnit *mm;

    G4LogicalVolume *logicWindow = GetBeamWindow();
    logicWindow -> SetVisAttributes(GetColor("WHITE", 0.6));
    auto pvpWindow_UpStream = new G4PVPlacement(0, G4ThreeVector(0., 0., -window_shiftBeamLine), logicWindow, "ChamberWindow_upstream", logicWorld, false, 2, true);
    auto pvpWindow_DownStream = new G4PVPlacement(0, G4ThreeVector(0.,0., window_shiftBeamLine), logicWindow, "ChamberWindow_downstream", logicWorld, false, 3, true);

    // ========================== readout PadPlane PCB and Pad ================================
    const double readOutPCB_shiftHeight = (fTPCDrum->fChamberHeight*fHalfUnit - fTPCDrum->fReadoutPCBThickness*fHalfUnit) *mm;

    G4LogicalVolume *logicReadoutPCB = GetReadoutPCB();
    G4LogicalVolume *logicReadoutPad = GetReadoutPadPlane();

    logicReadoutPCB -> SetVisAttributes(GetColor("BLUE"));
    logicReadoutPad -> SetVisAttributes(GetColor("YELLOW", 0.4));

    auto pvpReadoutPCB = new G4PVPlacement(new G4RotationMatrix(0., CLHEP::pi/2, 0.), G4ThreeVector(0., -readOutPCB_shiftHeight, 0.), logicReadoutPCB, "ReadOutPCB", logicChamber, false, 4, true);
    auto pvpReadoutPad = new G4PVPlacement(0, G4ThreeVector(0., -readOutPCB_shiftHeight, 0.), logicReadoutPad, "ReadOutPad", logicChamber, false, 5, true);


    // ========================== Field Cage ================================
    G4VisAttributes* greenColor = GetColor("GREEN", 0.1);

    // FieldCage PCB
    G4LogicalVolume* logicFieldCage_SideWall = GetFieldCageSideWall();
    G4LogicalVolume* logicFieldCage_CathodePCB = GetFieldCageCathode();
    G4LogicalVolume* logicFieldCage_DownStreamPCB = GetFieldCageDownStream();
    G4LogicalVolume* logicFieldCage_UpStreamPCB = GetFieldCageUpStream();

    logicFieldCage_SideWall -> SetVisAttributes(greenColor);
    logicFieldCage_CathodePCB -> SetVisAttributes(greenColor);
    logicFieldCage_DownStreamPCB -> SetVisAttributes(greenColor);
    logicFieldCage_UpStreamPCB -> SetVisAttributes(greenColor);

    double fieldCage_shiftHeight = fTPCDrum->GetFieldCageGeantYPos() *mm;
    double fieldCageHeight = fTPCDrum->fFieldCageHeight *mm;
    double fieldCageLength = fTPCDrum->fFieldCageLength *mm;
    double pcbWidth = fTPCDrum->fPCBThickness *mm;
    auto pvpFieldCage_LeftPCB = new G4PVPlacement(new G4RotationMatrix(0., 0., 0.), G4ThreeVector((pcbWidth+fieldCageLength*fHalfUnit), -fieldCage_shiftHeight, 0.), logicFieldCage_SideWall, "fieldCage_leftPCB", logicChamber, false, 6, true);
    auto pvpFieldCage_RightPCB = new G4PVPlacement(new G4RotationMatrix(0., 0., 0.), G4ThreeVector(-(pcbWidth+fieldCageLength*fHalfUnit), -fieldCage_shiftHeight, 0.), logicFieldCage_SideWall, "fieldCage_RightPCB", logicChamber, false, 7, true);
    auto pvpFieldCage_CathodePCB = new G4PVPlacement(new G4RotationMatrix(0., 0., 0.), G4ThreeVector(0., -fieldCage_shiftHeight+fieldCageHeight*fHalfUnit, 0.), logicFieldCage_CathodePCB, "fieldCage_CathodePCB", logicChamber, false, 8, true);
    auto pvpFieldCage_DownStreamtPCB = new G4PVPlacement(new G4RotationMatrix(0., 0., 0.), G4ThreeVector(0., -fieldCage_shiftHeight, (pcbWidth+fieldCageLength*fHalfUnit)), logicFieldCage_DownStreamPCB, "fieldCage_downstreamPCB", logicChamber, false, 9, true);
    auto pvpFieldCage_UpStreamtPCB = new G4PVPlacement(new G4RotationMatrix(0., 0., 0.), G4ThreeVector(0., -fieldCage_shiftHeight, -(pcbWidth+fieldCageLength*fHalfUnit)), logicFieldCage_UpStreamPCB, "fieldCage_UpstreamPCB", logicChamber, false, 10, true);


    // ========================== Gating Grid ======================================
    if(fPar->CheckPar("TPCDrum/GatingGrid")){
        if(fPar->GetParBool("TPCDrum/GatingGrid")){
            G4LogicalVolume* logicGatingGrid = GetGatingGrid();
            logicGatingGrid -> SetVisAttributes(GetColor("GREEN",0.4));
            double gatingGridHeight = fTPCDrum -> GetGatingGridGeantYPos();
            auto pvpGatingGrid = new G4PVPlacement(new G4RotationMatrix(0., 0., 0.), G4ThreeVector(0., -gatingGridHeight, 0.), logicGatingGrid, "GatingGrid", logicChamber, false, 11, true);
        }
    }

    // ========================== Silicon telespoce ================================
    STDSiArray* siArray = new STDSiArray();
    siArray -> Init();
    for(int i=0; i<siArray->GetSiNum(); i++){
        double siWidth = siArray->GetSiWidth();
        double siHeight = siArray->GetSiHeight();
        double siThickness = siArray->GetSiThickness();
        TString siName = siArray->GetSiName(i);
        G4Box *solidSi = new G4Box(siName.Data(), siWidth *fHalfUnit, siHeight *fHalfUnit, siThickness *fHalfUnit);
        G4Material* metSi = GetSolidMaterial("Si");
        G4LogicalVolume* logicSi = new G4LogicalVolume(solidSi, metSi, siName.Data());
        logicSi -> SetVisAttributes(GetColor("YELLOW", 0.4));

        double x = 0.;
        double y = 0.;
        double z = siArray->GetSiArrayPlaneDistAtPadCenter();
        
        fTPCDrum->GetCoordinateGeantToPad(x, y, z);

        double SiLocalPosX = siArray->GetSiCenterPos(i, 0);
        double SiLocalPosY = y;
        double SiLocalPosZ = siArray->GetSiCenterPos(i, 1);

        fTPCDrum->GetCoordinatePadToGeant(SiLocalPosX, SiLocalPosY, SiLocalPosZ);
        double rotation = (i<6)? CLHEP::pi/2 : 0.;

        auto pvpSi = new G4PVPlacement(new G4RotationMatrix(0., 0., rotation), G4ThreeVector(SiLocalPosX, SiLocalPosY, SiLocalPosZ), logicSi, siName.Data(), logicChamber, false, 12+i, true);
        runManager -> SetSensitiveDetector(pvpSi);
    }

    // ====================== NPTool =======================
    // G4ProductionCuts* ecut = new G4ProductionCuts();
    // double eEnergyThreshold = 10.*keV;
    // ecut->SetProductionCut(eEnergyThreshold,"e-");
    fReactionRegion = new G4Region("NPSimulationProcess");

    fReactionRegion->AddRootLogicalVolume(logicChamber);
    fReactionRegion->SetUserLimits(new G4UserLimits(1. * mm));    

    G4FastSimulationManager* mng = fReactionRegion->GetFastSimulationManager();
    unsigned int size = fReactionModel.size();
    for (unsigned int i = 0; i < size; i++) {
        mng->RemoveFastSimulationModel(fReactionModel[i]);
    }
    fReactionModel.clear();

    G4VFastSimulationModel* fsm;
    fsm = new NPS::BeamReaction("BeamReaction", fReactionRegion);
    fReactionModel.push_back(fsm);
    if (hoyleEqualEnergy)
        fsm = new G4HoyleEqualEnergyModel("HoyleEqualEnergy", fReactionRegion);
    else
        fsm = new NPS::Decay("Decay", fReactionRegion);
    fReactionModel.push_back(fsm);

    runManager -> SetSensitiveDetector(pvpChamber);
    return physWorld;
}

G4SubtractionSolid* TPCDrumConstruction::GetGatingGridVolume(){
    double fieldCageLength = fTPCDrum->fFieldCageLength *mm;
    double gatingGridSupporterWidth = fTPCDrum->fGatingGridSupporterWidth *mm;
    double gatingGridActiveWidth = fTPCDrum->fGatingGridActiveWidth *mm;
    double gatingGridActivePCBWidth = fTPCDrum->fGatingGridActivePCBWidth *mm;
    double PCBThickness = fTPCDrum->fPCBThickness *mm;

    const double GatingGridEmptyAreaWidth = fieldCageLength*fHalfUnit - gatingGridSupporterWidth - gatingGridActiveWidth*fHalfUnit - gatingGridActivePCBWidth;
    const double GatingGridEmptyAreaCenter = GatingGridEmptyAreaWidth*fHalfUnit + gatingGridActiveWidth*fHalfUnit + gatingGridActivePCBWidth;

    G4Box* solidGatingGrid_base = new G4Box("gatingGrid_base", fieldCageLength*fHalfUnit, PCBThickness*fHalfUnit, fieldCageLength*fHalfUnit);
    G4Box* solidGatingGrid_subtEmptyActive = new G4Box("gatingGrid_subtEmptyActive", GatingGridEmptyAreaWidth*fHalfUnit, PCBThickness*fHalfUnit + 1.*mm, (fieldCageLength - 2*gatingGridSupporterWidth)*fHalfUnit);
    G4Box* solidGatingGrid_subtActiveArea = new G4Box("GatingGrid_subtEmptyCenter", gatingGridActiveWidth*fHalfUnit, PCBThickness*fHalfUnit + 1.*mm, (fieldCageLength - 2*gatingGridSupporterWidth)*fHalfUnit);
    G4VSolid* SolidGatingGrid_subtRight = new G4SubtractionSolid("GatingGrid_tmp", solidGatingGrid_base, solidGatingGrid_subtEmptyActive, 0,  G4ThreeVector(GatingGridEmptyAreaCenter, 0, 0));
    G4VSolid* SolidGatingGrid_subtLeft = new G4SubtractionSolid("GatingGrid_tmp", SolidGatingGrid_subtRight, solidGatingGrid_subtEmptyActive, 0,  G4ThreeVector(-GatingGridEmptyAreaCenter, 0, 0));
    return new G4SubtractionSolid("GatingGrid_tmp", SolidGatingGrid_subtLeft, solidGatingGrid_subtActiveArea, 0,  G4ThreeVector(0, 0, 0));
}

G4LogicalVolume* TPCDrumConstruction::GetWorld()
{
    G4Box* solidWorld = new G4Box("World", fWorldX*fHalfUnit, fWorldY*fHalfUnit, fWorldZ*fHalfUnit);
    return new G4LogicalVolume(solidWorld, GetGasMaterial("AIR"), "World");
}

G4LogicalVolume* TPCDrumConstruction::GetTPCDrumChamber()
{
    double chamberR = fTPCDrum->fChamberR *mm;
    double chamberHeight = fTPCDrum->fChamberHeight *mm;
    double flangeR = fTPCDrum->fFlangeR *mm;
    double beamLineLength = fTPCDrum->fBeamLineLength *mm;

    G4Tubs* solidChamber = new G4Tubs("chamber", 0., chamberR*fHalfUnit, chamberHeight*fHalfUnit, 0., 360*deg);
    G4Tubs* solidFlange_Beamline = new G4Tubs("flange_beamline", 0., flangeR*fHalfUnit, beamLineLength*fHalfUnit, 0., 360*deg);
    G4UnionSolid* detector = new G4UnionSolid("TPCDrum", solidFlange_Beamline, solidChamber, new G4RotationMatrix(0., CLHEP::pi/2, 0.), G4ThreeVector(0., 0., 0.));
    return new G4LogicalVolume(detector, GetGasMaterial("HE4_CO2"), "TPCDrum");
}

G4LogicalVolume* TPCDrumConstruction::GetBeamWindow()
{
    double flangeR = fTPCDrum->fFlangeR *mm;
    double windowThickness = fPar -> GetParDouble("TPCDrum/WindowThickness") *mm;

    G4Tubs* solidWindow = new G4Tubs("ChamberWindow", 0., flangeR*fHalfUnit, windowThickness*fHalfUnit, 0., 360*deg);
    return new G4LogicalVolume(solidWindow, GetSolidMaterial(fPar -> GetParString("TPCDrum/WindowMaterial")), "ChamberWindow");;
}

G4LogicalVolume* TPCDrumConstruction::GetReadoutPCB()
{
    double readoutPCBR = fTPCDrum->fReadoutPCBR *mm;
    double readoutPCBThickness = fTPCDrum->fReadoutPCBThickness *mm;

    G4Tubs* solidReadoutPCB = new G4Tubs("ReadOutPCB", 0., readoutPCBR*fHalfUnit, readoutPCBThickness*fHalfUnit, 0., 360*deg);
    return new G4LogicalVolume(solidReadoutPCB, GetSolidMaterial("PCB"), "ReadOutPCB");
}

G4LogicalVolume* TPCDrumConstruction::GetReadoutPadPlane()
{
    STDPadPlane padplane;
    double padGap = padplane.GetPadGap();
    double padHeight = padplane.GetPadHeight();
    double type1PadWidth = padplane.GetType1PadWidth();
    double type2PadWidth = padplane.GetType2PadWidth();

    double layerNum = padplane.GetLayerNum();
    double rowNum = padplane.GetRowNum();
    double type1PadNum = padplane.GetType1PadNum();
    double type2PadNum = padplane.GetType2PadNum();

    double activePadPlaneWidth = type1PadNum*(type1PadWidth+padGap) + type2PadNum*(type2PadWidth+padGap);
    double activePadPlaneHeight = layerNum*(padHeight+padGap);

    double readoutPCBThickness = fTPCDrum->fReadoutPCBThickness *mm;

    G4Box* solidReadoutPad = new G4Box("ReadOutPad", activePadPlaneWidth*fHalfUnit, readoutPCBThickness*fHalfUnit, activePadPlaneHeight*fHalfUnit);
    return new G4LogicalVolume(solidReadoutPad, GetSolidMaterial("PCB"), "ReadOutPad");
}

G4LogicalVolume* TPCDrumConstruction::GetFieldCageSideWall()
{
    double fieldCageHeight = fTPCDrum->fFieldCageHeight *mm;
    double fieldCageLength = fTPCDrum->fFieldCageLength *mm;
    double PCBThickness = fTPCDrum->fPCBThickness *mm;

    G4Box* solidFieldCage_Side = new G4Box("solidFieldCage_Side", PCBThickness*fHalfUnit, fieldCageHeight*fHalfUnit, fieldCageLength*fHalfUnit);
    return new G4LogicalVolume(solidFieldCage_Side, GetSolidMaterial("PCB"), "solidFieldCage_Side");
}

G4LogicalVolume* TPCDrumConstruction::GetFieldCageCathode()
{
    double fieldCageLength = fTPCDrum->fFieldCageLength *mm;
    double PCBThickness = fTPCDrum->fPCBThickness *mm;

    G4Box* solidFieldCage_Cathode = new G4Box("fieldCage_CathodePCB", fieldCageLength*fHalfUnit, PCBThickness*fHalfUnit, fieldCageLength*fHalfUnit);
    return new G4LogicalVolume(solidFieldCage_Cathode, GetSolidMaterial("PCB"), "fieldCage_CathodePCB");
}

G4LogicalVolume* TPCDrumConstruction::GetFieldCageDownStream()
{
    double fieldCageHeight = fTPCDrum->fFieldCageHeight *mm;
    double fieldCageLength = fTPCDrum->fFieldCageLength *mm;
    double PCBThickness = fTPCDrum->fPCBThickness *mm;
    double fieldCageDownstreamWindowSizeX = fTPCDrum->fFieldCageDownstreamWindowSizeX *mm;
    double fieldCageDownstreamWindowSizeY = fTPCDrum->fFieldCageDownstreamWindowSizeY *mm;

    G4Box* solidFieldCage_DownStreamPCB_base = new G4Box("fieldCage_downstreamPCB_base", fieldCageLength*fHalfUnit, fieldCageHeight*fHalfUnit, PCBThickness*fHalfUnit);
    G4Box* solidFieldCage_DownStreamPCB_subt = new G4Box("fieldCage_downstreamPCB_subt", fieldCageDownstreamWindowSizeX*fHalfUnit, fieldCageDownstreamWindowSizeY*fHalfUnit, PCBThickness*fHalfUnit + 1.*mm);
    G4VSolid* SolidFieldCage_DownStreamPCB = new G4SubtractionSolid("fieldCage_downstreamPCB", solidFieldCage_DownStreamPCB_base, solidFieldCage_DownStreamPCB_subt, 0,  G4ThreeVector(0,0,0));
    return new G4LogicalVolume(SolidFieldCage_DownStreamPCB, GetSolidMaterial("PCB"), "fieldCage_downstreamPCB");
}

G4LogicalVolume* TPCDrumConstruction::GetFieldCageUpStream()
{
    double fieldCageHeight = fTPCDrum->fFieldCageHeight *mm;
    double fieldCageLength = fTPCDrum->fFieldCageLength *mm;
    double PCBThickness = fTPCDrum->fPCBThickness *mm;
    double fieldCageUpstreamWindowSize = fTPCDrum->fFieldCageUpstreamWindowSize *mm;
    double fieldCageUpstreamWindowHeight = fTPCDrum->fFieldCageUpstreamWindowHeight *mm;

    G4Box* solidFieldCage_UpStreamPCB_base = new G4Box("fieldCage_UpstreamPCB_base", fieldCageLength*fHalfUnit, fieldCageHeight*fHalfUnit, PCBThickness*fHalfUnit);
    G4Box* solidFieldCage_UpStreamPCB_subt = new G4Box("fieldCage_UpstreamPCB_subt", fieldCageUpstreamWindowSize*fHalfUnit, fieldCageUpstreamWindowSize*fHalfUnit, PCBThickness*fHalfUnit + 1.*mm);
    G4VSolid* SolidFieldCage_UpStreamPCB = new G4SubtractionSolid("fieldCage_UpstreamPCB", solidFieldCage_UpStreamPCB_base, solidFieldCage_UpStreamPCB_subt, 0,  G4ThreeVector(0, fieldCageUpstreamWindowHeight - fieldCageHeight*fHalfUnit, 0));
    return new G4LogicalVolume(SolidFieldCage_UpStreamPCB, GetSolidMaterial("PCB"), "fieldCage_UpstreamPCB");
}

G4LogicalVolume* TPCDrumConstruction::GetGatingGrid()
{
    G4VSolid* SolidGatingGrid = GetGatingGridVolume();
    return new G4LogicalVolume(SolidGatingGrid, GetSolidMaterial("PCB"), "GatingGrid");
}

G4Material* TPCDrumConstruction::GetGasMaterial(TString gasName)
{
    gasName.ToUpper();

    G4double torrToAtom = 1./760. * CLHEP::STP_Pressure;
    G4double pressure = fPar -> GetParDouble("TPCDrum/GasPressure") * torrToAtom; // [atom]
    G4double mixRatio = fPar -> GetParDouble("TPCDrum/GasContourMixRatio") * 0.01;

    G4double STPTemperature = 273.15;
    G4double labTemperature = STPTemperature + 20.*kelvin;

    G4double gasDensity_He4 = 0.0001785 * STPTemperature/labTemperature *g/cm3;
    G4double gasDensity_CO2 = 0.001977 * STPTemperature/labTemperature *g/cm3;

    G4NistManager *nist = G4NistManager::Instance();
    G4Element* helium = nist -> FindOrBuildElement("He", false);
    G4Element* carbon = nist -> FindOrBuildElement("C", false);
    G4Element* oxygen = nist -> FindOrBuildElement("O", false);

    G4Material* air = nist -> FindOrBuildMaterial("G4_AIR");
    G4Material* He4Gas = new G4Material("He4Gas", gasDensity_He4, 1, kStateGas, labTemperature, pressure);
    He4Gas -> AddElement(helium, 1);
    G4Material* CO2Gas = new G4Material("CO2Gas", gasDensity_CO2, 2, kStateGas, labTemperature, pressure);
    CO2Gas -> AddElement(carbon, 1);
    CO2Gas -> AddElement(oxygen, 2);

    if(gasName.Index("AIR") != -1){return air;}
    else if(gasName.Index("HE4_CO2") != -1){
        G4double gasDensity_He4_CO2 = (1. - mixRatio) * gasDensity_He4 + mixRatio * gasDensity_CO2;
        G4Material* he4_co2_gas = new G4Material("He4_Co2_gas", gasDensity_He4_CO2, 2, kStateGas, labTemperature, CLHEP::STP_Pressure);
        he4_co2_gas -> AddMaterial(He4Gas, (1. - mixRatio)*gasDensity_He4/gasDensity_He4_CO2);
        he4_co2_gas -> AddMaterial(CO2Gas, mixRatio*gasDensity_CO2/gasDensity_He4_CO2);

        return  he4_co2_gas;
    }
    
    G4Material* vacuum = new G4Material("vacuum", 1, 2.016*g/mole, 1.e-10, kStateGas, labTemperature, 1.e-10*CLHEP::STP_Pressure);
    return vacuum;
}

G4Material* TPCDrumConstruction::GetSolidMaterial(TString name)
{
    G4NistManager *nist = G4NistManager::Instance();

    name.ToUpper();
    if(name.Index("PCB") != -1){
        double FR4Density        = 1.85 * g/cm3;
        double EpoxyDensity      = 1.25 * g/cm3;
        double fiberGlassDensity = 1.99 * g/cm3;

        double SiO2Density  = 2.196 * g/cm3;
        double B2O3Density  = 1.85  * g/cm3;
        double Al2O3Density = 3.987 * g/cm3;
        double CaODensity   = 3.34  * g/cm3;

        G4Element* H = nist -> FindOrBuildElement("H", false);
        G4Element* B = nist -> FindOrBuildElement("B", false);
        G4Element* C = nist -> FindOrBuildElement("C", false);
        G4Element* O = nist -> FindOrBuildElement("O", false);
        G4Element* Al = nist -> FindOrBuildElement("Al", false);
        G4Element* Si = nist -> FindOrBuildElement("Si", false);
        G4Element* Ca = nist -> FindOrBuildElement("Ca", false);

        // for Epoxy
        G4Material* epoxy = new G4Material("epoxy", EpoxyDensity, 2);
        epoxy -> AddElement(H, 2);
        epoxy -> AddElement(C, 2);

        // // for FiberGlass
        G4Material* SiO2 = new G4Material("SiO2", SiO2Density, 2);
        SiO2 -> AddElement(Si, 1);
        SiO2 -> AddElement(O, 2);

        G4Material* B2O3 = new G4Material("B2O3", B2O3Density, 2);
        B2O3 -> AddElement(B, 2);
        B2O3 -> AddElement(O, 3);

        G4Material* Al2O3 = new G4Material("Al2O3", Al2O3Density, 2);
        Al2O3 -> AddElement(Al, 2);
        Al2O3 -> AddElement(O, 3);

        G4Material* CaO = new G4Material("CaO", CaODensity, 2);
        CaO -> AddElement(Ca, 1);
        CaO -> AddElement(O, 1);

        G4Material* FiberGlass = new G4Material("FiberGlass", fiberGlassDensity, 4);
        FiberGlass -> AddMaterial(SiO2, 0.6*SiO2Density/fiberGlassDensity);
        FiberGlass -> AddMaterial(B2O3, 0.05*B2O3Density/fiberGlassDensity);
        FiberGlass -> AddMaterial(Al2O3, 0.13*Al2O3Density/fiberGlassDensity);
        FiberGlass -> AddMaterial(CaO, 0.22*CaODensity/fiberGlassDensity);

        G4Material* PCB_FR4 = new G4Material("PCB_FR4", FR4Density, 2);
        PCB_FR4 -> AddMaterial(epoxy, 0.4*EpoxyDensity/FR4Density);
        PCB_FR4 -> AddMaterial(FiberGlass, 0.6*fiberGlassDensity/FR4Density);

        return PCB_FR4;
    }
    else if(name.Index("SI") != -1){
        double SiDetectorDensity = 2.33 *g/cm3;

        G4Element* Si = nist -> FindOrBuildElement("Si", false);

        G4Material* SiDetector = new G4Material("SiDetector", SiDetectorDensity, 1);
        SiDetector -> AddElement(Si, 1);

        return SiDetector;
    }
    else if(name.Index("HAVAR") != -1){
        double HavarFoilDensity = 8.3 *g/cm3;

        G4Element* Co = nist -> FindOrBuildElement("Co", false);
        G4Element* Cr = nist -> FindOrBuildElement("Cr", false);
        G4Element* Ni = nist -> FindOrBuildElement("Ni", false);
        G4Element* Fe = nist -> FindOrBuildElement("Fe", false);
        G4Element* W = nist -> FindOrBuildElement("W", false);
        G4Element* Mo = nist -> FindOrBuildElement("Mo", false);
        G4Element* Mn = nist -> FindOrBuildElement("Mn", false);
        G4Element* C = nist -> FindOrBuildElement("C", false);
        G4Element* Be = nist -> FindOrBuildElement("Be", false);

        G4Material* Havar = new G4Material("Havar", HavarFoilDensity, 9);
        Havar -> AddElement(Co, 0.42);
        Havar -> AddElement(Cr, 0.195);
        Havar -> AddElement(Ni, 0.127);
        Havar -> AddElement(Fe, 0.1905);
        Havar -> AddElement(W, 0.027);
        Havar -> AddElement(Mo, 0.022);
        Havar -> AddElement(Mn, 0.016);
        Havar -> AddElement(C, 0.002);
        Havar -> AddElement(Be, 0.0005);

        return Havar;
    }
}

G4VisAttributes* TPCDrumConstruction::GetColor(TString color, double transparency)
{
    G4VisAttributes* visAtt = new G4VisAttributes();
    color.ToUpper();
    if(color.Index("WHITE") != -1){
        visAtt -> SetColor(1., 1., 1., transparency);
        return visAtt;
    }
    else if(color.Index("GRAY") != -1){
        visAtt -> SetColor(0.5, 0.5, 0.5, transparency);
        return visAtt;
    }
    else if(color.Index("RED") != -1){
        visAtt -> SetColor(1., 0., 0., transparency);
        return visAtt;
    }
    else if(color.Index("BLUE") != -1){
        visAtt -> SetColor(0., 0., 1., transparency);
        return visAtt;
    }
    else if(color.Index("GREEN") != -1){
        visAtt -> SetColor(0., 1., 0., transparency);
        return visAtt;
    }
    else if(color.Index("YELLOW") != -1){
        visAtt -> SetColor(1., 1., 0., transparency);
        return visAtt;
    }
    else if(color.Index("MAGENTA") != -1){
        visAtt -> SetColor(1., 0., 1., transparency);
        return visAtt;
    }
    else if(color.Index("CYAN") != -1){
        visAtt -> SetColor(0., 1., 1., transparency);
        return visAtt;
    }
    visAtt -> SetColor(0., 0., 0., transparency);
    return visAtt;
}