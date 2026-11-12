#include "STDSiArray.h"

ClassImp(STDSiArray);

STDSiArray::STDSiArray()
{
    fName = "STDSiArray";
    if (fChannelArray==nullptr)
        fChannelArray = new TObjArray();
}

bool STDSiArray::Init()
{
    lk_info << "Initializing STDSiArray" << std::endl;
    
    InitSiArrayGeometry();
    InitChannelMapping();
    InitChannelArray();

    fPadPlane = new STDPadPlane();

    return true;
}

void STDSiArray::Clear(Option_t *option)
{
    LKDetectorPlane::Clear(option);
}

void STDSiArray::Print(Option_t *option) const
{
    lk_info << "STDSiArray" << std::endl;
}

bool STDSiArray::IsInBoundary(Double_t x, Double_t z)
{

    return true;
}

Int_t STDSiArray::FindSiDetID(Double_t x, Double_t z)
{
    for(int i=0; i<fSiDetNum; i++){
        int unitPadID = fSiArrayPoly[i] -> FindBin(x, z);
        if(unitPadID > 0){return i;}
    }
    return -1;
}

Int_t STDSiArray::FindUnitPadID(Double_t x, Double_t z)
{
    for(int i=0; i<fSiDetNum; i++){
        int unitPadID = fSiArrayPoly[i] -> FindBin(x, z);
        if(unitPadID > 0){return unitPadID-1;}
    }
    return -1;
}

Int_t STDSiArray::GetStripID4PadID(int unitPadID)
{
    if(unitPadID < 0 || unitPadID >= fStripNum*fOhmicNum){return -1;}
    return int(unitPadID/4);
}

Int_t STDSiArray::GetStripID4JuncID(int juncID)
{
    if(juncID < 0 || juncID >= fJuncNum){return -1;}
    juncID = juncID+1;
    if(juncID%2==0){juncID -= 1;}
    int stripID = juncID/2;
    return stripID;
}

Int_t STDSiArray::GetOhmicID(int unitPadID)
{
    if(unitPadID < 0 || unitPadID >= fStripNum*fOhmicNum){return -1;}
    return int(unitPadID%4);
}

Int_t STDSiArray::GetUnitPadID(int ohmicID, int juncID)
{
    if(ohmicID < 0 || ohmicID >= fOhmicNum){return -1;}
    int stripID = GetStripID4JuncID(juncID);
    if(stripID == -1){return -1;}
    return ohmicID+(stripID*fOhmicNum);
}

Int_t STDSiArray::GetSiDetID(int aget, int chan)
{
    if(fChannelMap.find(make_pair(aget, chan)) != fChannelMap.end()){
        return fChannelMap.find(make_pair(aget, chan))->second.first;
    }
    return -1;
}

Int_t STDSiArray::GetOhmicID(int aget, int chan)
{
    if(aget != 0){return -1;} // exclude junction channel aget
    if(fChannelMap.find(make_pair(0, chan)) != fChannelMap.end()){
        return fChannelMap.find(make_pair(0, chan))->second.second;
    }
    return -1;
}

Int_t STDSiArray::GetJuncID(int aget, int chan)
{
    if(0 < aget && aget < 3){ // only junction channel of 8th si detector
        if(fChannelMap.find(make_pair(aget, chan)) != fChannelMap.end()){
            return fChannelMap.find(make_pair(aget, chan))->second.second;
        }
    }
    return -1;
}

Int_t STDSiArray::GetStripID(int aget, int chan)
{
    int juncID = GetJuncID(aget, chan);
    return GetStripID4JuncID(juncID);
}

Int_t STDSiArray::GetAsAdID(){return 3;}

Int_t STDSiArray::GetAgetID(int siDetID, bool isOhmic)
{
    if(isOhmic){return 0;}
    if(siDetID<0 || siDetID>=fSiDetNum){return -1;}
    if(3 <= siDetID && siDetID <= 6){return 1;} // LU, LM, LD, U
    if(siDetID <= 2 || siDetID == 7){return 2;} // RU, RM, RD, D
    return -1;
}

Int_t STDSiArray::GetChanID4Ohmic(int siDetID, int ohmicID)
{
    if(siDetID<0 || siDetID>=fSiDetNum){return -1;}
    if(ohmicID<0 || ohmicID>=fOhmicNum){return -1;}
    for(auto it = fChannelMap.begin(); it != fChannelMap.end(); ++it){
        if(it->first.first != 0){continue;}
        if((it->second).first == siDetID && (it->second).second == ohmicID){
            return it->first.second;
        }
    }
    return -1;
}

Int_t STDSiArray::GetChanID4Strip(int siDetID, int stripID, bool isFirstPairID)
{
    if(siDetID<0 || siDetID>=fSiDetNum){return -1;}
    if(stripID<0 || stripID>=fStripNum){return -1;}
    int tmpChanIdx = stripID*2 + 1;
    if(isFirstPairID){tmpChanIdx = stripID*2;}
    for(auto it = fChannelMap.begin(); it != fChannelMap.end(); ++it){
        if(it->first.first == 0){continue;}
        if((it->second).first == siDetID && (it->second).second == tmpChanIdx){
            return it->first.second;
        }
    }
    return -1;
}

Int_t STDSiArray::GetChannelIdx(int aget, int chan)
{
    if(aget < 0 || aget >= fAGETNum){return -1;}
    if(chan < 0 || chan >= fChanNum){return -1;}
    if(IsFPNChannel(chan)){return -1;}
    int fpnCounts = 0;
    if(chan >= 56){fpnCounts = 4;}
    else if(chan >= 45){fpnCounts = 3;}
    else if(chan >= 22){fpnCounts = 2;}
    else if(chan >= 11){fpnCounts = 1;}

    int chanIdx = chan+(aget*64) - fpnCounts;

    if(chanIdx >= fAGETNum*64+1 || chanIdx < 0){return -1;}
    return chanIdx;
}

bool STDSiArray::IsOhmic(int aget, int chan)
{
    if(aget != 0){return false;}
    if(GetOhmicID(aget, chan) == -1){return false;}
    return true;
}
bool STDSiArray::IsJunction(int aget, int chan)
{
    if(aget == 0){return false;}
    if(GetJuncID(aget, chan) == -1){return false;}
    return true;
}

Double_t STDSiArray::GetCenterUnitPadX(int siDetID, int unitPadID){return fUnitPadPosMap_UnitPadIdx[siDetID].find(unitPadID)->second.first;}
Double_t STDSiArray::GetCenterUnitPadX(int siDetID, int strip, int ohmic){return fUnitPadPosMap_so[siDetID].find(make_pair(strip, ohmic))->second.first;}
Double_t STDSiArray::GetCenterUnitPadZ(int siDetID, int unitPadID){return fUnitPadPosMap_UnitPadIdx[siDetID].find(unitPadID)->second.second;}
Double_t STDSiArray::GetCenterUnitPadZ(int siDetID, int strip, int ohmic){return fUnitPadPosMap_so[siDetID].find(make_pair(strip, ohmic))->second.second;}

void STDSiArray::ConvertSiLocalPos2Pad(int siDetID, double& x, double& y, double& z)
{
    double tmpX = x;
    double tmpY = y;
    double tmpZ = z;

    double padHeight = fPadPlane->GetPadHeight();
    double padGap = fPadPlane->GetPadGap();
    double layerNum = fPadPlane->GetLayerNum();
    double shiftY = (padHeight+padGap)*(layerNum/2.-1.) + (padHeight+padGap)/2.;
    tmpY = fSiArrayPlaneDistAtPadCenter + shiftY;

    double widthOfX = (siDetID < 6)? fSiHeight/double(fOhmicNum) : fSiWidth/double(fStripNum);
    double widthOfZ = (siDetID < 6)? fSiWidth/double(fStripNum) : fSiHeight/double(fOhmicNum);
    double shiftSignX = (siDetID < 3 || siDetID==6)? +1 : -1;
    double shiftSignZ = (siDetID < 3 || siDetID==7)? -1 : +1;
    double shiftX = GetCenterUnitPadX(siDetID, 0) + shiftSignX * widthOfX/2.;
    double shiftZ = GetCenterUnitPadZ(siDetID, 0) + shiftSignZ * widthOfZ/2.;

    if(siDetID==6 || siDetID==7){
        tmpX = z;
        tmpZ = x;
    }
    tmpX = shiftX - shiftSignX * tmpX;
    tmpZ = shiftX - shiftSignZ * tmpZ;

    x = tmpX;
    y = tmpY;
    z = tmpZ;
}

void STDSiArray::ConvertPad2SiLocalPos(int siDetID, double& x, double& y, double& z)
{
    double tmpX = x;
    double tmpY = y;
    double tmpZ = z;

    tmpY = 0;

    double widthOfX = (siDetID < 6)? fSiHeight/double(fOhmicNum) : fSiWidth/double(fStripNum);
    double widthOfZ = (siDetID < 6)? fSiWidth/double(fStripNum) : fSiHeight/double(fOhmicNum);
    double shiftSignX = (siDetID < 3 || siDetID==6)? +1 : -1;
    double shiftSignZ = (siDetID < 3 || siDetID==7)? -1 : +1;
    double shiftX = GetCenterUnitPadX(siDetID, 0) + shiftSignX * widthOfX/2.;
    double shiftZ = GetCenterUnitPadZ(siDetID, 0) + shiftSignZ * widthOfZ/2.;

    tmpX = shiftX - tmpX;
    tmpZ = shiftZ - tmpZ;

    x = fabs(tmpX);
    y = tmpY;
    z = fabs(tmpZ);
    if(siDetID==6 || siDetID==7){
        x = fabs(tmpZ);
        z = fabs(tmpX);
    }
}


Int_t STDSiArray::GetFPNChannelID(int chan)
{
    if(0 <= chan && chan < 17) {return 11;} 
    else if(chan < 34){return 22;} 
    else if(chan < 51){return 45;} 
    else if(chan < 68){return 56;}
    return -1;
}

bool STDSiArray::IsFPNChannel(int chan)
{
    if(chan == 11 || chan == 22 || chan == 45 || chan == 56){return true;}
    return false;
}

Double_t STDSiArray::GetSiCenterPos(int detIdx, int xzIdx)
{
    if(detIdx < 0 || detIdx > fSiDetNum){return -999.;}
    if(xzIdx < 0 || xzIdx > 1){return -999.;}
    return fSiDetectorCenter[detIdx][xzIdx];
}

TString STDSiArray::GetSiName(int detIdx)
{
    if(detIdx < 0 || detIdx > fSiDetNum){return "";}
    return fSiDetectorName[detIdx];
}

TH2Poly* STDSiArray::GetSiPoly(int detIdx)
{
    if(detIdx < 0 || detIdx > fSiDetNum){return 0;}
    if(fSiArrayPoly[detIdx]){
        return fSiArrayPoly[detIdx];
    }
    return 0;
}

void STDSiArray::InitSiArrayGeometry()
{
    double beamCenterZ = 128.74; // [mm] distance from padplane to Si array beam window center
    double awayXBeamCenter = 25.6; // [mm] distance from beam window center to side-Si active area boundary
    double awayZBeamCenter = 20.7; // [mm] distance from beam window center to up/down-Si active area boundary
    double sideSiDist = 5.; // [mm] distance between side-Si
    double startHeight = 163.59 - fSiWidth/2.; // [mm] start height for side-Si

    double z = startHeight;
    for(int i=0; i<fSiDetNum-2; i++){
        if(i == 3){z = startHeight;}
        double sign = (i < 3)? +1. : -1.;

        double x = sign*(fSiHeight/2. +awayXBeamCenter);
        fSiDetectorCenter[i][0] = x;
        fSiDetectorCenter[i][1] = z;
        z -= (fSiWidth + sideSiDist);
    }

    for(int i=fSiDetNum-2; i<fSiDetNum; i++){
        double sign = (i%2==0)? +1. : -1.;
        double height = beamCenterZ + sign*(fSiHeight/2. +awayZBeamCenter);
        fSiDetectorCenter[i][0] = 0.;
        fSiDetectorCenter[i][1] = height;
    }

    fSiDetectorName[0] = "SiDet_RU";
    fSiDetectorName[1] = "SiDet_RM";
    fSiDetectorName[2] = "SiDet_RD";
    fSiDetectorName[3] = "SiDet_LU";
    fSiDetectorName[4] = "SiDet_LM";
    fSiDetectorName[5] = "SiDet_LD";
    fSiDetectorName[6] = "SiDet_U";
    fSiDetectorName[7] = "SiDet_D";

    for(int i=0; i<fSiDetNum; i++){
        fSiArrayPoly[i] = new TH2Poly();
        fSiArrayPoly[i] -> SetStats(0);
        fSiArrayPoly[i] -> SetTitle(";x [mm]; z [mm]");

        double siWidth = (i<6)? fSiHeight : fSiWidth;
        double siHeight = (i<6)? fSiWidth : fSiHeight;
        double siUnitPadWidth = (i<6)? fSiHeight/double(fOhmicNum) : fSiWidth/double(fStripNum);
        double siUnitPadHeight = (i<6)? fSiWidth/double(fStripNum) : fSiHeight/double(fOhmicNum);

        double centerPosX = fSiDetectorCenter[i][0];
        double centerPosZ = fSiDetectorCenter[i][1];

        double SiCoordSignX = (i < 3 || i==6)? +1. : -1.;
        double SiCoordSignZ = (i < 3 || i==7)? -1. : +1.;
        double SiSignX = (i < 3 || i==6)? -1. : +1.;
        double SiSignZ = (i < 3 || i==7)? +1. : -1.;


        for(int s=0; s<fStripNum; s++){
            for(int o=0; o<fOhmicNum; o++){
                // Si local coordinate defined at J1-O1 channel edge
                double siLocalX = siUnitPadWidth/2. + o * siUnitPadWidth;
                double siLocalZ = siUnitPadHeight/2. + s * siUnitPadHeight;
                if(i >= 6){
                    siLocalX = siUnitPadWidth/2. + s * siUnitPadWidth;
                    siLocalZ = siUnitPadHeight/2. + o * siUnitPadHeight;
                }
                int unitPadIdx = o+(s*fOhmicNum);

                // Convert coordinate from Si-local to padplane-local
                double siUnitPadCenterX = centerPosX +SiCoordSignX * siWidth/2. +SiSignX * siLocalX;
                double siUnitPadCenterZ = centerPosZ +SiCoordSignZ * siHeight/2. +SiSignZ * siLocalZ;

                fUnitPadPosMap_so[i].insert({make_pair(s, o), make_pair(siUnitPadCenterX, siUnitPadCenterZ)});
                fUnitPadPosMap_UnitPadIdx[i].insert({unitPadIdx, make_pair(siUnitPadCenterX, siUnitPadCenterZ)});

                // Make the boundary of unitSiPad
                double boundaryX[5];
                double boundaryZ[5];
                for(int b=0; b<5; b++){
                    double xSign = (b<2 || b==4)? -1. : +1.;
                    double zSign = (b==1 || b==2)? -1. : +1.;
        
                    boundaryX[b] = siUnitPadCenterX + xSign * siUnitPadWidth/2.;
                    boundaryZ[b] = siUnitPadCenterZ + zSign * siUnitPadHeight/2.;
                }
                fSiArrayPoly[i] -> AddBin(5, boundaryX, boundaryZ);
            }
        }
    }
}

void STDSiArray::InitChannelMapping()
{
    fChannelMap.clear();

    // Physical AGET channels with the four FPN channels removed.
    vector<int> signalChannels;
    for(int chan=0; chan<fChanNum; chan++){
        if(IsFPNChannel(chan)){continue;}
        signalChannels.push_back(chan);
    }

    // ------------------------------------------------------------------
    // AGET 0: Ohmic mapping
    // ------------------------------------------------------------------
    // Only physical channels 17--50 (excluding FPN 22 and 45) are used.
    // The detector blocks progress upward in channel number as
    //   D, RD, RM, RU, U, LD, LM, LU,
    // while O0 is the largest physical channel inside each four-channel
    // detector block.
    const int ohmicDetectorOrder[fSiDetNum] = {
        7, // SiDet_D  : O0..O3 = 20,19,18,17
        2, // SiDet_RD : O0..O3 = 25,24,23,21
        1, // SiDet_RM : O0..O3 = 29,28,27,26
        0, // SiDet_RU : O0..O3 = 33,32,31,30
        6, // SiDet_U  : O0..O3 = 37,36,35,34
        5, // SiDet_LD : O0..O3 = 41,40,39,38
        4, // SiDet_LM : O0..O3 = 46,44,43,42
        3  // SiDet_LU : O0..O3 = 50,49,48,47
    };

    const int firstUsedOhmicSignalIndex = 16; // physical channel 17
    for(int block=0; block<fSiDetNum; block++){
        int siDetID = ohmicDetectorOrder[block];
        for(int ohmicID=0; ohmicID<fOhmicNum; ohmicID++){
            int signalIndex = firstUsedOhmicSignalIndex + block*fOhmicNum + (fOhmicNum-1-ohmicID);
            int chan = signalChannels[signalIndex];
            fChannelMap.insert({make_pair(0, chan), make_pair(siDetID, ohmicID)});
        }
    }

    // ------------------------------------------------------------------
    // AGET 1 and 2: Junction mapping
    // ------------------------------------------------------------------
    // Each detector occupies one block of 16 non-FPN channels. J0 is the
    // largest physical channel in its block.
    //
    // AGET 1, low -> high channel blocks: U, LD, LM, LU
    // AGET 2, low -> high channel blocks: D, RD, RM, RU
    const int junctionDetectorOrder[2][4] = {
        {6, 5, 4, 3}, // AGET 1: U, LD, LM, LU
        {7, 2, 1, 0}  // AGET 2: D, RD, RM, RU
    };

    for(int aget=1; aget<=2; aget++){
        for(int block=0; block<4; block++){
            int siDetID = junctionDetectorOrder[aget-1][block];
            
            for(int junctionID=0; junctionID<fJuncNum; junctionID++){
                int signalIndex = block*fJuncNum + (fJuncNum-1-junctionID);
                int chan = signalChannels[signalIndex];
                fChannelMap.insert({make_pair(aget, chan), make_pair(siDetID, junctionID)});
            }
        }
    }

    // AGET 3 intentionally has no Si junction mapping.
}

void STDSiArray::InitChannelArray()
{
    fChannelArray = new TObjArray();
    fMCTagArray = new TObjArray();

    for(int aget=0; aget<fAGETNum; aget++){
        for(int chan=0; chan<fChanNum; chan++){
            if(IsFPNChannel(chan)){continue;}

            GETChannel* channel = new GETChannel();
            channel -> SetAsad(3);
            channel -> SetAget(aget);
            channel -> SetChan(chan);
            int juncID = GetJuncID(aget, chan);
            int ohmicID = GetOhmicID(aget, chan);
            int unitPadID = GetUnitPadID(ohmicID, juncID);
            channel -> SetPadID(unitPadID);
            fChannelArray -> Add(channel);

            LKMCTag* mcTag = new LKMCTag();
            fMCTagArray -> Add(mcTag);
        }
    }
}

