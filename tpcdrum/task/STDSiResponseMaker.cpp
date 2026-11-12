#include "STDSiResponseMaker.h"

ClassImp(STDSiResponseMaker);

STDSiResponseMaker::STDSiResponseMaker()
: fIsNewChannelArray(false)
{
    fName = "STDSiResponseMaker";
}

bool STDSiResponseMaker::Init()
{
    fDetector = (TPCDrum *) fRun -> GetDetector();
    fSiArray = (STDSiArray*) fDetector -> GetDetectorPlane(1);
    fPadPlane = (STDPadPlane*)fDetector -> GetDetectorPlane(0);

    fTrackArray = fRun -> KeepBranchA("MCTrack");
    for(int si=0; si<fSiArray->GetSiNum(); si++){
        fStepArray.push_back(fRun->GetBranchA("MCStep"+fSiArray->GetSiName(si)));
    }

    fChannelArray = fRun -> GetBranchA("RawPad");
    if(fChannelArray == nullptr){
        fChannelArray = fRun -> RegisterBranchA("RawPad", "GETChannel");
        fIsNewChannelArray = true;
    }
    fMCTagArray = fRun -> RegisterBranchA("MCTagSiArray", "LKMCTag");

    fTBTime = 40.; // [ns]
    if(fPar -> CheckPar("TPCDrum/TimeBucketUnit")){
        fTBTime = fPar -> GetParDouble("TPCDrum/TimeBucketUnit");
    }
    fPulseDelay = 80.;
    if(fPar -> CheckPar("TPCDrum/PulseDelay")){
        fPulseDelay = fPar -> GetParDouble("TPCDrum/PulseDelay");
    }

    fTuneManager = STDSimTuningManager::GetSimTuningManager();

    fRandom = new TRandom3(0);

    return true;
}

void STDSiResponseMaker::Exec(Option_t *option)
{
    int tpcPadNum = fPadPlane -> GetPadNum();
    if(fIsNewChannelArray){
        fChannelArray -> Clear("C");

        // Make the tpc channel part
        for(int i=0; i<tpcPadNum; i++){
            fChannelArray -> ConstructedAt(i);
        }
    }
    fMCTagArray -> Clear("C");

    for(int si=0; si<fSiArray->GetSiNum(); si++){
        int stepNum = fStepArray[si] -> GetEntries();

        for(int i=0; i<stepNum; i++){
            fStep = (LKMCStep*)fStepArray[si] -> UncheckedAt(i);

            // Convert coordinate from Geant system to local pad system
            ConvertCoordinateGeantToPad();

            double x = fStep -> GetX();
            double y = fStep -> GetY();
            double z = fStep -> GetZ();
            double t = fStep -> GetTime();
            double e = fStep -> GetEdep(); // [MeV]
            int trackId = fStep -> GetTrackID();

            int siDetID = fSiArray -> FindSiDetID(x, z);
            int siUnitPadID = fSiArray -> FindUnitPadID(x, z);

            unsigned int tb = t/fTBTime + fPulseDelay;
            if(tb >= 512){continue;}

            // ============== Ohmic channels ===============
            int ohmicIdx = fSiArray -> GetOhmicID(siUnitPadID);
            int ohmicChanID = fSiArray -> GetChanID4Ohmic(siDetID, ohmicIdx);
            int channelIdx_ohmic = fSiArray -> GetChannelIdx(0, ohmicChanID);

            double ohmicEnergy = e;
            MakeOhmicResponse(ohmicEnergy);
		    ohmicEnergy = ohmicEnergy*1e+6; // [eV]

            fChannel = (GETChannel*)fSiArray -> GetChannelFast(channelIdx_ohmic);
            fChannel -> GetBufferArray()[tb] += ohmicEnergy;

            fMCTag = (LKMCTag*)fSiArray -> GetMCTag(channelIdx_ohmic);
            fMCTag -> AddMCWeightTag(trackId, ohmicEnergy, tb);

            // ============== Junction channels ===============
            int stripID = fSiArray -> GetStripID4PadID(siUnitPadID);
            int juncChanID1 = fSiArray -> GetChanID4Strip(siDetID, stripID, true);
            int juncChanID2 = fSiArray -> GetChanID4Strip(siDetID, stripID, false);
            int agetID = fSiArray -> GetAgetID(siDetID, false);
            int channelIdx_junc1 = fSiArray -> GetChannelIdx(agetID, juncChanID1);
            int channelIdx_junc2 = fSiArray -> GetChannelIdx(agetID, juncChanID2);

            double juncEnergy1 = 0;
            double juncEnergy2 = 0;
            MakeJunctionResponse(siDetID, x, z, e, juncEnergy1, juncEnergy2);

            juncEnergy1 = juncEnergy1*1e6; // [eV]
            juncEnergy2 = juncEnergy2*1e6; // [eV]

            // Junction first pair channel
            fChannel = (GETChannel*)fSiArray -> GetChannelFast(channelIdx_junc1);
            fChannel -> GetBufferArray()[tb] += juncEnergy1;

            fMCTag = (LKMCTag*)fSiArray -> GetMCTag(channelIdx_junc1);
            fMCTag -> AddMCWeightTag(trackId, juncEnergy1, tb);

            // Junction second pair channel
            fChannel = (GETChannel*)fSiArray -> GetChannelFast(channelIdx_junc2);
            fChannel -> GetBufferArray()[tb] += juncEnergy2;

            fMCTag = (LKMCTag*)fSiArray -> GetMCTag(channelIdx_junc2);
            fMCTag -> AddMCWeightTag(trackId, juncEnergy2, tb);
        }
    }

    int tmpADC[512];
    memset(tmpADC, 0., sizeof(tmpADC));

    // Save data
    int chanIdx = 0;
    for(int aget=0; aget<fSiArray->GetAgetNum(); aget++){
        for(int chan=0; chan<fSiArray->GetChanNum()-4; chan++){
            fChannel = (GETChannel*)fSiArray -> GetChannelFast(chanIdx);
            fMCTag = (LKMCTag*)fSiArray -> GetMCTag(chanIdx);

            fChannel -> Copy(*(GETChannel*)fChannelArray -> ConstructedAt(tpcPadNum+chanIdx));
            fMCTag -> Copy(*(LKMCTag*)fMCTagArray -> ConstructedAt(chanIdx));

            fChannel -> SetWaveformY(tmpADC);
            fMCTag -> Clear();
            chanIdx++;
        }
    }
}

bool STDSiResponseMaker::EndOfRun()
{
    return true;
}

void STDSiResponseMaker::ConvertCoordinateGeantToPad()
{
    double x = fStep -> GetX();
    double y = fStep -> GetY();
    double z = fStep -> GetZ();

    fDetector -> GetCoordinateGeantToPad(x, y, z);

    fStep -> SetX(x);
    fStep -> SetY(y);
    fStep -> SetZ(z);
}

void STDSiResponseMaker::MakeOhmicResponse(double& e)
{
    double energy = e; // set the ohmic accumulated energy to half
    const double energyResolution = 0.05; // [MeV], 1 MeV resolution at MeV scale
    energy = fRandom -> Gaus(energy, energyResolution);
    e = energy;
}

void STDSiResponseMaker::MakeJunctionResponse(int siDetID, double x, double z, double e, double& junc1Energy, double& junc2Energy)
{
    double tmpX = x;
    double tmpY = 0;
    double tmpZ = z;
    fSiArray -> ConvertPad2SiLocalPos(siDetID, tmpX, tmpY, tmpZ);

    double energy = e; // set the junction accumulated energy to half
    const double energyResolution = 0.05; // [MeV], 1 MeV resolution at MeV scale
    energy = fRandom -> Gaus(energy, energyResolution);

    double energyRatioSecond = tmpX/fSiArray->GetSiHeight();
    double energyRatioFirst = 1. - energyRatioSecond;
	
    junc1Energy = energy * energyRatioFirst;
    junc2Energy = energy * energyRatioSecond;
}
