void makeSimData()
{
    auto run = new LKRun();
    run -> AddDetector(new TPCDrum());
    run -> SetInputFile("./te.root");
    run -> SetOutputFile("te_digi.root");
    run -> Add(new STDDriftElectronMaker);
    run -> Add(new STDSiResponseMaker);   // Si detector response

    run -> Add(new STDElectronicsMaker);

    run -> Init();
    run -> Run();
}
