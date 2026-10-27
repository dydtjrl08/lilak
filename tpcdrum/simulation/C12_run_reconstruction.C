// ROOT launcher for the independent staged reconstruction package.
// No #include, task patch or LILAK rebuild required. No + or ++.
// Example from simulation:
// root -l -b -q 'C12_run_reconstruction.C("inspect")'
// root -l -b -q 'C12_run_reconstruction.C("finding","01_prepared.root","03_finding.root")'
namespace C12RecoLauncher20261027 {
TString Quote(const char* text) {
    std::string result="'";
    for (const char* p=text; p && *p; ++p)
        result += *p=='\'' ? "'\\''" : std::string(1,*p);
    result += "'";
    return result.c_str();
}
}
int C12_run_reconstruction(const char* stage="inspect", const char* inputRoot="",
                           const char* outputRoot="", const char* config="../reconstruction/config.json",
                           const char* pythonExecutable="python3")
{
    using namespace C12RecoLauncher20261027;
    const TString script=TString(gSystem->DirName(__FILE__))+"/../reconstruction/run.py";
    if (gSystem->AccessPathName(script)) {
        ::Error("C12_run_reconstruction","Runner not found: %s",script.Data());
        if (gROOT->IsBatch()) gSystem->Exit(1);
        return 1;
    }
    TString command=Quote(pythonExecutable)+" -u "+Quote(script)+" --stage "+Quote(stage)+" --config "+Quote(config);
    if (inputRoot && *inputRoot) command+=" --input "+Quote(inputRoot);
    if (outputRoot && *outputRoot) {
        command += TString(stage)=="all" ? " --output-dir " : " --output ";
        command += Quote(outputRoot);
    }
    std::cout << "Reconstruction command: " << command << std::endl;
    const int status=gSystem->Exec(command);
    if (status) {
        ::Error("C12_run_reconstruction","Stage failed or physical reconstruction is incomplete; inspect diagnostics (status=%d)",status);
        if (gROOT->IsBatch()) gSystem->Exit(1);
        return 1;
    }
    return 0;
}
