// Read-only original/digi ROOT header inspector. No LILAK class access required.
// No includes. root -n -l -b -q 'export_file_header.C("input.root","NEW_header.json")'
namespace RecoHeaderExport20261027 {
TString Quote(TString s) {
    s.ReplaceAll("\\", "\\\\");s.ReplaceAll("\"", "\\\"");
    s.ReplaceAll("\n", "\\n");s.ReplaceAll("\r", "\\r");
    s.ReplaceAll("\t", "\\t");return "\""+s+"\"";
}
}
int export_file_header(const char* input,const char* output)
{
    using namespace RecoHeaderExport20261027;
    if(!input || !output || !gSystem->AccessPathName(output)) {
        ::Error("export_file_header","Input/output required and output must be new");return 1;
    }
    TFile file(input,"READ");
    auto* tree=dynamic_cast<TTree*>(file.Get("event"));
    if(file.IsZombie() || !tree) {
        ::Error("export_file_header","Missing input/event tree");return 1;
    }
    FILE* stream=fopen(output,"wx");
    if(!stream) {::Error("export_file_header","Cannot create new header output");return 1;}
    fprintf(stream,"{\"input\":%s,\"uuid\":%s,\"entries\":%lld,\"recovered\":%s,\"branches\":[",
            Quote(input).Data(),Quote(file.GetUUID().AsString()).Data(),tree->GetEntries(),
            file.TestBit(TFile::kRecovered)?"true":"false");
    auto* branches=tree->GetListOfBranches();
    for(int b=0;b<branches->GetEntriesFast();++b) {
        auto* branch=dynamic_cast<TBranch*>(branches->At(b));
        if(b)fprintf(stream,",");
        fprintf(stream,"{\"name\":%s,\"class\":%s}",Quote(branch->GetName()).Data(),Quote(branch->GetClassName()).Data());
    }
    fprintf(stream,"],\"keys\":[");
    TIter next(file.GetListOfKeys()); bool first=true;
    while(auto* key=dynamic_cast<TKey*>(next())) {
        if(!first)fprintf(stream,",");first=false;
        fprintf(stream,"{\"name\":%s,\"class\":%s}",Quote(key->GetName()).Data(),Quote(key->GetClassName()).Data());
    }
    fprintf(stream,"],\"waveforms_read\":false}\n");
    const bool good=ferror(stream)==0;fclose(stream);
    if(!good){::Error("export_file_header","Header write failed");return 1;}
    return 0;
}
