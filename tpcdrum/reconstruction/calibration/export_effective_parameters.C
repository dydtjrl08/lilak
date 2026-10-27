// Requires existing LILAK dictionaries via macros/rootlogon.C.
// Reads persisted parameter containers, not the waveforms or MC labels.
namespace RecoParametersExport20261027 {
TString Quote(TString s) {
    s.ReplaceAll("\\", "\\\\");s.ReplaceAll("\"", "\\\"");
    s.ReplaceAll("\n", "\\n");s.ReplaceAll("\r", "\\r");
    s.ReplaceAll("\t", "\\t");return "\""+s+"\"";
}
}
int export_effective_parameters(const char* input,const char* output)
{
    using namespace RecoParametersExport20261027;
    if(!input || !output || !gSystem->AccessPathName(output))return 1;
    TFile file(input,"READ");if(file.IsZombie())return 1;
    FILE* stream=fopen(output,"wx");if(!stream)return 1;
    fprintf(stream,"{\"schema\":\"tpcdrum-persisted-parameters-v1\",\"input\":%s,\"uuid\":%s,\"containers\":[",
            Quote(input).Data(),Quote(file.GetUUID().AsString()).Data());
    bool first=true;TIter next(file.GetListOfKeys());std::set<TString> seen;
    while(auto* key=dynamic_cast<TKey*>(next())) {
        if(!seen.insert(key->GetName()).second)continue;
        if(TString(key->GetClassName())!="LKParameterContainer")continue;
        auto* parameters=dynamic_cast<LKParameterContainer*>(file.Get(key->GetName()));
        if(!parameters)continue;
        if(!first)fprintf(stream,",");first=false;
        fprintf(stream,"{\"key\":%s,\"parameters\":[",Quote(key->GetName()).Data());
        bool firstParameter=true;
        for(int i=0;i<parameters->GetEntriesFast();++i) {
            auto* parameter=dynamic_cast<LKParameter*>(parameters->At(i));if(!parameter)continue;
            if(!firstParameter)fprintf(stream,",");firstParameter=false;
            fprintf(stream,"{\"name\":%s,\"value\":%s,\"raw\":%s,\"unit\":%s,\"type_flags\":%d,\"is_line_comment\":%s,\"is_comment_out\":%s}",
                    Quote(parameter->GetName()).Data(),Quote(parameter->GetValue()).Data(),
                    Quote(parameter->GetRaw()).Data(),Quote(parameter->GetUnit()).Data(),
                    parameter->GetType(),parameter->IsLineComment()?"true":"false",
                    parameter->IsCommentOut()?"true":"false");
        }
        fprintf(stream,"]}");
    }
    fprintf(stream,"]}\n");const bool good=ferror(stream)==0;fclose(stream);return good ? 0 : 1;
}
