// Read only the exact Garfield/pulse ROOT assets selected by digitization.
// No detector initialization, no random numbers, no digitization.
namespace RecoTransportExport20261027 {
TString Quote(TString s) {
    s.ReplaceAll("\\", "\\\\");s.ReplaceAll("\"", "\\\"");
    s.ReplaceAll("\n", "\\n");s.ReplaceAll("\r", "\\r");
    s.ReplaceAll("\t", "\\t");return "\""+s+"\"";
}
}
int export_transport_assets(const char* gasFile,const char* pulseFile,const char* output)
{
    using namespace RecoTransportExport20261027;
    if(!output || !gSystem->AccessPathName(output)) {
        ::Error("export_transport_assets","Output must be a new filename");return 1;
    }
    FILE* stream=fopen(output,"wx");if(!stream)return 1;
    fprintf(stream,"{\"schema\":\"tpcdrum-transport-assets-v1\",\"gas_file\":%s,\"graphs\":[",Quote(gasFile?gasFile:"").Data());
    bool first=true;
    if(gasFile && *gasFile) {
        TFile gas(gasFile,"READ");
        if(gas.IsZombie()){fclose(stream);::Error("export_transport_assets","Cannot open exact gas asset");return 1;}
        const char* names[]={"DriftVelocity","TranseverseDiffusion","LongitudinalDiffusion"};
        for(const char* name:names) {
            auto* graph=dynamic_cast<TGraph*>(gas.Get(name));
            if(!graph){fclose(stream);::Error("export_transport_assets","Missing gas graph %s",name);return 1;}
            if(graph->GetN()<2){fclose(stream);::Error("export_transport_assets","Gas graph has too few points");return 1;}
            for(int i=0;i<graph->GetN();++i) {
                if(!TMath::Finite(graph->GetX()[i]) || !TMath::Finite(graph->GetY()[i])) {
                    fclose(stream);::Error("export_transport_assets","Gas graph contains nonfinite values");return 1;
                }
            }
            if(!first)fprintf(stream,",");first=false;
            fprintf(stream,"{\"name\":%s,\"title\":%s,\"x\":[",Quote(name).Data(),Quote(graph->GetTitle()).Data());
            for(int i=0;i<graph->GetN();++i){if(i)fprintf(stream,",");fprintf(stream,"%.17g",graph->GetX()[i]);}
            fprintf(stream,"],\"y\":[");
            for(int i=0;i<graph->GetN();++i){if(i)fprintf(stream,",");fprintf(stream,"%.17g",graph->GetY()[i]);}
            fprintf(stream,"]}");
        }
    }
    fprintf(stream,"],\"pulse_file\":%s,\"pulse_template\":[",Quote(pulseFile?pulseFile:"").Data());
    if(pulseFile && *pulseFile) {
        TFile pulse(pulseFile,"READ");auto* tree=dynamic_cast<TTree*>(pulse.Get("pulse"));
        if(pulse.IsZombie() || !tree || tree->GetEntries()<1){fclose(stream);return 1;}
        auto* leaf=tree->GetLeaf("template");
        if(!leaf || tree->GetEntry(0)<=0){fclose(stream);return 1;}
        if(leaf->GetLen()<1){fclose(stream);return 1;}
        for(int i=0;i<leaf->GetLen();++i)if(!TMath::Finite(leaf->GetValue(i))){fclose(stream);return 1;}
        for(int i=0;i<leaf->GetLen();++i){if(i)fprintf(stream,",");fprintf(stream,"%.17g",leaf->GetValue(i));}
    }
    fprintf(stream,"],\"drift_time_calibrated_for_charge_mean\":false,\"note\":\"Raw model assets only. Verify source units and fit timing offset/field-dependent transport before physical w reconstruction.\"}\n");
    const bool good=ferror(stream)==0;fclose(stream);return good ? 0 : 1;
}
