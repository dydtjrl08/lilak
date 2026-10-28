// Standalone calibration executable linked with the SAME built LILAK/NPTool
// libraries as run_sim.exe. It initializes geometry/physics, NEVER BeamOn/Run.
// LILAK's original Initialize() opens G4OutputFile with RECREATE; the subclass
// deliberately bypasses those output/generator actions to protect existing MC.
#include "LKG4RunManager.h"
#include "LKParameterContainer.h"
#include "TPCDrumConstruction.h"
#include "NPOptionManager.h"
#include "PhysicsList.hh"
#include "G4Alpha.hh"
#include "G4Box.hh"
#include "G4EmCalculator.hh"
#include "G4EmParameters.hh"
#include "G4IonTable.hh"
#include "G4LogicalVolume.hh"
#include "G4Material.hh"
#include "G4Navigator.hh"
#include "G4ParticleDefinition.hh"
#include "G4PhysicalVolumeStore.hh"
#include "G4Region.hh"
#include "G4SystemOfUnits.hh"
#include "G4TransportationManager.hh"
#include "G4VPhysicalVolume.hh"
#include "G4Version.hh"
#include <algorithm>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <map>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

class CalibrationRunManager : public LKG4RunManager {
public:
    void Initialize() override { G4RunManager::Initialize(); }
};

namespace {
void Need(bool ok, const std::string& message) {
    if (!ok) throw std::runtime_error(message);
}
std::string Quote(const std::string& value) {
    std::string out="\"";
    for (char c:value) {
        if (c=='\\' || c=='\"') { out+='\\'; out+=c; }
        else if (c=='\n') out+="\\n";
        else if (static_cast<unsigned char>(c)<32) out+=' ';
        else out+=c;
    }
    return out+'\"';
}
std::string Vector(const G4ThreeVector& p) {
    std::ostringstream out; out<<std::setprecision(17);
    out<<'['<<p.x()<<','<<p.y()<<','<<p.z()<<']'; return out.str();
}
G4ThreeVector PadPosition(const G4ThreeVector& g4) {
    return {-g4.x()/mm,g4.z()/mm+66.,g4.y()/mm+128.8};
}
G4ThreeVector PadDirection(const G4ThreeVector& g4) {
    return {-g4.x(),g4.z(),g4.y()};
}
G4ThreeVector G4Position(double u,double v,double w) {
    return {-u*mm,(w-128.8)*mm,(v-66.)*mm};
}
struct Use { const G4Material* material; const G4Region* region; std::string key; };
std::vector<Use> uses;
std::map<std::pair<const G4Material*,const G4Region*>,int> byUse;
std::string MaterialKey(G4LogicalVolume* logical) {
    auto pair=std::make_pair(logical->GetMaterial(),logical->GetRegion());
    auto found=byUse.find(pair);
    if(found==byUse.end()) {
        const int id=uses.size(); byUse[pair]=id;
        uses.push_back({pair.first,pair.second,"material_"+std::to_string(id)});
        return uses.back().key;
    }
    return uses[found->second].key;
}
void Traverse(G4VPhysicalVolume* physical,const G4RotationMatrix& parentRotation,
              const G4ThreeVector& parentTranslation,std::vector<std::string>& volumes,
              std::vector<std::string>& detectors) {
    const auto rotation=parentRotation*physical->GetObjectRotationValue();
    const auto translation=parentTranslation+parentRotation*physical->GetObjectTranslation();
    auto* logical=physical->GetLogicalVolume();
    const auto key=MaterialKey(logical);
    std::ostringstream out;out<<std::setprecision(17);
    out<<"{\"name\":"<<Quote(physical->GetName())<<",\"copy_number\":"<<physical->GetCopyNo()
       <<",\"material\":"<<Quote(key)<<",\"solid\":"<<Quote(logical->GetSolid()->GetName())
       <<",\"center_uvw_mm\":"<<Vector(PadPosition(translation))<<'}';
    volumes.push_back(out.str());
    if (physical->GetName().find("SiDet_")==0) {
        const auto* box=dynamic_cast<const G4Box*>(logical->GetSolid());
        Need(box!=nullptr,"Si module is not a G4Box; export an explicit active-surface geometry");
        std::ostringstream si;si<<std::setprecision(17);
        si<<"{\"name\":"<<Quote(physical->GetName())<<",\"material\":"<<Quote(key)
          <<",\"center_uvw_mm\":"<<Vector(PadPosition(translation))
          <<",\"axis_u_uvw\":"<<Vector(PadDirection(rotation*G4ThreeVector(1,0,0)))
          <<",\"axis_v_uvw\":"<<Vector(PadDirection(rotation*G4ThreeVector(0,1,0)))
          <<",\"normal_uvw\":"<<Vector(PadDirection(rotation*G4ThreeVector(0,0,1)))
          <<",\"width_mm\":"<<2*box->GetXHalfLength()/mm
          <<",\"height_mm\":"<<2*box->GetYHalfLength()/mm
          <<",\"thickness_mm\":"<<2*box->GetZHalfLength()/mm
          <<",\"dead_layer_geometry_included\":false}";
        detectors.push_back(si.str());
    }
    for(int i=0;i<logical->GetNoDaughters();++i)
        Traverse(logical->GetDaughter(i),rotation,translation,volumes,detectors);
}
void Array(std::ostream& out,const std::vector<std::string>& values) {
    out<<'[';for(std::size_t i=0;i<values.size();++i){if(i)out<<',';out<<values[i];}out<<']';
}
void TracePaths(const std::string& input,const std::string& output,G4VPhysicalVolume* world) {
    if(input.empty())return;
    std::ifstream in(input);Need(bool(in),"Cannot open reconstructed path CSV");
    std::ofstream out(output);Need(bool(out),"Cannot create material-path CSV");
    out<<"path_id,segment_id,part,material,length_mm,volume\n"<<std::setprecision(17);
    std::string line; std::getline(in,line);
    Need(line=="path_id,segment_id,u0,v0,w0,u1,v1,w1","Unexpected path CSV header");
    while(std::getline(in,line)) {
        if(line.empty())continue;
        std::replace(line.begin(),line.end(),',',' ');std::istringstream row(line);
        long long pathID,segmentID;double u0,v0,w0,u1,v1,w1;
        Need(bool(row>>pathID>>segmentID>>u0>>v0>>w0>>u1>>v1>>w1),"Malformed path CSV row");
        auto p=G4Position(u0,v0,w0),end=G4Position(u1,v1,w1);
        auto delta=end-p;double remaining=delta.mag();
        if(remaining==0)continue;
        const auto direction=delta/remaining;
        G4Navigator navigator; navigator.SetWorldVolume(world);
        auto* physical=navigator.LocateGlobalPointAndSetup(p,&direction,false,false);
        int part=0;
        while(remaining>1.e-9*mm) {
            Need(physical!=nullptr,"Reconstructed path exits the exported world");
            G4double safety=0.;
            const double proposed=navigator.ComputeStep(p,direction,remaining,safety);
            const double step=std::min(remaining,proposed);
            Need(std::isfinite(step) && step>1.e-12*mm,"Zero/nonfinite navigation step; no material length fabricated");
            const auto name=physical->GetName();
            Need(name.find(',')==std::string::npos,"Volume name contains comma; cannot encode CSV");
            out<<pathID<<','<<segmentID<<','<<part++<<','<<MaterialKey(physical->GetLogicalVolume())
               <<','<<step/mm<<','<<name<<'\n';
            p+=step*direction;remaining-=step;
            if(remaining>1.e-9*mm) {
                navigator.SetGeometricallyLimitedStep();
                physical=navigator.LocateGlobalPointAndSetup(p,&direction,true,false);
            }
            Need(part<100000,"Too many geometry boundaries on one path segment");
        }
    }
    Need(bool(out),"Material path output failed");
}
}

int main(int argc,char** argv) {
    try {
        Need(argc==3 || argc==4,"Usage: export_geant4_materials.exe parameters.mac NEW_output_directory [reco_path_segments.csv]");
        const std::string directory=argv[2];
        auto* run=new CalibrationRunManager();
        run->AddParameterContainer(argv[1]);
        auto* par=run->GetParameterContainer();
        NPOptionManager::getInstance()->SetIsSimulation();
        NPOptionManager::getInstance()->SetReactionFile(par->GetParString("NPTool/ReactionFile").Data());
        run->SetUserInitialization(new TPCDrumConstruction());
        run->SetUserInitialization(new PhysicsList());
        G4EmParameters::Instance()->SetBuildCSDARange(true);
        run->Initialize(); // NO LILAK Run(), BeamOn(), G4OutputFile, or MC generation.
        auto* alpha=G4Alpha::AlphaDefinition();
        auto* carbon=G4IonTable::GetIonTable()->GetIon(6,12,0.);
        Need(alpha && carbon,"Cannot obtain exact alpha and C12 ground masses from this Geant4");
        auto* world=G4TransportationManager::GetTransportationManager()->GetNavigatorForTracking()->GetWorldVolume();
        Need(world!=nullptr,"Initialized world is missing");
        std::vector<std::string> volumes,detectors;
        Traverse(world,G4RotationMatrix(),G4ThreeVector(),volumes,detectors);
        G4EmCalculator calculator;
        std::vector<std::string> materialRows,tables,failed;
        for(const auto& use:uses) {
            const auto* m=use.material;
            std::ostringstream record;record<<std::setprecision(17);
            record<<"{\"key\":"<<Quote(use.key)<<",\"geant4_name\":"<<Quote(m->GetName())
                  <<",\"region\":"<<Quote(use.region?use.region->GetName():"none")
                  <<",\"density_g_cm3\":"<<m->GetDensity()/(g/cm3)
                  <<",\"radiation_length_mm\":"<<m->GetRadlen()/mm
                  <<",\"temperature_K\":"<<m->GetTemperature()/kelvin
                  <<",\"pressure_torr\":"<<m->GetPressure()/(atmosphere/760.)<<'}';
            materialRows.push_back(record.str());
            std::vector<double> energy,dedx,ranges; bool valid=true;
            for(int i=0;i<1200;++i) {
                const double e=std::exp(std::log(.001)+i*(std::log(1000.)-std::log(.001))/1199.);
                const double d=calculator.ComputeTotalDEDX(e*MeV,alpha,m)/(MeV/mm);
                const double r=calculator.GetCSDARange(e*MeV,alpha,m,use.region)/mm;
                if(!std::isfinite(d)||!std::isfinite(r)||d<=0||r<=0) {
                    if(energy.empty())continue;
                    valid=false;break;
                }
                if(!ranges.empty() && r<=ranges.back()){valid=false;break;}
                energy.push_back(e);dedx.push_back(d);ranges.push_back(r);
            }
            if(!valid || energy.size()<2){failed.push_back(Quote(use.key));continue;}
            const std::string csvName=use.key+"_alpha.csv";
            std::ofstream csv(directory+"/"+csvName);Need(bool(csv),"Cannot create stopping CSV");
            csv<<"energy_mev,dedx_mev_per_mm,range_mm\n"<<std::setprecision(17);
            for(std::size_t i=0;i<energy.size();++i)csv<<energy[i]<<','<<dedx[i]<<','<<ranges[i]<<'\n';
            Need(bool(csv),"Stopping CSV write failed");
            std::ostringstream table;table<<std::setprecision(17);
            table<<"{\"material\":"<<Quote(use.key)<<",\"particle_pdg\":1000020040,"
                 <<"\"particle_mass_mev\":"<<alpha->GetPDGMass()/MeV<<",\"csv\":"<<Quote(csvName)
                 <<",\"units\":{\"energy\":\"MeV\",\"range\":\"mm\",\"length\":\"mm\",\"dedx\":\"MeV/mm\"},"
                 <<"\"provenance\":{\"source\":\"initialized_Geant4_G4EmCalculator\",\"particle_pdg\":1000020040,"
                 <<"\"density_g_cm3\":"<<m->GetDensity()/(g/cm3)
                 <<",\"dedx_definition\":\"ComputeTotalDEDX: electronic plus applicable nuclear stopping\","
                 <<"\"range_definition\":\"GetCSDARange: initialized energy-loss process table\","
                 <<"\"range_dedx_consistency_validated\":false},";
            const char* names[]={"energy_mev","dedx_mev_per_mm","range_mm"};
            const std::vector<double>* values[]={&energy,&dedx,&ranges};
            for(int j=0;j<3;++j){if(j)table<<',';table<<Quote(names[j])<<":[";
                for(std::size_t i=0;i<values[j]->size();++i){if(i)table<<',';table<<(*values[j])[i];}table<<']';}
            table<<'}';tables.push_back(table.str());
        }
        std::ofstream out(directory+"/geant4_materials.json");Need(bool(out),"Cannot create material manifest");
        out<<std::setprecision(17)<<"{\"schema\":\"tpcdrum-geant4-material-export-v1\","
           <<"\"geant4_version\":"<<Quote(G4Version)<<",\"physics_list\":\"NPTool PhysicsList; preserve PhysicsListOption.txt\","
           <<"\"parameter_file\":"<<Quote(argv[1])<<",\"nominal_gas_pressure_torr\":"<<par->GetParDouble("TPCDrum/GasPressure")
           <<",\"nominal_CO2_percent\":"<<par->GetParDouble("TPCDrum/GasContourMixRatio")
           <<",\"masses_mev\":{\"alpha\":"<<alpha->GetPDGMass()/MeV<<",\"carbon12_ground\":"<<carbon->GetPDGMass()/MeV<<"},"
           <<"\"coordinate_transform\":\"u=-X_G4;v=Z_G4+66;w=Y_G4+128.8 mm\",\"materials\":";
        Array(out,materialRows);out<<",\"detectors\":";Array(out,detectors);
        out<<",\"physical_volumes\":";Array(out,volumes);out<<",\"tables\":";Array(out,tables);
        out<<",\"failed_tables\":";Array(out,failed);
        out<<",\"all_material_tables_complete\":"<<(failed.empty()?"true":"false")
           <<",\"event_generation_performed\":false,\"physical_accuracy_validated\":false}\n";
        Need(bool(out),"Manifest write failed");out.close();
        if(argc==4)TracePaths(argv[3],directory+"/reco_material_paths.csv",world);
        std::cout<<"Exported "<<tables.size()<<" alpha stopping tables, "<<detectors.size()
                 <<" Si modules; failed tables="<<failed.size()<<". No events generated.\n";
        return failed.empty()?0:3;
    } catch(const std::exception& error) {
        std::cerr<<"CALIBRATION EXPORT FAILED: "<<error.what()<<'\n';return 1;
    }
}
