#ifndef TPCDRUM_HOYLE_EQUAL_ENERGY_HH
#define TPCDRUM_HOYLE_EQUAL_ENERGY_HH

// Standalone C++11 kinematics: no dependency on ROOT, Geant4 or NPTool.
// Units: c=1; masses/energies in MeV, momenta in MeV/c.
#include <array>
#include <cmath>
#include <stdexcept>
#include <utility>

namespace hoyle_equal_energy {
constexpr double pi=3.141592653589793238462643383279502884;

struct Vec3 {
    double x,y,z;
    Vec3(double a=0.,double b=0.,double c=0.):x(a),y(b),z(c){}
    Vec3 operator+(const Vec3& b)const{return Vec3(x+b.x,y+b.y,z+b.z);}
    Vec3 operator-(const Vec3& b)const{return Vec3(x-b.x,y-b.y,z-b.z);}
    Vec3 operator*(double s)const{return Vec3(s*x,s*y,s*z);}
    double dot(const Vec3& b)const{return x*b.x+y*b.y+z*b.z;}
    Vec3 cross(const Vec3& b)const{return Vec3(y*b.z-z*b.y,z*b.x-x*b.z,x*b.y-y*b.x);}
    double norm2()const{return dot(*this);}
    double norm()const{return std::sqrt(norm2());}
    Vec3 unit()const{
        const double length=norm();
        if(!(length>0.)||!std::isfinite(length))throw std::invalid_argument("Invalid vector normalization");
        return (*this)*(1./length);
    }
};
inline bool Finite(const Vec3& p){return std::isfinite(p.x)&&std::isfinite(p.y)&&std::isfinite(p.z);}

struct FourVector {
    Vec3 p; double e;
    FourVector(const Vec3& momentum=Vec3(),double energy=0.):p(momentum),e(energy){}
    FourVector operator+(const FourVector& b)const{return FourVector(p+b.p,e+b.e);}
    double mass()const{
        if(!Finite(p)||!std::isfinite(e)||e<=0.)throw std::invalid_argument("Invalid parent four-vector");
        const double q=p.norm();
        const double m2=(e-q)*(e+q);
        if(!(m2>0.)||!std::isfinite(m2))throw std::invalid_argument("Parent must be future timelike");
        return std::sqrt(m2);
    }
};

inline FourVector Boost(const FourVector& q,const Vec3& beta){
    const double b2=beta.norm2();
    if(!Finite(beta)||!std::isfinite(b2)||b2>=1.)throw std::invalid_argument("Boost requires |beta|<1");
    if(b2==0.)return q;
    const double gamma=1./std::sqrt(1.-b2), bp=beta.dot(q.p);
    // gamma^2/(gamma+1) = (gamma-1)/beta^2, stable near zero speed.
    return FourVector(q.p+beta*((gamma*gamma/(gamma+1.))*bp+gamma*q.e),gamma*(q.e+bp));
}

struct Decay {
    std::array<FourVector,3> com,lab;
    Vec3 normal,u,v;             // canonical unoriented plane: normal.z >= 0
    double phase;               // radians in [0, 2*pi/3)
    double parent_mass,alpha_mass,kinetic_com;
    std::array<int,3> label_to_slot;
};

template<class Uniform> double Draw(Uniform& random){
    const double value=random();
    if(!std::isfinite(value)||value<0.||value>=1.)
        throw std::invalid_argument("Uniform source must return a value in [0,1)");
    return value;
}

// A canonical hemisphere samples unoriented planes, not polar angles.
// Phase covers only 120 degrees because cyclic alpha labels are equivalent.
// A final random permutation keeps each stored daughter label isotropic.
// This is an imposed equal-energy (DDE) final state, not a branching-ratio model.
template<class Uniform>
Decay Generate(double parentMassMeV,const Vec3& beta,double alphaMassMeV,Uniform&& random){
    if(!std::isfinite(parentMassMeV)||!std::isfinite(alphaMassMeV)||alphaMassMeV<=0.||
       !(parentMassMeV>3.*alphaMassMeV))
        throw std::invalid_argument("Require positive masses and M_parent > 3*m_alpha");
    if(!Finite(beta)||!std::isfinite(beta.norm2())||beta.norm2()>=1.)
        throw std::invalid_argument("Invalid parent beta");
    Decay out;
    out.parent_mass=parentMassMeV;out.alpha_mass=alphaMassMeV;
    out.kinetic_com=(parentMassMeV-3.*alphaMassMeV)/3.;
    const double energy=alphaMassMeV+out.kinetic_com;
    const double momentum=std::sqrt(out.kinetic_com*(out.kinetic_com+2.*alphaMassMeV));
    const double nz=Draw(random),azimuth=2.*pi*Draw(random);
    const double transverse=std::sqrt((1.-nz)*(1.+nz));
    out.normal=Vec3(transverse*std::cos(azimuth),transverse*std::sin(azimuth),nz);
    // Measure-zero equator: make the normal sign unique there as well.
    if(out.normal.z==0. && (out.normal.y<0. || (out.normal.y==0. && out.normal.x<0.)))
        out.normal=out.normal*(-1.);
    const Vec3 reference=std::abs(out.normal.z)<.9 ? Vec3(0,0,1) : Vec3(1,0,0);
    out.u=reference.cross(out.normal).unit();out.v=out.normal.cross(out.u).unit();
    out.phase=(2.*pi/3.)*Draw(random);
    std::array<FourVector,3> slots;
    for(int i=0;i<3;++i){
        const double angle=out.phase+(2.*pi/3.)*i;
        const Vec3 direction=out.u*std::cos(angle)+out.v*std::sin(angle);
        slots[i]=FourVector(direction*momentum,energy);
    }
    out.label_to_slot={{0,1,2}};
    for(int i=2;i>0;--i){
        const int j=static_cast<int>(Draw(random)*(i+1));
        std::swap(out.label_to_slot[i],out.label_to_slot[j]);
    }
    for(int i=0;i<3;++i){
        out.com[i]=slots[out.label_to_slot[i]];
        out.lab[i]=Boost(out.com[i],beta);
    }
    return out;
}

template<class Uniform>
Decay Generate(const FourVector& parent,double alphaMassMeV,Uniform&& random){
    const double mass=parent.mass();
    return Generate(mass,parent.p*(1./parent.e),alphaMassMeV,std::forward<Uniform>(random));
}
} // namespace hoyle_equal_energy
#endif
