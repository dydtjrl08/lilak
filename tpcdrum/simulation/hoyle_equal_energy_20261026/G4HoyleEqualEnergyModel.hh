#ifndef TPCDRUM_G4_HOYLE_EQUAL_ENERGY_MODEL_HH
#define TPCDRUM_G4_HOYLE_EQUAL_ENERGY_MODEL_HH

// Project-local adapter. No installed NPTool source or library is modified.
#include "HoyleEqualEnergy.hh"
#include "G4Alpha.hh"
#include "G4DynamicParticle.hh"
#include "G4Exception.hh"
#include "G4FastStep.hh"
#include "G4FastTrack.hh"
#include "G4Ions.hh"
#include "G4ParticleDefinition.hh"
#include "G4Region.hh"
#include "G4SystemOfUnits.hh"
#include "G4Track.hh"
#include "G4VFastSimulationModel.hh"
#include "G4ios.hh"
#include "Randomize.hh"
#include <algorithm>
#include <cmath>
#include <stdexcept>

class G4HoyleEqualEnergyModel : public G4VFastSimulationModel {
public:
    explicit G4HoyleEqualEnergyModel(const G4String& name, G4Region* region,
                                     double excitationMeV = 7.65,
                                     double toleranceMeV = 0.02)
        : G4VFastSimulationModel(name, region),
          excitation_(excitationMeV * CLHEP::MeV),
          tolerance_(toleranceMeV * CLHEP::MeV) {
        if (!std::isfinite(excitation_) || !std::isfinite(tolerance_) ||
            excitation_ <= 0. || tolerance_ <= 0.)
            throw std::runtime_error("Invalid Hoyle excitation window.");
        G4cout << "[HoyleEqualEnergy] independent equal-energy 3alpha model; "
               << "excitation=" << excitationMeV << " +/- " << toleranceMeV
               << " MeV; lifetime=0; no branching-ratio prediction." << G4endl;
    }

    G4bool IsApplicable(const G4ParticleDefinition& definition) override {
        const auto* ion = dynamic_cast<const G4Ions*>(&definition);
        return ion && definition.GetAtomicNumber() == 6 &&
               definition.GetAtomicMass() == 12 &&
               std::abs(ion->GetExcitationEnergy() - excitation_) <= tolerance_;
    }

    G4bool ModelTrigger(const G4FastTrack& fastTrack) override {
        const auto* parent = fastTrack.GetPrimaryTrack();
        return parent && IsApplicable(*parent->GetParticleDefinition());
    }

    void DoIt(const G4FastTrack& fastTrack, G4FastStep& fastStep) override {
        try {
            const auto* parent = fastTrack.GetPrimaryTrack();
            if (!parent || !IsApplicable(*parent->GetParticleDefinition()))
                throw std::runtime_error("Unexpected particle in Hoyle decay.");
            auto* alpha = G4Alpha::AlphaDefinition();
            const double parentMassMeV = parent->GetDynamicParticle()->GetMass() / CLHEP::MeV;
            const double alphaMassMeV = alpha->GetPDGMass() / CLHEP::MeV;
            const auto momentum = parent->GetMomentum();
            const double energy = parent->GetTotalEnergy();
            const hoyle_equal_energy::Vec3 beta = {
                momentum.x() / energy, momentum.y() / energy, momentum.z() / energy};
            const auto decay = hoyle_equal_energy::Generate(
                parentMassMeV, beta, alphaMassMeV, []() { return G4UniformRand(); });

            // Global coordinates are used explicitly in CreateSecondaryTrack.
            const auto position = parent->GetPosition();
            const double time = parent->GetGlobalTime();
            fastStep.KillPrimaryTrack();
            fastStep.ProposePrimaryTrackPathLength(0.);
            fastStep.ProposeTotalEnergyDeposited(0.);
            fastStep.SetNumberOfSecondaryTracks(3);
            for (const auto& four : decay.lab) {
                const G4ThreeVector p(four.p.x, four.p.y, four.p.z);
                const double kinetic = (four.e - alphaMassMeV) * CLHEP::MeV;
                if (!std::isfinite(kinetic) || kinetic < -1.e-8 * CLHEP::MeV ||
                    !std::isfinite(p.mag2()))
                    throw std::runtime_error("Invalid laboratory alpha four-vector.");
                const auto direction = p.mag2() > 0. ? p.unit() : G4ThreeVector(0.,0.,1.);
                G4DynamicParticle daughter(alpha, direction, std::max(0., kinetic));
                fastStep.CreateSecondaryTrack(daughter, position, time, false);
            }
        } catch (const std::exception& error) {
            G4Exception("G4HoyleEqualEnergyModel::DoIt", "HoyleEqualEnergyKinematics",
                        FatalException, error.what());
        }
    }

private:
    double excitation_;
    double tolerance_;
};
#endif
