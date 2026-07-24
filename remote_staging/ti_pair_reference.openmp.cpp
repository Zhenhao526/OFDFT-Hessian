#include "ti_pair_reference.h"

#include "source_cell/unitcell.h"

#include <cmath>
#include <cstdlib>
#include <fstream>
#include <limits>
#include <stdexcept>

#ifdef _OPENMP
#include <omp.h>
#endif

namespace TiPairReference
{
namespace
{

void expect_key(std::istream& input, const std::string& expected)
{
    std::string key;
    input >> key;
    if (!input || key != expected)
    {
        throw std::runtime_error("invalid TI pair model: expected " + expected + ", found " + key);
    }
}

} // namespace

Model load_model(const std::string& path)
{
    std::ifstream input(path);
    if (!input)
    {
        throw std::runtime_error("cannot open TI pair model: " + path);
    }
    std::string magic;
    input >> magic;
    if (magic != "MPN_PAIR_REFERENCE_V1")
    {
        throw std::runtime_error("unsupported TI pair model format: " + magic);
    }
    Model model;
    std::size_t count = 0;
    expect_key(input, "n_centers");
    input >> count;
    expect_key(input, "sigma_angstrom");
    input >> model.sigma_angstrom;
    expect_key(input, "cutoff_angstrom");
    input >> model.cutoff_angstrom;
    expect_key(input, "constant_ev_per_atom");
    input >> model.constant_ev_per_atom;
    expect_key(input, "core_amplitude_ev");
    input >> model.core_amplitude_ev;
    expect_key(input, "core_cutoff_angstrom");
    input >> model.core_cutoff_angstrom;
    expect_key(input, "core_power");
    input >> model.core_power;
    expect_key(input, "centers_angstrom");
    model.centers_angstrom.resize(count);
    for (double& value : model.centers_angstrom)
    {
        input >> value;
    }
    expect_key(input, "coefficients_ev");
    model.coefficients_ev.resize(count);
    for (double& value : model.coefficients_ev)
    {
        input >> value;
    }
    if (!input || count == 0 || model.sigma_angstrom <= 0.0 || model.cutoff_angstrom <= 0.0)
    {
        throw std::runtime_error("incomplete TI pair model: " + path);
    }
    return model;
}

Configuration load_from_environment()
{
    const char* lambda_text = std::getenv("MPN_TI_LAMBDA");
    const char* model_path = std::getenv("MPN_TI_PAIR_MODEL");
    const char* reference_only_text = std::getenv("MPN_TI_REFERENCE_ONLY");
    if (lambda_text == nullptr && model_path == nullptr && reference_only_text == nullptr)
    {
        return Configuration{};
    }
    if (lambda_text == nullptr || model_path == nullptr)
    {
        throw std::runtime_error(
            "MPN_TI_LAMBDA and MPN_TI_PAIR_MODEL must be set when MPN TI is enabled");
    }
    Configuration configuration;
    configuration.enabled = true;
    configuration.lambda = std::stod(lambda_text);
    if (configuration.lambda < 0.0 || configuration.lambda > 1.0)
    {
        throw std::runtime_error("MPN_TI_LAMBDA must be in [0, 1]");
    }
    configuration.reference_only = reference_only_text != nullptr
                                   && std::string(reference_only_text) != "0"
                                   && std::string(reference_only_text) != "false";
    if (configuration.reference_only && configuration.lambda != 0.0)
    {
        throw std::runtime_error(
            "MPN_TI_REFERENCE_ONLY is only valid for lambda=0 and fixed-volume reference MD");
    }
    configuration.model_path = model_path;
    configuration.model = load_model(configuration.model_path);
    return configuration;
}

Result evaluate(const UnitCell& unit_cell, const Model& model)
{
    Result result;
    const int atom_count = unit_cell.nat;
#ifdef _OPENMP
    const int thread_count = omp_get_max_threads();
#else
    const int thread_count = 1;
#endif

    std::vector<double> thread_energies(thread_count, 0.0);
    thread_energies[0] = model.constant_ev_per_atom * atom_count;
    std::vector<double> thread_nearest(
        thread_count, std::numeric_limits<double>::infinity());
    std::vector<std::vector<ModuleBase::Vector3<double>>> thread_forces(
        thread_count,
        std::vector<ModuleBase::Vector3<double>>(
            atom_count, ModuleBase::Vector3<double>(0.0, 0.0, 0.0)));

#pragma omp parallel
    {
        int thread_id = 0;
#ifdef _OPENMP
        thread_id = omp_get_thread_num();
#endif
        double& local_energy = thread_energies[thread_id];
        double& local_nearest = thread_nearest[thread_id];
        auto& local_forces = thread_forces[thread_id];

#pragma omp for schedule(static)
        for (int i = 0; i < atom_count; ++i)
        {
            const auto& direct_i
                = unit_cell.atoms[unit_cell.iat2it[i]].taud[unit_cell.iat2ia[i]];
            for (int j = i + 1; j < atom_count; ++j)
            {
                const auto& direct_j
                    = unit_cell.atoms[unit_cell.iat2it[j]].taud[unit_cell.iat2ia[j]];
                double fx = direct_j.x - direct_i.x;
                double fy = direct_j.y - direct_i.y;
                double fz = direct_j.z - direct_i.z;
                fx -= std::round(fx);
                fy -= std::round(fy);
                fz -= std::round(fz);
                const ModuleBase::Vector3<double> displacement
                    = (fx * unit_cell.a1 + fy * unit_cell.a2 + fz * unit_cell.a3)
                      * unit_cell.lat0_angstrom;
                const double distance = displacement.norm();
                local_nearest = std::min(local_nearest, distance);
                if (distance <= 0.0 || distance >= model.cutoff_angstrom)
                {
                    continue;
                }

                const double angle = M_PI * distance / model.cutoff_angstrom;
                const double cutoff = 0.5 * (std::cos(angle) + 1.0);
                const double cutoff_derivative
                    = -0.5 * M_PI / model.cutoff_angstrom * std::sin(angle);
                double pair_energy = 0.0;
                double pair_derivative = 0.0;
                for (std::size_t basis = 0; basis < model.centers_angstrom.size(); ++basis)
                {
                    const double delta = distance - model.centers_angstrom[basis];
                    const double gaussian = std::exp(
                        -0.5 * delta * delta
                        / (model.sigma_angstrom * model.sigma_angstrom));
                    const double gaussian_derivative
                        = -delta / (model.sigma_angstrom * model.sigma_angstrom) * gaussian;
                    pair_energy += model.coefficients_ev[basis] * gaussian * cutoff;
                    pair_derivative += model.coefficients_ev[basis]
                                       * (gaussian_derivative * cutoff
                                          + gaussian * cutoff_derivative);
                }
                if (distance < model.core_cutoff_angstrom)
                {
                    const double reduced = 1.0 - distance / model.core_cutoff_angstrom;
                    pair_energy += model.core_amplitude_ev
                                   * std::pow(reduced, model.core_power);
                    pair_derivative += -model.core_amplitude_ev * model.core_power
                                       / model.core_cutoff_angstrom
                                       * std::pow(reduced, model.core_power - 1);
                }
                local_energy += pair_energy;
                const ModuleBase::Vector3<double> pair_force
                    = (pair_derivative / distance) * displacement;
                local_forces[i] += pair_force;
                local_forces[j] -= pair_force;
            }
        }
    }

    result.energy_ev = 0.0;
    result.forces_ev_per_angstrom.assign(
        atom_count, ModuleBase::Vector3<double>(0.0, 0.0, 0.0));
    result.nearest_neighbor_angstrom = std::numeric_limits<double>::infinity();
    for (int thread_id = 0; thread_id < thread_count; ++thread_id)
    {
        result.energy_ev += thread_energies[thread_id];
        result.nearest_neighbor_angstrom
            = std::min(result.nearest_neighbor_angstrom, thread_nearest[thread_id]);
        for (int atom = 0; atom < atom_count; ++atom)
        {
            result.forces_ev_per_angstrom[atom] += thread_forces[thread_id][atom];
        }
    }
    return result;
}

} // namespace TiPairReference
