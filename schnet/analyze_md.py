import numpy as np
import matplotlib.pyplot as plt
import argparse

from schnetpack import units as spk_units
from schnetpack import properties
from schnetpack.md.data import HDF5Loader
from schnetpack.md.data import PowerSpectrum
from ase.io import write


ANGSTROM_E_TO_DEBYE_MULTIPLIER =  1.0 / 0.208194


def plot_energy(data):
    energies_calculator = data.get_property(properties.energy, atomistic=False).squeeze()
    energies_system = data.get_potential_energy()

    time_axis = np.arange(data.entries) * data.time_step / spk_units.fs

    energies_system *= spk_units.convert_units('kJ / mol', 'Hartree')

    plt.figure()
    plt.plot(time_axis, energies_system, label='E$_\mathrm{pot}$ (System)')
    plt.plot(time_axis, energies_calculator, label='E$_\mathrm{pot}$ (Logger)', ls='--')
    plt.ylabel('E [Hartree]')
    plt.xlabel('t [fs]')
    plt.legend()
    plt.tight_layout()
    plt.savefig('{}-energy.png'.format(name_prefix))


def plot_temperature(data):
    temperature = data.get_temperature()

    temperature_mean = np.cumsum(temperature) / (np.arange(data.entries) + 1)

    time_axis = np.arange(data.entries) * data.time_step / spk_units.fs

    plt.figure(figsize=(8,4))
    plt.plot(time_axis, temperature, label='T')
    plt.plot(time_axis, temperature_mean, label='T (avg.)')
    plt.ylabel('T [K]')
    plt.xlabel('t [fs]')
    plt.legend()
    plt.tight_layout()
    plt.savefig('{}-temperature.png'.format(name_prefix))


def plot_power_spectrum(data):
    spectrum = PowerSpectrum(data, resolution=2048)
    spectrum.compute_spectrum(molecule_idx=0)

    frequencies, intensities = spectrum.get_spectrum()

    plt.figure()
    plt.plot(frequencies, intensities)
    plt.xlim(0, 4000)
    plt.ylim(0, 100)
    plt.ylabel('I [a.u.]')
    plt.xlabel('$\omega$ [cm$^{-1}$]')
    plt.tight_layout()
    plt.savefig('{}-power-spectrum.png'.format(name_prefix))


def save_xyz(data):
    md_atoms = data.convert_to_atoms()

    write(
        '{}-trajectory.xyz'.format(name_prefix),
        md_atoms,
        format='xyz'
    )


def save_dipole_moment(data):
    charges = data.get_property('charges', atomistic=True)
    positions = data.get_positions()

    dipole_output_file = open('{}-dipole.dip'.format(name_prefix), 'w')
    steps_number = positions.shape[0]

    for i in range(steps_number):
        dipole_moment = np.array([0.0, 0.0, 0.0])
        
        for atom_position, atom_charge in zip(positions[i], charges[i]):
            pos = np.array(atom_position)
            dipole_moment += pos * ANGSTROM_E_TO_DEBYE_MULTIPLIER * atom_charge
        
        dipole_output_file.write('{} {} {} {} {} {} {}\n'.format(i, *dipole_moment, *dipole_moment))

    dipole_output_file.close()    


parser = argparse.ArgumentParser()
parser.add_argument('input', metavar='input', type=str, help='Path to HDF5 file')
parser.add_argument('-p', '--prefix', type=str, default='ec', help='Prefix for output file names')

args = parser.parse_args()
hdf5_file_path = args.input
name_prefix = args.prefix

data = HDF5Loader(hdf5_file_path)

plot_energy(data)
plot_temperature(data)
plot_power_spectrum(data)

save_xyz(data)
save_dipole_moment(data)
