import numpy as np
import matplotlib.pyplot as plt
import argparse
import glob

from schnetpack import units as spk_units
from schnetpack.md.data import HDF5Loader
from schnetpack.md.data import PowerSpectrum
from ase.io import write


ANGSTROM_E_TO_DEBYE_MULTIPLIER =  1.0 / 0.208194
SKIPPED_STEPS = 10000


def plot_energy(data_files, upper_xlim=30000):
    energy = []
    time = []

    for data in data_files:
        energies_system = data.get_potential_energy()
        energies_system *= spk_units.convert_units('kJ / mol', 'Hartree')

        time_axis = (np.arange(data.entries) * data.time_step / spk_units.fs)
        if len(time) > 0:
            time_axis = time_axis + time[-1][-1] + 1

        energy.append(energies_system)
        time.append(time_axis)

    energy_system = np.concatenate(energy)
    time_axis = np.concatenate(time)

    plt.figure()
    plt.plot(time_axis, energy_system, label='E$_\mathrm{pot}$', ls='-')
    plt.ylabel('E [Hartree]')
    plt.xlabel('t [fs]')
    plt.xlim((0, upper_xlim))
    plt.legend()
    plt.tight_layout()
    plt.savefig('{}-energy.png'.format(name_prefix))


def plot_temperature(data_files, upper_xlim=30000):
    temperatures = []
    entries = []
    time = []

    for data in data_files:
        temperatures.append(data.get_temperature())
        entries.append(data.entries)
        
        time_axis = np.arange(data.entries) * data.time_step / spk_units.fs
        if len(time) > 0:
            time_axis = time_axis + time[-1][-1] + 1
        time.append(time_axis)
    
    temperature = np.concatenate(temperatures)
    entries = np.sum(entries)

    temperature_mean = np.cumsum(temperature) / (np.arange(entries) + 1)
    time_axis = np.concatenate(time)

    plt.figure(figsize=(8,4))
    plt.plot(time_axis, temperature, label='T')
    plt.plot(time_axis, temperature_mean, label='T (avg.)')
    plt.ylabel('T [K]')
    plt.xlabel('t [fs]')
    plt.xlim((0, upper_xlim))
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


def save_xyz(data_files):
    for index, data in enumerate(data_files):
        md_atoms = data.convert_to_atoms()

        write(
            '{}-trajectory-{}.xyz'.format(name_prefix, index),
            md_atoms,
            format='xyz'
        )


def save_dipole_moment(data_files):
    for index, data in enumerate(data_files):
        charges = data.get_property('charges', atomistic=True)
        positions = data.get_positions()

        dipole_output_file = open('{}-dipole-{}.dip'.format(name_prefix, index), 'w')
        steps_number = positions.shape[0]

        for i in range(steps_number):
            dipole_moment = np.array([0.0, 0.0, 0.0])
            
            for atom_position, atom_charge in zip(positions[i], charges[i]):
                pos = np.array(atom_position)
                dipole_moment += pos * ANGSTROM_E_TO_DEBYE_MULTIPLIER * atom_charge
            
            dipole_output_file.write('{} {} {} {} {} {} {}\n'.format(i, *dipole_moment, *dipole_moment))

        dipole_output_file.close()    


parser = argparse.ArgumentParser()
parser.add_argument('input', metavar='input', type=str, help='Path to HDF5 files')
parser.add_argument('-p', '--prefix', type=str, default='ec', help='Prefix for output file names')

args = parser.parse_args()
hdf5_files_path = args.input
name_prefix = args.prefix

data_files = glob.glob(hdf5_files_path + '*.hdf5')

first_data_file = data_files[0]
initial_data = HDF5Loader(first_data_file, skip_initial=SKIPPED_STEPS)

data = [initial_data]
for i in data_files[1:]:
    data.append(HDF5Loader(i))

plot_energy(data)
plot_temperature(data)
if len(data) == 1:
    plot_power_spectrum(*data)

save_xyz(data)
save_dipole_moment(data)
