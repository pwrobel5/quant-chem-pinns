import dftbplus
import torch
import os
import ase
import schnetpack as spk
import schnetpack.transform as trn
import numpy as np


LIB_PATH = './libdftbplus'
MODEL_PATH = './ec-model'

BOHR__AA = 0.529177249
AA__BOHR = 1 / BOHR__AA

CUTOFF = 7.5


class ChargeWriter:

    VALENCE_POPULATIONS_ATOMS = {
        1: 1,
        2: 2,
        3: 1,
        4: 2,
        5: 3,
        6: 4,
        7: 5,
        8: 6,
        9: 7,
        10: 8
    }

    HEADER = '6\n F F F T {} 1 {:.16f}\n'

    def __init__(self, numbers, charges, charge_file_name='charges-test.dat'):
        self._numbers = numbers
        self._charges = charges
        self._charge_file_name = charge_file_name
    
    def set_charges(self, charges):
        self._charges = charges

    def __correct_populations(self, valence_populations):
        desired_sum = sum([ChargeWriter.VALENCE_POPULATIONS_ATOMS[i] for i in self._numbers])
        current_sum = sum(valence_populations)
        difference = desired_sum - current_sum
        atom_weights = [ChargeWriter.VALENCE_POPULATIONS_ATOMS[i] / desired_sum for i in self._numbers]
        
        for i in range(0, len(valence_populations)):
            valence_populations[i] += atom_weights[i] * difference
    
    def __get_valence_populations(self):
        valence_populations = []

        for number, charge in zip(self._numbers, self._charges):
            population = ChargeWriter.VALENCE_POPULATIONS_ATOMS[number] - charge.item()
            valence_populations.append(population)

        self.__correct_populations(valence_populations)
        return valence_populations

    def save_charges(self):
        valence_populations = self.__get_valence_populations()
        charge_file = open(self._charge_file_name, 'w')

        charge_file.write(ChargeWriter.HEADER.format(len(valence_populations), sum(valence_populations)))

        for population, number in zip(valence_populations, self._numbers):
            if number > 2:
                charge_file.write('{:.16f} {:.16f} {:.16f} {:.16f}\n'.format(population, 0.0, 0.0, 0.0))
            else:
                charge_file.write('{:.16f}\n'.format(population))
        
        charge_file.write('0 0\n')

        charge_file.close()


def main():
    coords = np.array([
        [0.5988008852E+00,   -0.1668846477E+00,    0.2470279826E+00],
        [0.1521534191E+01,   -0.4254637235E-02,   -0.7352164909E+00],
        [0.1366975883E+01,   -0.9198025954E+00,   -0.1725846921E+01],
        [0.1842079641E+00,   -0.1704116375E+01,   -0.1472906293E+01],
        [-0.1761773511E+00,   -0.1353760629E+01,   -0.1573463642E-01],
        [0.2393339077E+01,    0.8679055763E+00,   -0.7284768724E+00],
        [-0.1240732758E+01,   -0.1134297732E+01,    0.1387197196E+00],
        [-0.6095170717E+00,   -0.1418711623E+01,   -0.2192803956E+01],
        [0.4125564444E+00,   -0.2766383589E+01,   -0.1631649701E+01],
        [0.1113463133E+00,   -0.2149307676E+01,    0.7013618701E+00]
    ])

    latvecs = np.array([
        [4.802, 0.0, 0.0],
        [0.0, 4.802, 0.0],
        [0.0, 0.0, 4.802]
    ])

    pbc = np.array([True, True, True])

    numbers = np.array([8, 6, 8, 6, 6, 8, 1, 1, 1, 1])

    atoms = ase.Atoms(numbers=numbers, positions=coords, cell=latvecs, pbc=pbc)
    
    converter = spk.interfaces.AtomsConverter(neighbor_list=trn.ASENeighborList(cutoff=CUTOFF))
    inputs = converter(atoms)
    
    model = torch.load(os.path.join(MODEL_PATH, 'best_inference_model'), map_location=torch.device('cpu'))
    prediction = model(inputs)
    charges = prediction['charges']

    writer = ChargeWriter(numbers, charges)
    writer.save_charges()

    #print('Prediction:', prediction['charges'])

    #coords *= AA__BOHR
    #latvecs *= AA__BOHR


    '''
    cdftb = dftbplus.DftbPlus(libpath=LIB_PATH,
                              hsdpath='dftb_in.hsd',
                              logfile='ec.log')
    
    cdftb.set_geometry(coords, latvecs=latvecs)

    natoms = cdftb.get_nr_atoms()

    merminen = cdftb.get_energy()
    gradients = cdftb.get_gradients()
    grosschg = cdftb.get_gross_charges()
    
    cdftb.close()
    '''


if __name__ == '__main__':
    main()
