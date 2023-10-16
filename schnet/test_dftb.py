import dftbplus
import torch
import os
import ase
import schnetpack as spk
import schnetpack.transform as trn
import numpy as np

from custom import ChargeWriter


LIB_PATH = './libdftbplus'
MODEL_PATH = './ec-model'

BOHR__AA = 0.529177249
AA__BOHR = 1 / BOHR__AA

CUTOFF = 7.5


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
