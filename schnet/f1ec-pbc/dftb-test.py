import dftbplus
import numpy as np


LIB_PATH = './libdftbplus'

BOHR__AA = 0.529177249
AA__BOHR = 1 / BOHR__AA

def main():
    coords = np.array([
        [0.2636156612,       -0.2652769597,        0.3575835888],
        [1.4035533059,        0.0796006416,       -0.3209995537],
        [1.5159568123,       -0.6379196411,       -1.5028288809],
        [0.5023746293,       -1.6941495354,       -1.5234422561],
        [-0.2510152456,      -1.5333330268,       -0.1780531366],
        [2.2186755417,        0.9054858298,        0.0697647234],
        [-1.3453500261,      -1.4682524425,       -0.2704407520],
        [-0.1640503540,      -1.5439949263,       -2.4118701739],
        [1.0385725806,       -2.6626816563,       -1.6165761681],
        [0.0185967241,       -2.3336393155,        0.5612577730]
    ])

    latvecs = np.array([
        [4.802, 0.0, 0.0],
        [0.0, 4.802, 0.0],
        [0.0, 0.0, 4.802]
    ])

    coords *= AA__BOHR
    latvecs *= AA__BOHR

    cdftb = dftbplus.DftbPlus(libpath=LIB_PATH,
                              hsdpath='dftb_in.hsd',
                              logfile='f1ec-test.log')
    
    cdftb.set_geometry(coords, latvecs=latvecs)

    natoms = cdftb.get_nr_atoms()

    merminen = cdftb.get_energy()
    gradients = cdftb.get_gradients()
    grosschg = cdftb.get_gross_charges()
    
    print('energy:', merminen)
    print('gradients:', gradients)
    print('charges:', grosschg)
    print('natoms:', natoms)

    cdftb.close()


if __name__ == '__main__':
    main()
