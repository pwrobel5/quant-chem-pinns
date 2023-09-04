from schnetpack.data import ASEAtomsData
from ase.io import read
from ase import Atoms

import sys
import numpy as np

SCALLING_FACTOR = 5.291772109E-01

if len(sys.argv) != 3:
    print('Invalid number of initial arguments!')
    exit(1)

xyz_file_name = sys.argv[1]
db_file_name = sys.argv[2]
atoms = read(xyz_file_name, index=':')
print('Found {} calculations'.format(len(atoms)))

# everything neeeds to be a numpy array, without it Torch makes errors
# multi_dimensional parameters such as charges (but not forces)
# need to be packed into one-element list, to have properly working 
# schnetpack with multidimensional outputs
property_list = []
atoms_list = []
for atom in atoms:
    positions = np.array(atom.positions)
    numbers = np.array(atom.numbers)
    pbc = np.array(atom.pbc)
    cell = np.array(atom.cell)
    ats = Atoms(positions=positions, numbers=numbers, pbc=pbc, cell=cell)
    info = atom.info
    info['charges'] = np.array([atom.get_initial_charges()]).T
    
    properties = {}
    for k, v in info.items():
        properties[k] = np.array(v)
    
    property_list.append(properties)
    atoms_list.append(ats)

new_dataset = ASEAtomsData.create(
    db_file_name,
    distance_unit='Ang',
    property_unit_dict={'charges': '_e'}
)
new_dataset.add_systems(property_list, atoms_list)

print('Available properties:')
for p in new_dataset.available_properties:
    print('-', p)

print('Properties for first snapshot')
example = new_dataset[0]
for k, v in example.items():
    print('-', k, ':', v.shape, ', value = ', v)
