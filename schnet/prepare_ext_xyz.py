import sys
import argparse

COMMENTARY_FORMAT = 'Properties=species:S:1:pos:R:3:charges:R:1\n'
VALENCE_CHARGES = {
    'h':  1,
    'li': 1,
    'c':  4,
    'n':  5,
    'o':  6,
    's':  6,
    'f':  7,
    'na': 1,
}


def get_valence_shell_charge(atom_symbol: str) -> float:
    symbol = atom_symbol.lower()
    charge = VALENCE_CHARGES.get(symbol, -1)

    if charge < 0:
        print('WARNING: unrecognized symbol: {}\n'.format(symbol))
    
    return charge

parser = argparse.ArgumentParser()
parser.add_argument('xyz_file_name', type=str, help='input .xyz file')
parser.add_argument('ext_xyz_file_name', type=str, help='output extended .xyz file')
parser.add_argument('-b', '--box-size', type=float, help='box size for using PBC')

args = parser.parse_args()

xyz_file_name = args.xyz_file_name
ext_file_name = args.ext_xyz_file_name
box_size = args.box_size

if box_size:
    COMMENTARY = 'Lattice="{} 0.0 0.0 0.0 {} 0.0 0.0 0.0 {}" {}'.format(box_size, box_size, box_size, COMMENTARY_FORMAT)
else:
    COMMENTARY = COMMENTARY_FORMAT

xyz_file = open(xyz_file_name, 'r')
ext_file = open(ext_file_name, 'w')

for line in xyz_file:
    atoms_number = int(line)
    xyz_file.readline()  # omit commentary

    ext_file.write('{}\n'.format(atoms_number))
    ext_file.write(COMMENTARY)

    for _ in range(atoms_number):
        atom_line = xyz_file.readline()
        atom_line = atom_line.split()

        atom_symbol = atom_line[0]
        x, y, z = float(atom_line[1]), float(atom_line[2]), float(atom_line[3])
        partial_charge = float(atom_line[4])
        charge = get_valence_shell_charge(atom_symbol) - partial_charge

        ext_file.write('{:s} {:12.8f} {:12.8f} {:12.8f} {:12.8f}\n'.format(
            atom_symbol, x, y, z, charge
        ))

xyz_file.close()
ext_file.close()