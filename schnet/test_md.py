import os
import schnetpack as spk
import torch

from schnetpack.md import System, MaxwellBoltzmannInit, Simulator
from schnetpack.md.integrators import VelocityVerlet
from schnetpack.md.neighborlist_md import NeighborListMD
from schnetpack.transform import ASENeighborList
from ase.io import read
from custom import DftbCalculator

md_workdir = 'md-test'
n_steps = 10

if not os.path.exists(md_workdir):
    os.mkdir(md_workdir)

model_path = 'ec-model'
molecule_path = 'ec.xyz'

molecule = read(molecule_path)
n_replicas = 1

md_system = System()
md_system.load_molecules(
    molecule,
    n_replicas,
    position_unit_input='Angstrom'
)

system_temperature = 298  # Kelvin
md_initializer = MaxwellBoltzmannInit(
    system_temperature,
    remove_center_of_mass=True,
    remove_translation=True,
    remove_rotation=True,
    # wrap_positions=True
)

md_initializer.initialize_system(md_system)

time_step = 0.5  # fs
md_integrator = VelocityVerlet(time_step)

cutoff = 7.5  # Angstrom
cutoff_shell = 2.0  # Angstrom
md_neighbor_list = NeighborListMD(
    cutoff,
    cutoff_shell,
    ASENeighborList
)

md_calculator = DftbCalculator(
    model_path,
    'forces',
    'Hartree',
    'Bohr',
    md_neighbor_list,
    energy_key='energy',
    required_properties=[]
)

md_simulator = Simulator(
    md_system,
    md_integrator,
    md_calculator
)

if torch.cuda.is_available():
    md_device = 'cuda'
else:
    md_device = 'cpu'

md_precision = torch.float32

md_simulator = md_simulator.to(md_precision)
md_simulator = md_simulator.to(md_device)

md_simulator.simulate(n_steps)
