import os
import torch

from schnetpack.md import System, MaxwellBoltzmannInit, Simulator
from schnetpack.md.integrators import VelocityVerlet
from schnetpack.md.neighborlist_md import NeighborListMD
from schnetpack.md.simulation_hooks import LangevinThermostat, callback_hooks
from schnetpack.transform import ASENeighborList
from ase.io import read

from custom import DftbCalculator

md_workdir = 'ec-md'
n_steps = 2000

if not os.path.exists(md_workdir):
    os.mkdir(md_workdir)

model_path = 'ec-model/best_inference_model'
molecule_path = 'ec-non-pbc.xyz'

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
    #wrap_positions=True
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
    'Angstrom',
    md_neighbor_list,
    md_system,
    energy_key='energy',
    required_properties=['charges']
)

bath_temperature = 300  # K
time_constant = 100  # fs

langevin = LangevinThermostat(bath_temperature, time_constant)

data_streams = [
    callback_hooks.MoleculeStream(store_velocities=True),
    callback_hooks.PropertyStream(target_properties=['charges', 'energy'])
]

log_file = os.path.join(md_workdir, 'simulation.hdf5')
buffer_size = 100

file_logger = callback_hooks.FileLogger(
    log_file,
    buffer_size,
    data_streams=data_streams,
    every_n_steps=1,
    precision=64
)

chk_file = os.path.join(md_workdir, 'simulation.chk')
checkpoint = callback_hooks.Checkpoint(chk_file, every_n_steps=100)

tensorboard_dir = os.path.join(md_workdir, 'logs')
tensorboard_logger = callback_hooks.TensorBoardLogger(
    tensorboard_dir,
    ['energy', 'temperature']
)

simulation_hooks = [
    langevin,
    file_logger,
    checkpoint,
    tensorboard_logger
]

md_simulator = Simulator(
    md_system,
    md_integrator,
    md_calculator,
    simulator_hooks=simulation_hooks
)


md_precision = torch.float64
md_simulator = md_simulator.to(md_precision)

md_simulator.simulate(n_steps)
