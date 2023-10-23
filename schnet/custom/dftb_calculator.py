import torch
import logging
import schnetpack
import dftbplus
import numpy as np

from typing import Union, List, Dict
from schnetpack.md import System
from schnetpack.md.calculators.base_calculator import MDCalculator
from schnetpack.md.neighborlist_md import NeighborListMD
from schnetpack.model import AtomisticModel
from schnetpack import units as spk_units
from custom import ChargeWriter


log = logging.getLogger(__name__)


class DftbCalculator(MDCalculator):
    '''
    MD calculator for DFTB calculation based on Schnet charges.

    Args:
        model_file (str): Path to stored schnetpack model.
        force_key (str): String indicating the entry corresponding to the molecular forces
        energy_unit (float, float): Placeholder, not used.
        position_unit (float, float): Placeholder, not used (DFTB+ uses Bohr converted then to internal MD units).
        neighbor_list (schnetpack.md.neighbor_list.MDNeighborList): Neighbor list object for determining which
                                                                    interatomic distances should be computed.
        system (System): simulated system, during initialization only atomic numbers are read.
        energy_key (str, optional): If provided, label is used to store the energies returned by the model to the
                                      system.
        required_properties (list): List of properties to be computed by the calculator
        property_conversion (dict(float)): Optional dictionary of conversion factors for other properties predicted by
                                           the model. Only changes the units used for logging the various outputs.
        lib_path (str): Path to DFTB+ shared library file.
        hsd_path (str): Path to DFTB+ input HSD file.
        log_file (str): Path for output DFTB+ log file.
    '''

    def __init__(
        self,
        model_file: str,
        force_key: str,
        energy_unit: Union[str, float],
        position_unit: Union[str, float],
        neighbor_list: NeighborListMD,
        system: System,
        energy_key: str = None,
        required_properties: List = ['charges'],
        property_conversion: Dict[str, Union[str, float]] = {},
        lib_path: str = './libdftbplus',
        hsd_path: str = 'dftb_in.hsd',
        log_file: str = 'out.log',
    ):
        super(DftbCalculator, self).__init__(
            required_properties=required_properties,
            force_key=force_key,
            energy_unit=energy_unit,
            position_unit=position_unit,
            energy_key=energy_key,
            property_conversion=property_conversion,
        )
        self.model = self._load_model(model_file)
        self.neighbor_list = neighbor_list
        self.charge_writer = self._initialize_charge_writer(system)
        
        self.cdftb = dftbplus.DftbPlus(libpath=lib_path,
                                       hsdpath=hsd_path,
                                       logfile=log_file)       

        self.position_conversion_dftb = 1.0 / spk_units.unit2internal('Bohr')
        self.energy_conversion = spk_units.unit2internal('Hartree')
        self.force_conversion = spk_units.unit2internal('Hartree') / spk_units.unit2internal('Bohr')
        self.is_pbc = torch.any(system.pbc)

    
    def _initialize_charge_writer(self, system: System) -> ChargeWriter:
        '''
        Read atom numbers from system variable and initialize ChargeWriter for further calculation.

        Args:
            system (System): object representing the system modeled in MD (only atomic numbers are read at this point).
        
        Returns:
            ChargeWriter: initialized with atomic numbers object for writing charges.
        '''
        
        numbers = system.atom_types.tolist()
        result = ChargeWriter(numbers, [], 'charges.dat')
        
        return result

    def _load_model(self, model_file: str) -> AtomisticModel:
        '''
        Load an individual model.

        Args:
            model_file (str): path to model.

        Returns:
           AtomisticTask: loaded schnetpack model
        '''

        log.info('Loading model from {:s}'.format(model_file))
        # load model and keep it on CPU, device can be changed afterwards
        model = torch.load(model_file, map_location='cpu').to(torch.float64)
        model = model.eval()

        log.info('Deactivating inference mode for simulation...')
        self._deactivate_postprocessing(model)

        return model

    @staticmethod
    def _deactivate_postprocessing(model: AtomisticModel) -> AtomisticModel:
        if hasattr(model, 'postprocessors'):
            for pp in model.postprocessors:
                if isinstance(pp, schnetpack.transform.AddOffsets):
                    log.info('Found `AddOffsets` postprocessing module...')
                    log.info(
                        'Constant offset of {:20.11f} per atom  will be removed...'.format(
                            pp.mean.detach().cpu().numpy()
                        )
                    )
        model.do_postprocessing = False
        return model

    def calculate(self, system: System):
        '''
        Main routine, generates a properly formatted input for the schnetpack model from the system, performs the
        computation (charges from schnetpack model + forces and energies from DFTB+) and uses the results to update the system state.

        Args:
            system (schnetpack.md.System): System object containing current state of the simulation.
        '''

        inputs = self._generate_input(system)
        self.results = self.model(inputs)
        
        self.charge_writer.set_charges(self.results['charges'])
        self.charge_writer.save_charges()

        positions = np.float64(system.positions.numpy()).squeeze() * self.position_conversion_dftb
        cell = None
        if self.is_pbc:
            cell = np.float64(system.cells.numpy()).squeeze() * self.position_conversion_dftb
 
        self.cdftb.set_geometry(positions, latvecs=cell)

        energy = torch.tensor(self.cdftb.get_energy().reshape(system.n_replicas, system.n_molecules, 1))
        self.results[self.energy_key] = energy

        forces = -1.0 * torch.tensor(self.cdftb.get_gradients().reshape(system.n_replicas, system.total_n_atoms, 3))
        self.results[self.force_key] = forces

        self._update_system(system)

    def _generate_input(self, system: System) -> Dict[str, torch.Tensor]:
        '''
        Function to extracts neighbor lists, atom_types, positions e.t.c. from the system and generate a properly
        formatted input for the schnetpack model.

        Args:
            system (schnetpack.md.System): System object containing current state of the simulation.

        Returns:
            dict(torch.Tensor): Schnetpack inputs in dictionary format.
        '''

        inputs = self._get_system_molecules(system)
        neighbors = self.neighbor_list.get_neighbors(inputs)
        inputs.update(neighbors)
        return inputs
    
    def __del__(self):
        if self.cdftb is not None:
            self.cdftb.close()
