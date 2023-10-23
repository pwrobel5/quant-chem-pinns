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

    def __init__(self, numbers, charges, charge_file_name='charges.dat'):
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


