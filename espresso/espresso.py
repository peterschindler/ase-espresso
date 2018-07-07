#****************************************************************************
# Copyright (C) 2013-2015 SUNCAT
# This file is distributed under the terms of the
# GNU General Public License. See the file `COPYING'
# in the root directory of the present distribution,
# or http://www.gnu.org/copyleft/gpl.txt .
#****************************************************************************

import os
from ase.calculators.calculator import kptdensity2monkhorstpack
from ase.calculators.calculator import FileIOCalculator
from . import validate
from . import subdirs
from . import espsite
from .postprocess import PostProcess
from .io import Mixins
import numpy as np
import atexit
import warnings

class ConvergenceError(Exception):
    pass


class KohnShamConvergenceError(ConvergenceError):
    pass


class Espresso(PostProcess, Mixins, FileIOCalculator):
    """ASE interface for Quantum Espresso"""

    implemented_properties = [
        'energy', 'free_energy', 'forces', 'stress', 'magmom', 'magmoms'
    ]

    def __init__(
            self,
            atoms,
            site=espsite.config(),
            **kwargs):

        self.outdir = '.'
        self.site = site
        self.removewf = True
        self.removesave = True

        self.ecutrho = None
        self.ecutwfc = 400
        self.dipole = None
        self.field = None
        self.beefensemble = None
        procrange = None

        # Auto create variables from input
        self.input_update()

        if atoms is not None:
            self.atoms = atoms
            atoms.set_calculator(self)

        self.parameters = kwargs.copy()
        for key, val in self.parameters.items():
            if key in validate.__dict__:
                f = validate.__dict__[key]
                f(self, val)
            else:
                warnings.warn('No validation for {}'.format(key))

    def input_update(self):
        """Run initialization functions, such that this can be called
        if variables in espresso are changes using set or directly.
        """
        self.create_outdir()  # Create the tmp output folder

        # sdir is the directory the script is run or submitted from
        self.sdir = subdirs.getsubmitorcurrentdir(self.site)

        if self.ecutrho is None:
            self.ecutrho = 10 * self.ecutwfc
        else:
            assert self.ecutrho >= self.ecutwfc

        if self.dipole is None:
            self.dipole = {'status': False}
        if self.field is None:
            self.field = {'status': False}

        if self.beefensemble:
            if self.xc.upper().find('BEEF') < 0:
                raise KeyError(
                    'ensemble-energies only work with xc=BEEF '
                    'or variants of it!')

        self.started = False
        self.got_energy = False

    def create_outdir(self):

        self.localtmp = subdirs.mklocaltmp(self.outdir, self.site)
        self.log = self.localtmp + '/log.pwo'
        self.scratch = subdirs.mkscratch(self.localtmp, self.site)

        atexit.register(subdirs.cleanup, self.localtmp, self.scratch,
                        self.removewf, self.removesave, self, self.site)

    def calculate(self, atoms=None, properties=['energy'],
                  system_changes=None):
        """ase 3.7+ compatibility method"""
        if atoms is None:
            atoms = self.atoms
        self.update(atoms)

    def get_potential_energy(self, atoms=None, force_consistent=False):
        self.update(atoms)
        if force_consistent:
            return self.energy_free
        else:
            return self.energy_zero

    def get_forces(self, atoms):
        self.update(atoms)
        return self.forces.copy()

    def get_stress(self, dummyself=None):
        """Returns stress tensor in Voigt notation """
        if self.calcstress:
            # ASE convention for the stress tensor appears to differ
            # from the PWscf one by a factor of -1
            stress = -1.0 * self.get_final_stress()
            # converting to Voigt notation as expected by ASE
            stress = np.array([
                stress[0, 0], stress[1, 1], stress[2, 2], stress[1, 2],
                stress[0, 2], stress[0, 1]
            ])
            self.results['stress'] = stress
            return stress
        else:
            raise NotImplementedError

    def update(self, atoms):
        if self.atoms is None:
            self.set_atoms(atoms)

        x = atoms.cell - self.atoms.cell
        morethanposchange = (np.max(x) > 1e-13
                             or np.min(x) < -1e-13
                             or len(atoms) != len(self.atoms)
                             or (atoms.get_atomic_numbers() !=
                                 self.atoms.get_atomic_numbers()).any())
        x = atoms.positions - self.atoms.positions

        if (np.max(x) > 1e-13
            or np.min(x) < -1e-13
            or morethanposchange
            or (not self.started and not self.got_energy)
            or self.recalculate):

            self.recalculate = True
            self.results = {}
            if self.calculation in ('scf', 'nscf') or morethanposchange:
                self.stop()
            self.read(atoms)

        self.atoms = atoms.copy()
