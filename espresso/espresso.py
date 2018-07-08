from ase.calculators.calculator import FileIOCalculator
from ase.units import Rydberg
from . import validate
from . import subdirs
from . import espsite
from . import utils
from .postprocess import PostProcess
from .io import Mixins
import numpy as np
import atexit
import warnings
import os
defaults = validate.variables


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
            ecutwfc,
            site=espsite.config(),
            **kwargs):

        self.site = site
        self.removewf = True
        self.removesave = True
        self.params = kwargs.copy()
        self.params['ecutwfc'] = ecutwfc / Rydberg
        atoms.set_calculator(self)
        self.symbols = self.atoms.get_chemical_symbols()
        self.species = np.unique(self.symbols)

        # Certain keys are used to define IO features or atoms object
        # information. For calculation consistency, user input is ignored.
        ignored_keys = ['prefix', 'ibrav', 'nat', 'ntyp']

        # Run validation checks
        bad_keys = []
        for key, val in self.params.items():
            if key in validate.__dict__:
                f = validate.__dict__[key]
                f(self, val)
            else:
                if key not in defaults:
                    warnings.warn('Not a valid key {}'.format(key))
                    bad_keys += [key]
                else:
                    warnings.warn('No validation for {}'.format(key))

            if key in ignored_keys:
                bad_keys += [key]

        for bkey in bad_keys:
            del self.params[bkey]

        # Auto create variables from input
        self.input_update()
        self.get_nvalence()

        print(self.params)

    def input_update(self):
        """Run initialization functions, such that this can be called
        if variables in espresso are changes using set or directly.
        """
        outdir = self.get_param('outdir')
        self.localtmp = subdirs.mklocaltmp(outdir, self.site)
        self.log = self.localtmp + '/log.pwo'
        self.scratch = subdirs.mkscratch(self.localtmp, self.site)

        atexit.register(subdirs.cleanup, self.localtmp, self.scratch,
                        self.removewf, self.removesave, self, self.site)

        # sdir is the directory the script is run or submitted from
        self.sdir = subdirs.getsubmitorcurrentdir(self.site)

        if self.get_param('ecutrho') is None:
            self.params['ecutrho'] = self.get_param('ecutwfc') * 10

        if self.get_param('dipfield') is not None:
            self.params['tefield'] = True

        if self.get_param('tstress') is not None:
            self.params['tprnfor'] = True

        self.params['nat'] = len(self.symbols)
        self.params['ntyp'] = len(self.species)

        # Apply any remaining default settings
        for key, value in defaults.items():
            setting = self.get_param(key)
            if setting is not None:
                self.params[key] = setting

        # if self.beefensemble:
        #     if self.xc.upper().find('BEEF') < 0:
        #         raise KeyError(
        #             'ensemble-energies only work with xc=BEEF '
        #             'or variants of it!')

        self.started = False
        self.got_energy = False

    def set_atoms(self, atoms):
        self.atoms = atoms.copy()

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

    def get_param(self, parameter):
        """Return the parameter associated with a calculator,
        otherwise, return the default value.
        """
        value = self.params.get(parameter, defaults[parameter])

        return value

    def get_nvalence(self):
        """Get number of valence electrons from pseudopotential or paw setup"""
        nel = {}
        for species in self.species:
            fname = os.path.join(self.get_param('pseudo_dir'), '{}.UPF'.format(species))
            valence = utils.grepy(fname, 'z valence|z_valence').split()[0]
            nel[species] = int(float(valence))

        nvalence = np.zeros_like(self.symbols, int)
        for i, symbol in enumerate(self.symbols):
            nvalence[i] = nel[symbol]

        return nvalence, nel

    def check_spinpol(self):
        mm = self.atoms.get_initial_magnetic_moments()
        sp = mm.any()
        self.summed_magmoms = np.sum(mm)
        if sp:
            if not self.spinpol and not self.noncollinear:
                raise KeyError(
                    'Explicitly specify spinpol=True or noncollinear=True '
                    'for spin-polarized systems'
                )
            elif abs(self.sigma) <= self.sigma_small and not self.fix_magmom:
                raise KeyError(
                    'Please use fix_magmom=True for sigma=0.0 eV and '
                    'spinpol=True. Hopefully this is not an extended '
                    'system...?'
                )
        else:
            if self.spinpol and abs(self.sigma) <= self.sigma_small:
                self.fix_magmom = True
        if abs(self.sigma) <= self.sigma_small:
            self.occupations = 'fixed'

    def get_fermi_level(self):
        efermi = self.inputfermilevel
        if efermi:
            return efermi

        self.stop()
        efermi = utils.grepy(self.log, 'Fermi energy')
        if efermi is not None:
            efermi = float(efermi.split()[-2])

        return efermi

    def open_calculator(self, filename='calc.tgz', mode='w'):
        """Save the contents of calc.save directory."""
        savefile = os.path.join(self.scratch, 'calc.save')

        self.topath(filename)
        self.stop()

        if mode == 'r':
            self.update(self.atoms)
            tarfile.open(filename, 'r', savefile)

            with open(savefile + '/fermilevel.txt', 'r') as f:
                self.inputfermilevel = float(f.readline())

        if mode == 'w':
            with open(savefile + '/fermilevel.txt', 'w') as f:
                f.write('{:.15f}\n'.format(self.get_fermi_level()))

            tarfile.open(filename, 'w', savefile)
