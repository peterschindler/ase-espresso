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
from . import subdirs
from . import espsite
from .postprocess import PostProcess
from .io import Mixins
import numpy as np
import atexit


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
            atoms=None,
            ecutwfc=350.0,
            ecutrho=None,
            ecutfock=None,
            nbnd=-10,
            kpts=(1, 1, 1),
            kptshift=(0, 0, 0),
            fft_grid=None,
            calculation='bfgs',
            ion_dynamics='bfgs',
            nstep=None,
            constr_tol=None,
            forc_conv_thr=0.05,
            cell_dynamics=None,
            press=None,
            dpress=None,
            cell_factor=None,
            cell_dofree=None,
            nosym=False,
            noinv=False,
            nosym_evc=False,
            no_t_rev=False,
            xc='PBE',
            beefensemble=False,
            printensemble=False,
            psppath=None,
            spinpol=False,
            noncollinear=False,
            spinorbit=False,
            outdir=None,
            calcstress=False,
            smearing='fd',
            sigma=0.1,
            fix_magmom=False,
            isolated=None,
            nqx1=None,
            nqx2=None,
            nqx3=None,
            exx_fraction=None,
            screening_parameter=None,
            exxdiv_treatment=None,
            ecutvcut=None,
            tot_charge=None,
            charge=None,
            tot_magnetization=-1,
            occupations='smearing',
            dipole={'status': False},
            field={'status': False},
            disk_io='default',
            wf_collect=False,
            removewf=True,  # ND
            removesave=False,  # ND
            mixing_beta=0.7,
            mixing_mode=None,
            conv_thr=1e-4,
            diagonalization='david',
            diago_cg_maxiter=None,
            startingpot=None,
            startingwfc=None,
            ion_positions=None,
            parflags=None,
            single_calculator=True,
            procrange=None,
            numcalcs=None,
            verbose='low',
            iprint=None,
            tstress=None,
            tprnfor=None,
            dt=None,
            lkpoint_dir=None,
            max_seconds=None,
            etot_conv_thr=None,
            tefield=None,
            dipfield=None,
            lelfield=None,
            nberrycyc=None,
            lorbm=None,
            lberry=None,
            gdir=None,
            nppstr=None,
            force_symmorphic=None,
            use_all_frac=None,
            one_atom_occupations=None,
            starting_spin_angle=None,
            degauss=None,
            nspin=None,
            ecfixed=None,
            qcutz=None,
            q2sigma=None,
            x_gamma_extrapolation=None,
            lda_plus_u=None,
            lda_plus_u_kind=None,
            edir=None,
            emaxpos=None,
            eopreg=None,
            eamp=None,
            clambda=None,
            report=None,
            lspinorb=None,
            esm_bc=None,
            esm_w=None,
            esm_efield=None,
            esm_nfit=None,
            london=None,
            london_s6=None,
            london_rcut=None,
            xdm=None,
            xdm_a1=None,
            xdm_a2=None,
            electron_maxstep=None,
            scf_must_converge=None,
            adaptive_thr=None,
            conv_thr_init=None,
            conv_thr_multi=None,
            mixing_ndim=None,
            mixing_fixed_ns=None,
            ortho_para=None,
            diago_thr_init=None,
            diago_david_ndim=None,
            diago_full_acc=None,
            efield=None,
            tqr=None,
            remove_rigid_rot=None,
            tempw=None,
            tolp=None,
            delta_t=None,
            nraise=None,
            refold_pos=None,
            upscale=None,
            bfgs_ndim=None,
            vdw_corr=None,
            ts_vdw_econv_thr=None,
            ts_vdw_isolated=None,
            lfcpopt=None,
            fcp_mu=None,
            esm_a=None,
            trust_radius_max=None,
            trust_radius_min=None,
            trust_radius_ini=None,
            w_1=None,
            w_2=None,
            wmass=None,
            press_conv_thr=None,
            results={},
            name='espresso',
            restart=None,
            ignore_bad_restart_file=False,
            label=None,
            command=None,
            site=espsite.config()
    ):
        """
    Construct an ase-espresso calculator.
    Parameters (with defaults in parentheses):
     atoms (None)
        list of atoms object to be attached to calculator
        atoms.set_calculator can be used instead
     kpts ( (1,1,1) )
        k-point grid sub-divisions, k-point grid density,
        explicit list of k-points, or simply 'gamma' for gamma-point only.
     kptshift ( (0,0,0) )
        shift of k-point grid
     fft_grid ( None )
        specify tuple of fft grid points (nr1,nr2,nr3) for q.e.
        useful for series of calculations with changing cell size
        (e.g. lattice constant optimization) uses q.e. default if not
        specified. [RK]
     calculation ( 'bfgs' )
        relaxation mode:
        - 'relax', 'scf', 'nscf': corresponding Quantum Espresso standard modes
     ion_dynamics ( 'bfgs' )
        - 'relax' and other Quantum Espresso standard relaxation modes:
                  Quantum Espresso own algorithms for structural optimization
                  are used
     fmax (0.05)
        max force limit for Espresso-internal relaxation (eV/Angstrom)
     constr_tol (None)
        constraint tolerance for Espresso-internal relaxation
     cell_dynamics (None)
        algorithm (e.g. 'BFGS') to be used for Espresso-internal
        unit-cell optimization
     press (None)
        target pressure for such an optimization
     dpress (None)
        convergence limit towards target pressure
     cell_factor (None)
        should be >>1 if unit-cell volume is expected to shrink a lot during
        relaxation (would be more efficient to start with a better guess)
     cell_dofree (None)
        partially fix lattice vectors
     nosym (False)
     noinv (False)
     nosym_evc (False)
     no_t_rev (False)
        turn off corresp. symmetries
     xc ('PBE')
        xc-functional to be used
     beefensemble (False)
        calculate basis energies for ensemble error estimates based on
        the BEEF-vdW functional
     printensemble (False)
        let Espresso itself calculate 2000 ensemble energies
     psppath (None)
        Directory containing the pseudo-potentials or paw-setups to be used.
        The ase-espresso interface expects all pot. files to be of the type
        element.UPF (e.g. H.UPF).
        If None, the directory pointed to be ESP_PSP_PATH is used.
     spinpol (False)
        If True, calculation is spin-polarized
     noncollinear (False)
        Non-collinear magnetism.
     spinorbit (False)
        If True, spin-orbit coupling is considered.
        Make sure to provide j-dependent pseudo-potentials in psppath
        for those elements where spin-orbit coupling is important
     outdir (None)
        directory where Espresso's output is collected,
        default: qe<random>
     calcstress (False)
        If True, calculate stress
     occupations ('smearing')
        Controls how Kohn-Sham states are occupied.
        Possible values: 'smearing', 'fixed' (molecule or insulator),
        or 'tetrahedra'.
     smearing ('fd')
        method for Fermi surface smearing
        - 'fd','Fermi-Dirac': Fermi-Dirac
        - 'mv','Marzari-Vanderbilt': Marzari-Vanderbilt cold smearing
        - 'gauss','gaussian': Gaussian smearing
        - 'mp','Methfessel-Paxton': Methfessel-Paxton
        For ase 3.7+ compatibility, smearing can also be a tuple where
        the first parameter is the method and the 2nd parameter
        is the smearing width which overrides sigma below
     sigma (0.1)
        smearing width in eV
     tot_charge (None)
        charge the unit cell,
        +1 means 1 e missing, -1 means 1 extra e
     charge (None)
        overrides tot_charge (ase 3.7+ compatibility)
     tot_magnetization (-1)
        Fix total magnetization,
        -1 means unspecified/free,
     fix_magmom (False)
        If True, fix total magnetization to current value.
     isolated (None)
        invoke an 'assume_isolated' method for screening long-range interactions
        across 3D supercells, particularly electrostatics.
        Very useful for charged molecules and charged surfaces,
        but also improves convergence wrt. vacuum space for neutral molecules.
        - 'makov-payne', 'mp': only cubic systems.
        - 'dcc': don't use.
        - 'martyna-tuckerman', 'mt': method of choice for molecules, works for any supercell geometry.
        - 'esm': Effective Screening Medium Method for surfaces and interfaces.
     nqx1, nqx2, nqx3 (all None)
        3D mesh for q=k1-k2 sampling of Fock operator. Can be smaller
        than number of k-points.
     exx_fraction (None)
        Default depends on hybrid functional chosen.
     screening_parameter (0.106)
        Screening parameter for HSE-like functionals.
     exxdiv_treatment (gygi-baldereschi)
        Method to treat Coulomb potential divergence for small q.
     ecutvcut (0)
        Cut-off for above.
     dipole ( {'status':False} )
        If 'status':True, turn on dipole correction; then by default, the
        dipole correction is applied along the z-direction, and the dipole is
        put in the center of the vacuum region (taking periodic boundary
        conditions into account).
        This can be overridden with:
        - 'edir':1, 2, or 3 for x-, y-, or z-direction
        - 'emaxpos':float percentage wrt. unit cell where dip. correction
          potential will be max.
        - 'eopreg':float percentage wrt. unit cell where potential decreases
        - 'eamp':0 (by default) if non-zero overcompensate dipole: i.e. apply
          a field
     disk_io ('default)
        How often espresso writes wavefunctions to disk
     avoidio (False)
        Will overwrite disk_io parameter if True
     removewf (True)
     removesave (False)
     wf_collect (False)
        control how much io is used by espresso;
        'removewf':True means wave functions are deleted in scratch area before
        job is done and data is copied back to submission directory
        'removesave':True means whole .save directory is deleted in scratch area
     convergence ( {'energy':1e-6,
                    'mixing':0.7,
                    'maxsteps':100,
                    'diag':'david'} )
        Electronic convergence criteria and diag. and mixing algorithms.
        Additionally, a preconditioner for the mixing algoritms can be
        specified, e.g. 'mixing_mode':'local-TF' or 'mixing_mode':'TF'.
     startingpot (None)
        By default: 'atomic' (use superposition of atomic orbitals for
        initial guess)
        'file': construct potential from charge-density.dat
        Can be used with load_chg and save_chg methods.
     startingwfc (None)
        By default: 'atomic'.
        Other options: 'atomic+random' or 'random'.
        'file': reload wave functions from other calculations.
        See load_wf and save_wf methods.
     parflags (None)
        Parallelization flags for Quantum Espresso.
        E.g. parflags='-npool 2' will distribute k-points (and spin if
        spin-polarized) over two nodes.
        """
        self.ecutwfc = ecutwfc
        self.ecutrho = ecutrho
        if isinstance(kpts, (float, int)):
            kpts = kptdensity2monkhorstpack(atoms, kpts)
        elif isinstance(kpts, str):
            assert kpts == 'gamma'
        else:
            assert len(kpts) == 3
        self.disk_io = disk_io
        self.removewf = removewf
        self.removesave = removesave
        self.wf_collect = wf_collect
        self.kpts = kpts
        self.kptshift = kptshift
        self.calculation = calculation
        self.ion_dynamics = ion_dynamics
        self.nstep = nstep
        self.constr_tol = constr_tol
        self.cell_dynamics = cell_dynamics
        self.press = press
        self.dpress = dpress
        self.cell_factor = cell_factor
        self.cell_dofree = cell_dofree
        self.nosym = nosym
        self.noinv = noinv
        self.nosym_evc = nosym_evc
        self.no_t_rev = no_t_rev
        self.xc = xc
        self.beefensemble = beefensemble
        self.printensemble = printensemble
        if isinstance(smearing, str):
            self.smearing = smearing
            self.sigma = sigma
        else:
            self.smearing = smearing[0]
            self.sigma = smearing[1]
        self.spinpol = spinpol
        self.noncollinear = noncollinear
        self.spinorbit = spinorbit
        self.fix_magmom = fix_magmom
        self.isolated = isolated
        if charge is None:
            self.tot_charge = tot_charge
        else:
            self.tot_charge = charge
        self.tot_magnetization = tot_magnetization
        self.occupations = occupations
        self.outdir = outdir
        self.calcstress = calcstress
        self.psppath = psppath
        self.dipole = dipole
        self.field = field
        self.mixing_beta = mixing_beta
        self.mixing_mode = mixing_mode
        self.electron_maxstep = electron_maxstep
        self.conv_thr = conv_thr
        self.diagonalization = diagonalization
        self.diago_cg_maxiter = diago_cg_maxiter
        self.startingpot = startingpot
        self.startingwfc = startingwfc
        self.ion_positions = ion_positions
        self.nqx1 = nqx1
        self.nqx2 = nqx2
        self.nqx3 = nqx3
        self.exx_fraction = exx_fraction
        self.screening_parameter = screening_parameter
        self.exxdiv_treatment = exxdiv_treatment
        self.ecutvcut = ecutvcut
        self.parflags = ''
        self.serflags = ''
        if parflags is not None:
            self.parflags += parflags
        self.single_calculator = single_calculator

        self.mypath = os.path.abspath(os.path.dirname(__file__))

        self.atoms = None
        self.sigma_small = 1e-13
        self.started = False
        self.got_energy = False

        # automatically generated list
        self.iprint = iprint
        self.tstress = tstress
        self.tprnfor = tprnfor
        self.dt = dt
        self.lkpoint_dir = lkpoint_dir
        self.max_seconds = max_seconds
        self.etot_conv_thr = etot_conv_thr
        self.forc_conv_thr = forc_conv_thr
        self.tefield = tefield
        self.dipfield = dipfield
        self.lelfield = lelfield
        self.nberrycyc = nberrycyc
        self.lorbm = lorbm
        self.lberry = lberry
        self.gdir = gdir
        self.nppstr = nppstr
        self.nbnd = nbnd
        self.ecutfock = ecutfock
        self.force_symmorphic = force_symmorphic
        self.use_all_frac = use_all_frac
        self.one_atom_occupations = one_atom_occupations
        self.starting_spin_angle = starting_spin_angle
        self.degauss = degauss
        self.nspin = nspin
        self.ecfixed = ecfixed
        self.qcutz = qcutz
        self.q2sigma = q2sigma
        self.x_gamma_extrapolation = x_gamma_extrapolation
        self.lda_plus_u = lda_plus_u
        self.lda_plus_u_kind = lda_plus_u_kind
        self.edir = edir
        self.emaxpos = emaxpos
        self.eopreg = eopreg
        self.eamp = eamp
        self.clambda = clambda
        self.report = report
        self.lspinorb = lspinorb
        self.esm_bc = esm_bc
        self.esm_w = esm_w
        self.esm_efield = esm_efield
        self.esm_nfit = esm_nfit
        self.london = london
        self.london_s6 = london_s6
        self.london_rcut = london_rcut
        self.xdm = xdm
        self.xdm_a1 = xdm_a1
        self.xdm_a2 = xdm_a2
        self.scf_must_converge = scf_must_converge
        self.adaptive_thr = adaptive_thr
        self.conv_thr_init = conv_thr_init
        self.conv_thr_multi = conv_thr_multi
        self.mixing_beta = mixing_beta
        self.mixing_ndim = mixing_ndim
        self.mixing_fixed_ns = mixing_fixed_ns
        self.ortho_para = ortho_para
        self.diago_thr_init = diago_thr_init
        self.diago_david_ndim = diago_david_ndim
        self.diago_full_acc = diago_full_acc
        self.efield = efield
        self.tqr = tqr
        self.remove_rigid_rot = remove_rigid_rot
        self.tempw = tempw
        self.tolp = tolp
        self.delta_t = delta_t
        self.nraise = nraise
        self.refold_pos = refold_pos
        self.upscale = upscale
        self.bfgs_ndim = bfgs_ndim
        self.vdw_corr = vdw_corr
        self.ts_vdw_econv_thr = ts_vdw_econv_thr
        self.ts_vdw_isolated = ts_vdw_isolated
        self.lfcpopt = lfcpopt
        self.fcp_mu = fcp_mu
        self.esm_a = esm_a
        self.trust_radius_max = trust_radius_max
        self.trust_radius_min = trust_radius_min
        self.trust_radius_ini = trust_radius_ini
        self.w_1 = w_1
        self.w_2 = w_2
        self.wmass = wmass
        self.press_conv_thr = press_conv_thr
        self.results = results
        self.site = site

        # Variables that cannot be set by inputs
        self.nvalence = None
        self.nel = None
        self.calculators = []

        # Auto create variables from input
        self.input_update()

        # Initialize lists of cpu subsets if needed
        if procrange is None:
            self.proclist = False
        else:
            self.proclist = True
            procs = self.site.procs + []
            procs.sort()
            nprocs = len(procs)
            self.myncpus = nprocs / numcalcs
            i1 = self.myncpus * procrange
            self.mycpus = self.localtmp + '/myprocs%04d.txt' % procrange
            f = open(self.mycpus, 'w')
            for i in range(i1, i1 + self.myncpus):
                print(procs[i], file=f)
            f.close()

        if atoms is not None:
            atoms.set_calculator(self)

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

        if self.psppath is None:
            try:
                self.psppath = os.environ['ESP_PSP_PATH']
            except BaseException:
                raise('Unable to find pseudopotential path. '
                      'Consider setting ESP_PSP_PATH environment variable')
        if self.dipole is None:
            self.dipole = {'status': False}
        if self.field is None:
            self.field = {'status': False}

        if self.beefensemble:
            if self.xc.upper().find('BEEF') < 0:
                raise KeyError(
                    'ensemble-energies only work with xc=BEEF '
                    'or variants of it!'
                )

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
