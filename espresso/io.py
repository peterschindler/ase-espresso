from . import utils
import numpy as np
from ase.units import Rydberg, Bohr
from subprocess import Popen, PIPE, call, check_output
import tarfile
import os


class Mixins:

    def write_input(self,
                    filename='pw.inp',
                    overridekpts=None,
                    overridekptshift=None,
                    overridenbands=None,
                    suppressforcecalc=False,
                    usetetrahedra=False):
        if self.atoms is None:
            raise ValueError('no atoms defined')

        f = open(self.pwinp, 'w')

        ionssec = self.calculation not in ('scf', 'nscf', 'bands')

        # &CONTROL ###
        print(
            "&CONTROL\n"
            "   calculation='{}',\n"
            "   prefix='calc',".format(self.calculation), file=f)
        if self.nstep is not None:
            print("  nstep={},".format(self.nstep), file=f)

        print("  pseudo_dir='{}',".format(self.psppath), file=f)
        print("  outdir='.',", file=f)
        efield = (self.field['status'])
        dipfield = (self.dipole['status'])
        if efield or dipfield:
            print('  tefield=.true.,', file=f)
            if dipfield:
                print('  dipfield=.true.,', file=f)
        if not self.dontcalcforces and not suppressforcecalc:
            print('  tprnfor=.true.,', file=f)
            if self.calcstress:
                print('  tstress=.true.,', file=f)
            if self.output is not None:
                if 'avoidio' in self.output:
                    if self.output['avoidio']:
                        self.output['disk_io'] = 'none'
                if 'disk_io' in self.output:
                    if self.output['disk_io'] in ['high', 'low', 'none']:
                        print(
                            '  disk_io=\'' + self.output['disk_io'] + '\',',
                            file=f)

                if 'wf_collect' in self.output:
                    if self.output['wf_collect']:
                        print('  wf_collect=.true.,', file=f)

        # We basically ignore convergence of total energy differences
        # between ionic steps and only consider fmax as in ase
        print('  etot_conv_thr=1d0,', file=f)
        fc = utils.num2str(self.fmax / (Rydberg / Bohr))
        print('  forc_conv_thr={},'.format(fc), file=f)

        # turn on fifo communication if espsite.py is set up that way
        if hasattr(site, 'fifo'):
            if site.fifo:
                print('  ase_fifo=.true.,', file=f)

        # automatically generated parameters
        if self.iprint is not None:
            print('  iprint=' + str(self.iprint) + ',', file=f)
        if self.tstress is not None:
            print('  tstress=' + utils.bool2str(self.tstress) + ',', file=f)
        if self.tprnfor is not None:
            print('  tprnfor=' + utils.bool2str(self.tprnfor) + ',', file=f)
        if self.dt is not None:
            print('  dt=' + utils.num2str(self.dt) + ',', file=f)
        if self.lkpoint_dir is not None:
            print(
                '  lkpoint_dir=' + utils.bool2str(self.lkpoint_dir) + ',',
                file=f)
        if self.max_seconds is not None:
            print(
                '  max_seconds=' + utils.num2str(self.max_seconds) + ',',
                file=f)
        if self.etot_conv_thr is not None:
            print(
                '  etot_conv_thr=' + utils.num2str(self.etot_conv_thr) + ',',
                file=f)
        if self.forc_conv_thr is not None:
            print(
                '  forc_conv_thr=' + utils.num2str(self.forc_conv_thr) + ',',
                file=f)
        if self.tefield is not None:
            print('  tefield=' + utils.bool2str(self.tefield) + ',', file=f)
        if self.dipfield is not None:
            print('  dipfield=' + utils.bool2str(self.dipfield) + ',', file=f)
        if self.lelfield is not None:
            print('  lelfield=' + utils.bool2str(self.lelfield) + ',', file=f)
        if self.nberrycyc is not None:
            print('  nberrycyc=' + str(self.nberrycyc) + ',', file=f)
        if self.lorbm is not None:
            print('  lorbm=' + utils.bool2str(self.lorbm) + ',', file=f)
        if self.lberry is not None:
            print('  lberry=' + utils.bool2str(self.lberry) + ',', file=f)
        if self.gdir is not None:
            print('  gdir=' + str(self.gdir) + ',', file=f)
        if self.nppstr is not None:
            print('  nppstr=' + str(self.nppstr) + ',', file=f)
        if self.lfcpopt is not None:
            print('  lfcpopt=' + utils.bool2str(self.lfcpopt) + ',', file=f)

        ### &SYSTEM ###
        print('/\n&SYSTEM\n  ibrav=0,', file=f)
        print('  nat=' + str(self.natoms) + ',', file=f)
        self.atoms2species()  # self.convertmag2species()
        print(
            '  ntyp=' + str(self.nspecies) + ',',
            file=f)  # str(len(self.msym))+','
        if self.tot_charge is not None:
            print(
                '  tot_charge=' + utils.num2str(self.tot_charge) + ',', file=f)
        inimagscale = 1.0
        if self.fix_magmom:
            assert self.spinpol
            self.totmag = self.summed_magmoms
            print(
                '  tot_magnetization=' +
                utils.num2str(self.totmag * inimagscale) + ',',
                file=f)
        elif self.tot_magnetization != -1:
            self.totmag = self.tot_magnetization
            print(
                '  tot_magnetization=' +
                utils.num2str(self.totmag * inimagscale) + ',',
                file=f)
        print('  ecutwfc=' + utils.num2str(self.pw / Rydberg) + ',', file=f)
        print('  ecutrho=' + utils.num2str(self.dw / Rydberg) + ',', file=f)
        if self.fw is not None:
            print(
                '  ecutfock=' + utils.num2str(self.fw / Rydberg) + ',', file=f)
        # temporarily (and optionally) change number of bands for nscf calc.
        if overridenbands is not None:
            if self.nbands is None:
                nbandssave = None
            else:
                nbandssave = self.nbands
            self.nbands = overridenbands
        if self.nbands is not None:
            # set number of bands
            if self.nbands > 0:
                self.nbnd = int(self.nbands)
            else:
                # if self.nbands is negative create -self.nbands extra bands
                if self.nvalence is None:
                    self.nvalence, self.nel = self.get_nvalence()
                if self.noncollinear:
                    self.nbnd = int(np.sum(self.nvalence) - self.nbands * 2.)
                else:
                    self.nbnd = int(np.sum(self.nvalence) / 2. - self.nbands)
            print('  nbnd=' + str(self.nbnd) + ',', file=f)
        if overridenbands is not None:
            self.nbands = nbandssave
        if usetetrahedra:
            print('  occupations=\'tetrahedra\',', file=f)
        else:
            if abs(self.sigma) > 1e-13:
                print('  occupations=\'' + self.occupations + '\',', file=f)
                print('  smearing=\'' + self.smearing + '\',', file=f)
                print(
                    '  degauss=' + utils.num2str(self.sigma / Rydberg) + ',',
                    file=f)
            else:
                if self.spinpol:
                    assert self.fix_magmom
                print('  occupations=\'fixed\',', file=f)
        if self.spinpol:
            print('  nspin=2,', file=f)
            spcount = 1
            if self.nel is None:
                self.nvalence, self.nel = self.get_nvalence()
                # FOLLOW SAME ORDERING ROUTINE AS FOR PSP
            for species in self.species:
                spec = self.specdict[species]
                el = spec.s
                mag = spec.magmom / self.nel[el]
                assert np.abs(mag) <= 1.  # magnetization oversaturated!!!
                print(
                    '  starting_magnetization(%d)=%s,' %
                    (spcount, utils.num2str(float(mag))),
                    file=f)
                spcount += 1
        elif self.noncollinear:
            print('  noncolin=.true.,', file=f)
            if self.spinorbit:
                print('  lspinorb=.true.', file=f)
            spcount = 1
            if self.nel is None:
                self.nvalence, self.nel = self.get_nvalence()
            # FOLLOW SAME ORDERING ROUTINE AS FOR PSP
            for species in self.species:
                spec = self.specdict[species]
                el = spec.s
                mag = spec.magmom / self.nel[el]
                assert np.abs(mag) <= 1.  # magnetization oversaturated!!!
                print(
                    '  starting_magnetization(%d)=%s,' %
                    (spcount, utils.num2str(float(mag))),
                    file=f)
                spcount += 1
        if self.isolated is not None:
            print('  assume_isolated=\'' + self.isolated + '\',', file=f)
        print('  input_dft=\'' + self.xc + '\',', file=f)
        if self.beefensemble:
            print('  ensemble_energies=.true.,', file=f)
            if self.printensemble:
                print('  print_ensemble_energies=.true.,', file=f)
            else:
                print('  print_ensemble_energies=.false.,', file=f)
        edir = 3
        if dipfield:
            try:
                edir = self.dipole['edir']
            except BaseException:
                pass
        elif efield:
            try:
                edir = self.field['edir']
            except BaseException:
                pass
        if dipfield or efield:
            print('  edir=' + str(edir) + ',', file=f)
        if dipfield:
            if 'emaxpos' in self.dipole:
                emaxpos = self.dipole['emaxpos']
            else:
                emaxpos = self.find_max_empty_space(edir)
            if 'eopreg' in self.dipole:
                eopreg = self.dipole['eopreg']
            else:
                eopreg = 0.025
            if 'eamp' in self.dipole:
                eamp = self.dipole['eamp']
            else:
                eamp = 0.0
            print('  emaxpos=' + utils.num2str(emaxpos) + ',', file=f)
            print('  eopreg=' + utils.num2str(eopreg) + ',', file=f)
            print('  eamp=' + utils.num2str(eamp) + ',', file=f)
        if efield:
            if 'emaxpos' in self.field:
                emaxpos = self.field['emaxpos']
            else:
                emaxpos = 0.0
            if 'eopreg' in self.field:
                eopreg = self.field['eopreg']
            else:
                eopreg = 0.0
            if 'eamp' in self.field:
                eamp = self.field['eamp']
            else:
                eamp = 0.0
            print('  emaxpos=' + utils.num2str(emaxpos) + ',', file=f)
            print('  eopreg=' + utils.num2str(eopreg) + ',', file=f)
            print('  eamp=' + utils.num2str(eamp) + ',', file=f)
 
        if self.nqx1 is not None:
            print('  nqx1=%d,' % self.nqx1, file=f)
        if self.nqx2 is not None:
            print('  nqx2=%d,' % self.nqx2, file=f)
        if self.nqx3 is not None:
            print('  nqx3=%d,' % self.nqx3, file=f)

        if self.exx_fraction is not None:
            print(
                '  exx_fraction=' + utils.num2str(self.exx_fraction) + ',',
                file=f)
        if self.screening_parameter is not None:
            print(
                '  screening_parameter=' +
                utils.num2str(self.screening_parameter) + ',',
                file=f)
        if self.exxdiv_treatment is not None:
            print(
                '  exxdiv_treatment=\'' + self.exxdiv_treatment + '\',',
                file=f)
        if self.ecutvcut is not None:
            print('  ecutvcut=' + utils.num2str(self.ecutvcut) + ',', file=f)

        if self.nosym:
            print('  nosym=.true.,', file=f)
        if self.noinv:
            print('  noinv=.true.,', file=f)
        if self.nosym_evc:
            print('  nosym_evc=.true.,', file=f)
        if self.no_t_rev:
            print('  no_t_rev=.true.,', file=f)
        if self.fft_grid is not None:  # RK
            print('  nr1=%d,' % self.fft_grid[0], file=f)
            print('  nr2=%d,' % self.fft_grid[1], file=f)
            print('  nr3=%d,' % self.fft_grid[2], file=f)

        # automatically generated parameters
        if self.ecutfock is not None:
            print('  ecutfock=' + utils.num2str(self.ecutfock) + ',', file=f)
        if self.force_symmorphic is not None:
            print(
                '  force_symmorphic=' + utils.bool2str(self.force_symmorphic) +
                ',',
                file=f)
        if self.use_all_frac is not None:
            print(
                '  use_all_frac=' + utils.bool2str(self.use_all_frac) + ',',
                file=f)
        if self.one_atom_occupations is not None:
            print(
                '  one_atom_occupations=' +
                utils.bool2str(self.one_atom_occupations) + ',',
                file=f)
        if self.starting_spin_angle is not None:
            print(
                '  starting_spin_angle=' +
                utils.bool2str(self.starting_spin_angle) + ',',
                file=f)
        if self.degauss is not None:
            print('  degauss=' + utils.num2str(self.degauss) + ',', file=f)
        if self.nspin is not None:
            print('  nspin=' + str(self.nspin) + ',', file=f)
        if self.ecfixed is not None:
            print('  ecfixed=' + utils.num2str(self.ecfixed) + ',', file=f)
        if self.qcutz is not None:
            print('  qcutz=' + utils.num2str(self.qcutz) + ',', file=f)
        if self.q2sigma is not None:
            print('  q2sigma=' + utils.num2str(self.q2sigma) + ',', file=f)
        if self.x_gamma_extrapolation is not None:
            print(
                '  x_gamma_extrapolation=' +
                utils.bool2str(self.x_gamma_extrapolation) + ',',
                file=f)
        if self.lda_plus_u is not None:
            print(
                '  lda_plus_u=' + utils.bool2str(self.lda_plus_u) + ',',
                file=f)
        if self.lda_plus_u_kind is not None:
            print(
                '  lda_plus_u_kind=' + str(self.lda_plus_u_kind) + ',', file=f)
        if self.edir is not None:
            print('  edir=' + str(self.edir) + ',', file=f)
        if self.emaxpos is not None:
            print('  emaxpos=' + utils.num2str(self.emaxpos) + ',', file=f)
        if self.eopreg is not None:
            print('  eopreg=' + utils.num2str(self.eopreg) + ',', file=f)
        if self.eamp is not None:
            print('  eamp=' + utils.num2str(self.eamp) + ',', file=f)
        if self.clambda is not None:
            print('  lambda=' + utils.num2str(self.clambda) + ',', file=f)
        if self.report is not None:
            print('  report=' + str(self.report) + ',', file=f)
        if self.lspinorb is not None:
            print('  lspinorb=' + utils.bool2str(self.lspinorb) + ',', file=f)
        if self.esm_bc is not None:
            print('  esm_bc=\'' + self.esm_bc + '\',', file=f)
        if self.esm_w is not None:
            print('  esm_w=' + utils.num2str(self.esm_w) + ',', file=f)
        if self.esm_efield is not None:
            print(
                '  esm_efield=' + utils.num2str(self.esm_efield) + ',', file=f)
        if self.esm_nfit is not None:
            print('  esm_nfit=' + str(self.esm_nfit) + ',', file=f)
        if self.london is not None:
            print('  london=' + utils.bool2str(self.london) + ',', file=f)
        if self.london_s6 is not None:
            print('  london_s6=' + utils.num2str(self.london_s6) + ',', file=f)
        if self.london_rcut is not None:
            print(
                '  london_rcut=' + utils.num2str(self.london_rcut) + ',',
                file=f)
        if self.xdm is not None:
            print('  xdm=' + utils.bool2str(self.xdm) + ',', file=f)
        if self.xdm_a1 is not None:
            print('  xdm_a1=' + utils.num2str(self.xdm_a1) + ',', file=f)
        if self.xdm_a2 is not None:
            print('  xdm_a2=' + utils.num2str(self.xdm_a2) + ',', file=f)
        if self.vdw_corr is not None:
            print('  vdw_corr=\'' + self.vdw_corr + '\',', file=f)
        if self.ts_vdw_econv_thr is not None:
            print(
                '  ts_vdw_econv_thr=' + utils.num2str(self.ts_vdw_econv_thr) +
                ',',
                file=f)
        if self.ts_vdw_isolated is not None:
            print(
                '  ts_vdw_isolated=' + utils.bool2str(self.tsw_vdw_isolated) +
                ',',
                file=f)
        if self.fcp_mu is not None:
            print('  fcp_mu=' + utils.num2str(self.fcp_mu) + ',', file=f)
        if self.esm_a is not None:
            print('  esm_a=' + utils.num2str(self.esm_a) + ',', file=f)

        # &ELECTRONS ###
        print('/\n&ELECTRONS', file=f)
        try:
            diag = self.convergence['diag']
            print('  diagonalization=\'' + diag + '\',', file=f)
        except BaseException:
            pass

        print('  conv_thr=' + utils.num2str(self.conv_thr) + ',', file=f)

        for x in list(self.convergence.keys()):
            if x == 'mixing':
                print(
                    '  mixing_beta=' + utils.num2str(self.convergence[x]) +
                    ',',
                    file=f)
            elif x == 'maxsteps':
                print(
                    '  electron_maxstep=' + str(self.convergence[x]) + ',',
                    file=f)
            elif x == 'nmix':
                print(
                    '  mixing_ndim=' + str(self.convergence[x]) + ',', file=f)
            elif x == 'mixing_mode':
                print('  mixing_mode=\'' + self.convergence[x] + '\',', file=f)
            elif x == 'diago_cg_maxiter':
                print(
                    '  diago_cg_maxiter=' + str(self.convergence[x]) + ',',
                    file=f)
        if self.startingpot is not None:
            print('  startingpot=\'' + self.startingpot + '\',', file=f)
        if self.startingwfc is not None:
            print('  startingwfc=\'' + self.startingwfc + '\',', file=f)

        # automatically generated parameters
        if self.electron_maxstep is not None:
            print(
                '  electron_maxstep=' + str(self.electron_maxstep) + ',',
                file=f)
        if self.scf_must_converge is not None:
            print(
                '  scf_must_converge=' +
                utils.bool2str(self.scf_must_converge) + ',',
                file=f)
        if self.adaptive_thr is not None:
            print(
                '  adaptive_thr=' + utils.bool2str(self.adaptive_thr) + ',',
                file=f)
        if self.conv_thr_init is not None:
            print(
                '  conv_thr_init=' + utils.num2str(self.conv_thr_init) + ',',
                file=f)
        if self.conv_thr_multi is not None:
            print(
                '  conv_thr_multi=' + utils.num2str(self.conv_thr_multi) + ',',
                file=f)
        if self.mixing_beta is not None:
            print(
                '  mixing_beta=' + utils.num2str(self.mixing_beta) + ',',
                file=f)
        if self.mixing_ndim is not None:
            print('  mixing_ndim=' + str(self.mixing_ndim) + ',', file=f)
        if self.mixing_fixed_ns is not None:
            print(
                '  mixing_fixed_ns=' + str(self.mixing_fixed_ns) + ',', file=f)
        if self.ortho_para is not None:
            print('  ortho_para=' + str(self.ortho_para) + ',', file=f)
        if self.diago_thr_init is not None:
            print(
                '  diago_thr_init=' + utils.num2str(self.diago_thr_init) + ',',
                file=f)
        if self.diago_cg_maxiter is not None:
            print(
                '  diago_cg_maxiter=' + str(self.diago_cg_maxiter) + ',',
                file=f)
        if self.diago_david_ndim is not None:
            print(
                '  diago_david_ndim=' + str(self.diago_david_ndim) + ',',
                file=f)
        if self.diago_full_acc is not None:
            print(
                '  diago_full_acc=' + utils.bool2str(self.diago_full_acc) +
                ',',
                file=f)
        if self.efield is not None:
            print('  efield=' + utils.num2str(self.efield) + ',', file=f)
        if self.tqr is not None:
            print('  tqr=' + utils.bool2str(self.tqr) + ',', file=f)

        # &IONS ###
        if not ionssec:
            simpleconstr, otherconstr = [], []
        else:
            simpleconstr, otherconstr = utils.convert_constraints(self.atoms)

        if self.ion_dynamics is None:
            self.optdamp = False
        else:
            self.optdamp = (self.ion_dynamics.upper() == 'DAMP')
        if self.ion_dynamics is not None and ionssec:
            if len(otherconstr) != 0:
                print('/\n&IONS\n  ion_dynamics=\'damp\',', file=f)
                self.optdamp = True
            else:
                print(
                    '/\n&IONS\n  ion_dynamics=\'' + self.ion_dynamics + '\',',
                    file=f)
            if self.ion_positions is not None:
                print(
                    '  ion_positions=\'' + self.ion_positions + '\',', file=f)
        elif self.ion_positions is not None:
            print(
                '/\n&IONS\n  ion_positions=\'' + self.ion_positions + '\',',
                file=f)

        # automatically generated parameters
        if self.remove_rigid_rot is not None:
            print(
                '  remove_rigid_rot=' + utils.bool2str(self.remove_rigid_rot) +
                ',',
                file=f)
        if self.tempw is not None:
            print('  tempw=' + utils.num2str(self.tempw) + ',', file=f)
        if self.tolp is not None:
            print('  tolp=' + utils.num2str(self.tolp) + ',', file=f)
        if self.delta_t is not None:
            print('  delta_t=' + utils.num2str(self.delta_t) + ',', file=f)
        if self.nraise is not None:
            print('  nraise=' + str(self.nraise) + ',', file=f)
        if self.refold_pos is not None:
            print(
                '  refold_pos=' + utils.bool2str(self.refold_pos) + ',',
                file=f)
        if self.upscale is not None:
            print('  upscale=' + utils.num2str(self.upscale) + ',', file=f)
        if self.bfgs_ndim is not None:
            print('  bfgs_ndim=' + str(self.bfgs_ndim) + ',', file=f)
        if self.trust_radius_max is not None:
            print(
                '  trust_radius_max=' + utils.num2str(self.trust_radius_max) +
                ',',
                file=f)
        if self.trust_radius_min is not None:
            print(
                '  trust_radius_min=' + utils.num2str(self.trust_radius_min) +
                ',',
                file=f)
        if self.trust_radius_ini is not None:
            print(
                '  trust_radius_ini=' + utils.num2str(self.trust_radius_ini) +
                ',',
                file=f)
        if self.w_1 is not None:
            print('  w_1=' + utils.num2str(self.w_1) + ',', file=f)
        if self.w_2 is not None:
            print('  w_2=' + utils.num2str(self.w_2) + ',', file=f)

        # &CELL ###
        if self.cell_dynamics is not None:
            print(
                '/\n&CELL\n  cell_dynamics=\'' + self.cell_dynamics + '\',',
                file=f)
            if self.press is not None:
                print('  press=' + utils.num2str(self.press) + ',', file=f)
            if self.dpress is not None:
                print(
                    '  press_conv_thr=' + utils.num2str(self.dpress) + ',',
                    file=f)
            if self.cell_factor is not None:
                print(
                    '  cell_factor=' + utils.num2str(self.cell_factor) + ',',
                    file=f)
            if self.cell_dofree is not None:
                print('  cell_dofree=\'' + self.cell_dofree + '\',', file=f)

        # automatically generated parameters
        if self.wmass is not None:
            print('  wmass=' + utils.num2str(self.wmass) + ',', file=f)
        if self.press_conv_thr is not None:
            print(
                '  press_conv_thr=' + utils.num2str(self.press_conv_thr) + ',',
                file=f)

        # CELL_PARAMETERS
        print('/\nCELL_PARAMETERS {angstrom}', file=f)
        for i in range(3):
            print(
                '%21.15fd0 %21.15fd0 %21.15fd0' % tuple(self.atoms.cell[i]),
                file=f)

        print('ATOMIC_SPECIES', file=f)
        for species in self.species:  # PSP ORDERING FOLLOWS SPECIESINDEX
            spec = self.specdict[species]
            print(species, utils.num2str(spec.mass), spec.s + '.UPF', file=f)

        print('ATOMIC_POSITIONS {crystal}', file=f)
        if len(simpleconstr) == 0:
            for species, pos in self.specprops:
                print(
                    '%-4s %21.15fd0 %21.15fd0 %21.15fd0' % (species, pos[0],
                                                            pos[1], pos[2]),
                    file=f)
        else:
            for i, (species, pos) in enumerate(self.specprops):
                print(
                    '%-4s %21.15fd0 %21.15fd0 %21.15fd0   %d  %d  %d' %
                    (species, pos[0], pos[1], pos[2], simpleconstr[i][0],
                     simpleconstr[i][1], simpleconstr[i][2]),
                    file=f)

        if len(otherconstr) != 0:
            print('CONSTRAINTS', file=f)
            if self.constr_tol is None:
                print(len(otherconstr), file=f)
            else:
                print(len(otherconstr), utils.num2str(self.constr_tol), file=f)
            for x in otherconstr:
                print(x, file=f)

        if overridekpts is None:
            kp = self.kpts
        else:
            kp = overridekpts
        if kp is 'gamma':
            print('K_POINTS Gamma', file=f)
        else:
            x = np.shape(kp)
            if len(x) == 1:
                print('K_POINTS automatic', file=f)
                print(kp[0], kp[1], kp[2], end=' ', file=f)
                if overridekptshift is None:
                    print(
                        self.kptshift[0],
                        self.kptshift[1],
                        self.kptshift[2],
                        file=f)
                else:
                    print(
                        overridekptshift[0],
                        overridekptshift[1],
                        overridekptshift[2],
                        file=f)
            else:
                print('K_POINTS crystal', file=f)
                print(x[0], file=f)
                w = 1. / x[0]
                for k in kp:
                    if len(k) == 3:
                        print(
                            '%24.15e %24.15e %24.15e %24.15e' % (k[0], k[1],
                                                                 k[2], w),
                            file=f)
                    else:
                        print(
                            '%24.15e %24.15e %24.15e %24.15e' % (k[0], k[1],
                                                                 k[2], k[3]),
                            file=f)

        f.close()

    def atoms2species(self):
        """Define several properties of the quantum espresso species
        from the ase atoms object. Takes into account that different
        spins (or different U etc.) on same kind of chemical elements
        are considered different species in quantum espresso.
        """
        symbols = self.atoms.get_chemical_symbols()
        masses = self.atoms.get_masses()
        magmoms = list(self.atoms.get_initial_magnetic_moments())
        if len(magmoms) < len(symbols):
            magmoms += list(np.zeros(len(symbols) - len(magmoms), np.float))
        pos = self.atoms.get_scaled_positions()

        self.species = []
        self.specprops = []
        dic = {}
        symcounter = {}
        for s in symbols:
            symcounter[s] = 0
        for i in range(len(symbols)):
            key = symbols[i] + '_m%.14e' % (magmoms[i])
            if key in dic:
                self.specprops.append((dic[key][1], pos[i]))
            else:
                symcounter[symbols[i]] += 1
                spec = symbols[i] + str(symcounter[symbols[i]])
                dic[key] = [i, spec]
                self.species.append(spec)
                self.specprops.append((spec, pos[i]))

        self.nspecies = len(self.species)
        self.specdict = {}
        for i, s in list(dic.values()):
            if np.isnan(masses[i]):
                mi = 0.0
            else:
                mi = masses[i]
            self.specdict[s] = utils.SpecObject(
                s=s.strip('0123456789'),  # chemical symbol w/o index
                mass=mi,
                magmom=magmoms[i])

    def get_nvalence(self):
        nel = {}
        for x in self.species:
            el = self.specdict[x].s
            # get number of valence electrons from pseudopotential or paw setup
            p = Popen(
                'egrep -i \'z\ valence|z_valence\' ' + self.psppath +
                '/' + el + '.UPF | tr \'"\' \' \'',
                shell=True, stdout=PIPE).stdout
            out = p.readline().decode('utf-8')
            for y in out.split():
                if y[0].isdigit() or y[0] == '.':
                    nel[el] = int(round(float(y)))
                    break
            p.close()
        nvalence = np.zeros(len(self.specprops), np.int)
        for i, x in enumerate(self.specprops):
            nvalence[i] = nel[self.specdict[x[0]].s]
        return nvalence, nel

    def set_atoms(self, atoms):
        if self.atoms is None or not self.started:
            self.atoms = atoms.copy()
        else:
            if len(atoms) != len(self.atoms):
                self.stop()
                self.nvalence = None
                self.nel = None
                self.recalculate = True

            x = atoms.cell - self.atoms.cell
            if np.max(x) > 1E-13 or np.min(x) < -1E-13:
                self.stop()
                self.recalculate = True
            if (atoms.get_atomic_numbers() !=
                    self.atoms.get_atomic_numbers()).any():
                self.stop()
                self.nvalence = None
                self.nel = None
                self.recalculate = True
            x = atoms.positions - self.atoms.positions
            if np.max(x) > 1E-13 or np.min(x) < - \
                    1E-13 or (not self.started and not self.got_energy):
                self.recalculate = True
        self.atoms = atoms.copy()

    def read(self, atoms):

        # This should only be done once
        stdout = check_output(['which', 'pw.x']).decode()
        self.exedir = os.path.dirname(stdout)

        with open(self.log, 'a') as f:
            f.write('  python dir          : {}\n'.format(self.mypath))
            f.write('  espresso dir        : {}\n'.format(self.exedir))
            f.write('  pseudo dir          : {}\n'.format(self.psppath))

        if not self.started:
            fresh = True
            self.initialize(atoms)
        else:
            fresh = False

        if self.recalculate:
            if not fresh:
                self.cinp.flush()
            s = open(self.log, 'a')
            a = self.cout.readline().decode('utf-8')
            s.write(a)
            atom_occ = {}
            magmoms = np.zeros(len(atoms))
            while (a != '' and
                   a[:17] != '!    total energy' and
                   a[:13] != '     stopping' and
                   a[:20] != '     convergence NOT'):
                a = self.cout.readline().decode('utf-8')
                s.write(a)
                s.flush()

                if a[:19] == '     iteration #  1':
                    while (a != '' and a[:17] != '!    total energy'
                           and a[:13] != '     stopping'
                           and a[:20] != '     convergence NOT'
                           and a[:22] != ' --- exit write_ns ---'):
                        a = self.cout.readline().decode('utf-8')
                        s.write(a)
                        s.flush()
                        if a[:5] == 'atom ':
                            atomnum = int(a[8:10])
                            # 'atom    1   Tr[ns(na)] =   1.00000'
                            if a[12:25] == 'Tr[ns(na)] = ':
                                N0 = float(a[27:35]) / 2.
                            elif a[12:42] == 'Tr[ns(na)] (up, down, total) =':
                                N0 = [
                                    float(a[42:52]),
                                    float(a[53:62]),
                                    float(a[63:71])
                                ]
                                N0 = N0[-1]  # only taking the total occupation
                            atom_occ[atomnum - 1] = {}
                            atom_occ[atomnum - 1][0] = N0
                if a[:39] == '     End of self-consistent calculation':
                    while (a != '' and
                           a[:17] != '!    total energy' and
                           a[:13] != '     stopping' and
                           a[:20] != '     convergence NOT'):
                        a = self.cout.readline().decode('utf-8')
                        s.write(a)
                        s.flush()
                        if a[:5] == 'atom ':
                            atomnum = int(a[8:10])
                            if a[12:25] == 'Tr[ns(na)] = ':
                                Nks = float(a[27:35]) / 2.
                            elif a[12:42] == 'Tr[ns(na)] (up, down, total) =':
                                Nks = [
                                    float(a[42:52]),
                                    float(a[53:62]),
                                    float(a[63:71])
                                ]
                                # only taking the total occupation
                                Nks = Nks[-1]
                                magmom = Nks[0] - Nks[1]
                                magmoms[atomnum] = magmom
                            atom_occ[atomnum - 1]['ks'] = Nks
                    break
            if a[:20] == '     convergence NOT':
                self.stop()
                raise KohnShamConvergenceError(
                    'scf cycles did not converge\nincrease maximum '
                    'number of steps and/or decreasing mixing'
                )
            elif a[:13] == '     stopping':
                self.stop()
                self.checkerror()
                # if checkerror shouldn't find an error here,
                # throw this generic error
                raise RuntimeError('SCF calculation failed')
            elif a == '' and self.calculation in ('relax', 'scf', 'vc-relax',
                                                  'vc-md', 'md'):
                self.checkerror()
                # if checkerror shouldn't find an error here,
                # throw this generic error
                raise RuntimeError('SCF calculation failed')
            self.atom_occ = atom_occ
            self.results['magmoms'] = magmoms
            self.results['magmom'] = np.sum(magmoms)
            if self.calculation in ('relax', 'scf', 'vc-relax', 'vc-md', 'md'):
                self.energy_free = float(a.split()[-2]) * Rydberg
                # get S*T correction (there is none for Marzari-Vanderbilt=Cold
                # smearing)
                if (self.occupations == 'smearing' and
                    self.smearing[0].upper() != 'M' and
                    self.smearing[0].upper() != 'C' and
                    not self.optdamp):
                    a = self.cout.readline().decode('utf-8')
                    s.write(a)
                    exx = False
                    while a[:13] != '     smearing':
                        a = self.cout.readline().decode('utf-8')
                        s.write(a)
                        if a.find('EXX') > -1:
                            exx = True
                            break
                    if exx:
                        self.ST = 0.0
                        self.energy_zero = self.energy_free
                    else:
                        self.ST = -float(a.split()[-2]) * Rydberg
                        self.energy_zero = self.energy_free + 0.5 * self.ST
                else:
                    self.ST = 0.0
                    self.energy_zero = self.energy_free
            else:
                self.energy_free = None
                self.energy_zero = None

            self.got_energy = True
            self.results['energy'] = self.energy_zero
            self.results['free_energy'] = self.energy_free

            a = self.cout.readline().decode('utf-8')
            s.write(a)
            s.flush()

            if self.calculation in ('relax', 'scf', 'vc-relax', 'vc-md', 'md'):
                a = self.cout.readline().decode('utf-8')
                s.write(a)
                if not self.dontcalcforces:
                    while a[:11] != '     Forces':
                        a = self.cout.readline().decode('utf-8')
                        s.write(a)
                        s.flush()
                    a = self.cout.readline().decode('utf-8')
                    s.write(a)
                    self.forces = np.empty((self.natoms, 3), np.float)
                    for i in range(self.natoms):
                        a = self.cout.readline().decode('utf-8')
                        while a.find('force') < 0:
                            s.write(a)
                            a = self.cout.readline().decode('utf-8')
                        s.write(a)
                        forceinp = a.split()
                        self.forces[i][:] = [
                            float(x) for x in forceinp[len(forceinp) - 3:]
                        ]
                    self.forces *= (Rydberg / Bohr)
                else:
                    self.forces = None
            else:
                self.forces = None
            self.recalculate = False
            s.close()
            self.results['forces'] = self.forces
            self.stop()

            # get final energy and forces for internal QE relaxation run
            if self.calculation in ('relax', 'vc-relax', 'vc-md', 'md'):
                p = Popen(
                    'grep -a -n "!    total" ' + self.log + ' | tail -1',
                    shell=True, stdout=PIPE).stdout
                n = int(p.readline().decode('utf-8').split(':')[0]) - 1
                f = open(self.log, 'r')
                for i in range(n):
                    f.readline()
                self.energy_free = float(f.readline().split()[-2]) * Rydberg
                # get S*T correction (there is none for Marzari-Vanderbilt=Cold
                # smearing)
                if (self.occupations == 'smearing' and
                    self.smearing[0].upper() != 'M' and
                    self.smearing[0].upper() != 'C' and
                    not self.optdamp):
                    a = f.readline()
                    exx = False
                    while a[:13] != '     smearing':
                        a = f.readline()
                        if a.find('EXX') > -1:
                            exx = True
                            break
                    if exx:
                        self.ST = 0.0
                        self.energy_zero = self.energy_free
                    else:
                        self.ST = -float(a.split()[-2]) * Rydberg
                        self.energy_zero = self.energy_free + 0.5 * self.ST
                else:
                    self.ST = 0.0
                    self.energy_zero = self.energy_free

                if (self.U_projection_type == 'atomic' and not
                    self.dontcalcforces):
                    a = f.readline()
                    while a[:11] != '     Forces':
                        a = f.readline()
                    f.readline()
                    self.forces = np.empty((self.natoms, 3), np.float)
                    for i in range(self.natoms):
                        a = f.readline()
                        while a.find('force') < 0:
                            a = f.readline()
                        forceinp = a.split()
                        self.forces[i][:] = [
                            float(x) for x in forceinp[len(forceinp) - 3:]
                        ]
                    self.forces *= (Rydberg / Bohr)
                f.close()

            self.checkerror()

    def initialize(self, atoms):
        """Create the pw.inp input file and start the calculation."""
        if not self.started:
            self.atoms = atoms.copy()
            self.atoms2species()
            self.natoms = len(self.atoms)
            self.check_spinpol()
            self.write_input()

        if self.cancalc:
            self.start()

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

    def start(self):
        if not self.started:
            if self.single_calculator:
                while len(espresso_calculators) > 0:
                    espresso_calculators.pop().stop()
                espresso_calculators.append(self)
            if site.batch:
                cdir = os.getcwd()
                os.chdir(self.localtmp)
                call(site.perHostMpiExec + ' cp ' + self.localtmp +
                     '/pw.inp ' + self.scratch, shell=True)

                if not self.proclist:
                    self.cinp, self.cout = site.do_perProcMpiExec(
                        self.scratch, self.exedir + 'pw.x ' +
                        self.parflags + ' -in pw.inp')
                else:
                    (self.cinp,
                     self.cout,
                     self.cerr) = site.do_perSpecProcMpiExec(
                        self.mycpus, self.myncpus, self.scratch,
                        self.exedir + 'pw.x ' + self.parflags +
                        ' -in pw.inp|' + self.mypath + '/espfilter ' + str(
                            self.natoms) + ' ' + self.log + '0')
                os.chdir(cdir)
            else:
                call('cp ' + self.localtmp + '/pw.inp ' + self.scratch,
                     shell=True)
                cmd = 'cd ' + self.scratch + ' ; ' + self.exedir + 'pw.x ' + self.serflags + ' -in pw.inp'
                p = Popen(cmd, shell=True, stdin=PIPE,
                          stdout=PIPE, close_fds=True)
                self.cinp, self.cout = (p.stdin, p.stdout)

            self.started = True

    def stop(self):
        if self.started:
            self.cinp.flush()
            s = open(self.log, 'a')
            a = self.cout.readline().decode('utf-8')
            s.write(a)
            while a != '':
                a = self.cout.readline().decode('utf-8')
                s.write(a)
                s.flush()
            s.close()
            self.cinp.close()
            self.cout.close()
            self.started = False

    def topath(self, filename):
        if os.path.isabs(filename):
            return filename
        else:
            return os.path.join(self.sdir, filename)

    def get_nonselfconsistent_energies(self, type='beefvdw'):
        self.stop()
        cmd = 'grep -a -32 "BEEF-vdW xc energy contributions" ' + self.log + ' | tail -32'
        p = Popen(cmd, shell=True, stdout=PIPE).stdout
        s = p.readlines()
        xc = np.array([])
        for i, l in enumerate(s):
            l_ = float(l.split(":")[-1]) * Rydberg
            xc = np.append(xc, l_)
        assert len(xc) == 32
        return xc

    def checkerror(self):
        cmd = 'grep -a -n Giannozzi ' + self.log + ' | tail -1'
        p = Popen(cmd, shell=True, stdout=PIPE).stdout
        try:
            out = p.readline().decode('utf-8')
            n = int(out.split()[0].strip(':'))
        except BaseException:
            raise RuntimeError(
                'Espresso executable doesn\'t seem to have been started.')

        cmd = ('tail -n +%d ' % n) + self.log + ' | grep -a -n %%%%%%%%%%%%%%%% |tail -2'
        p = Popen(cmd, shell=True, stdout=PIPE).stdout
        s = p.readlines()

        if len(s) < 2:
            return

        a = int(s[0].decode('utf-8').split()[0].strip(':')) + 1
        b = int(s[1].decode('utf-8').split()[0].strip(':')) - a

        if b < 1:
            return

        cmd = ('tail -n +%d ' % (a + n - 1)) + self.log + ('|head -%d' % b)
        p = Popen(cmd, shell=True, stdout=PIPE).stdout
        err = p.readlines()

        if err[0].decode('utf-8').lower().find('error') < 0:
            return

        msg = ''
        for e in err:
            msg += e.decode('utf-8')
        raise RuntimeError(msg[:len(msg) - 1])

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

    def get_fermi_level(self):
        efermi = self.inputfermilevel
        if efermi:
            return efermi

        self.stop()
        efermi = utils.grepy(self.log, 'Fermi energy')
        if efermi is not None:
            efermi = float(efermi.split()[-2])

        return efermi


DEFAULTS = {
    'CONTROL': {
        'calculation': 'scf',
        'title': None,
        'verbosity': None,
        'restart_mode': None,
        'wf_collect': True,
        'nstep': None,
        'iprint': None,
        'tstress': None,
        'tprnfor': None,
        'dt': None,
        'outdir': '.',
        'wfcdir': None,
        'prefix': None,
        'lkpoint_dir': None,
        'max_seconds': None,
        'etot_conv_thr': None,
        'forc_conv_thr': None,
        'disk_io': None,
        'pseudo_dir': None,
        'tefield': None,
        'dipfield': None,
        'lelfield': None,
        'nberrycyc': None,
        'lorbm': None,
        'lberry': None,
        'gdir': None,
        'nppstr': None,
        'lfcpopt': None,
        'gate': None
    },
    'SYSTEM': {
        'ibrav',
        'celldm',
        'A',
        'B',
        'C',
        'cosAB',
        'cosAC',
        'cosBC',
        'nat',
        'ntyp',
        'nbnd',
        'tot_charge',
        'tot_magnetization',
        'starting_magnetization',
        'ecutwfc',
        'ecutrho',
        'ecutfock',
        'nr1',
        'nr2',
        'nr3',
        'nr1s',
        'nr2s',
        'nr3s',
        'nosym',
        'nosym_evc',
        'noinv',
        'no_t_rev',
        'force_symmorphic',
        'use_all_frac',
        'occupations',
        'one_atom_occupations',
        'starting_spin_angle',
        'degauss',
        'smearing',
        'nspin',
        'noncolin',
        'ecfixed',
        'qcutz',
        'q2sigma',
        'input_dft',
        'exx_fraction',
        'screening_parameter',
        'exxdiv_treatment',
        'x_gamma_extrapolation',
        'ecutvcut',
        'nqx1',
        'nqx2',
        'nqx3',
        'lda_plus_u',
        'lda_plus_u_kind',
        'Hubbard_U',
        'Hubbard_J0',
        'Hubbard_alpha',
        'Hubbard_beta',
        'Hubbard_J',
        'starting_ns_eigenvalue',
        'U_projection_type',
        'edir',
        'emaxpos',
        'eopreg',
        'eamp',
        'angle1',
        'angle2',
        'constrained_magnetization',
        'fixed_magnetization',
        'lambda',
        'report',
        'lspinorb',
        'assume_isolated',
        'esm_bc',
        'esm_w',
        'esm_efield',
        'esm_nfit',
        'fcp_mu',
        'vdw_corr',
        'london',
        'london_s6',
        'london_c6',
        'london_rvdw',
        'london_rcut',
        'ts_vdw_econv_thr',
        'ts_vdw_isolated',
        'xdm',
        'xdm_a1',
        'xdm_a2',
        'space_group',
        'uniqueb',
        'origin_choice',
        'rhombohedral',
        'zmon',
        'realxz',
        'block',
        'block_1',
        'block_2',
        'block_height'
    },
    'ELECTRONS': {
        'electron_maxstep',
        'scf_must_converge',
        'conv_thr',
        'adaptive_thr',
        'conv_thr_init',
        'conv_thr_multi',
        'mixing_mode',
        'mixing_beta',
        'mixing_ndim',
        'mixing_fixed_ns',
        'diagonalization',
        'ortho_para',
        'diago_thr_init',
        'diago_cg_maxiter',
        'diago_david_ndim',
        'diago_full_acc',
        'efield',
        'efield_cart',
        'efield_phase',
        'startingpot',
        'startingwfc',
        'tqr'
    },
    'IONS': {
        'ion_dynamics',
        'ion_positions',
        'pot_extrapolation',
        'wfc_extrapolation',
        'remove_rigid_rot',
        'ion_temperature',
        'tempw',
        'tolp',
        'delta_t',
        'nraise',
        'refold_pos',
        'upscale',
        'bfgs_ndim',
        'trust_radius_max',
        'trust_radius_min',
        'trust_radius_ini',
        'w_1',
        'w_2'},
    'CELL': {
        'cell_dynamics',
        'press',
        'wmass',
        'cell_factor',
        'press_conv_thr',
        'cell_dofree'
    }}
