from . import utils
import numpy as np
from ase.units import Rydberg, Bohr
from subprocess import Popen, PIPE, call, check_output
import tarfile
import os


class Mixins:

    def write_input(self,
                    filename='input.pwi',
                    overridenbands=None,
                    usetetrahedra=False):
        if self.atoms is None:
            raise ValueError('no atoms defined')

        # Open the write filen
        f = open(filename, 'w')

        ionssec = self.calculation not in ('scf', 'nscf', 'bands')

        # &CONTROL ###
        print(
            "&CONTROL\n"
            "  calculation='{}',\n"
            "  prefix='calc',".format(self.calculation), file=f)

        efield = (self.field['status'])
        dipfield = (self.dipole['status'])

        if efield or dipfield:
            print('  tefield=.true.,', file=f)
            if dipfield:
                print('  dipfield=.true.,', file=f)

        print('  tprnfor=.true.,', file=f)
        if self.calcstress:
            print('  tstress=.true.,', file=f)

        if self.disk_io is None:
            self.disk_io = 'none'

        if self.disk_io in ['high', 'low', 'none']:
            print('  disk_io=\'' + self.disk_io + '\',', file=f)

        if self.wf_collect:
            print('  wf_collect=.true.,', file=f)

        # We basically ignore convergence of total energy differences
        # between ionic steps and only consider fmax as in ase
        print('  etot_conv_thr=1d0,', file=f)
        self.forc_conv_thr /= Rydberg / Bohr
        print('  forc_conv_thr={},'.format(self.forc_conv_thr), file=f)

        ### &SYSTEM ###
        print('/\n&SYSTEM\n  ibrav=0,', file=f)
        print('  nat=' + str(self.natoms) + ',', file=f)
        self.atoms2species()
        print('  ntyp=' + str(self.nspecies) + ',', file=f)

        if self.tot_charge is not None:
            print('  tot_charge=' + utils.num2str(self.tot_charge) + ',', file=f)

        if self.fix_magmom:
            assert self.spinpol
            self.totmag = self.summed_magmoms
            print('  tot_magnetization=' + utils.num2str(self.totmag) + ',', file=f)
        elif self.tot_magnetization != -1:
            self.totmag = self.tot_magnetization
            print('  tot_magnetization=' + utils.num2str(self.totmag) + ',', file=f)

        print('  ecutwfc=' + utils.num2str(self.ecutwfc / Rydberg) + ',', file=f)
        print('  ecutrho=' + utils.num2str(self.ecutrho / Rydberg) + ',', file=f)
        if self.ecutfock is not None:
            print('  ecutfock=' + utils.num2str(self.ecutfock / Rydberg) + ',', file=f)

        if self.nbnd is not None:
            # set number of bands
            if self.nbnd > 0:
                self.nbnd = int(self.nbnd)
            else:
                # if self.nbnd is negative create -self.nbnd extra bands
                if self.nvalence is None:
                    self.nvalence, self.nel = self.get_nvalence()
                if self.noncollinear:
                    self.nbnd = int(np.sum(self.nvalence) - self.nbnd * 2.)
                else:
                    self.nbnd = int(np.sum(self.nvalence) / 2. - self.nbnd)
            print('  nbnd=' + str(self.nbnd) + ',', file=f)

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
                print('  starting_magnetization(%d)=%s,' % (spcount, utils.num2str(float(mag))), file=f)
                spcount += 1

        elif self.noncollinear:
            print('  noncolin=.true.,', file=f)
            if self.spinorbit:
                print('  lspinorb=.true.', file=f)
            spcount = 1
            if self.nel is None:
                self.nvalence, self.nel = self.get_nvalence()

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

        # &ELECTRONS ###
        print('/\n&ELECTRONS', file=f)

        print('  diagonalization=\'' + self.diagonalization + '\',', file=f)

        self.conv_thr /= Rydberg
        print('  conv_thr=' + utils.num2str(self.conv_thr) + ',', file=f)

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
                print('/\n&IONS\n  ion_dynamics=\'' + self.ion_dynamics + '\',', file=f)
            if self.ion_positions is not None:
                print('  ion_positions=\'' + self.ion_positions + '\',', file=f)
        elif self.ion_positions is not None:
            print('/\n&IONS\n  ion_positions=\'' + self.ion_positions + '\',', file=f)

        # &CELL ###
        if self.cell_dynamics is not None:
            print('/\n&CELL\n  cell_dynamics=\'' + self.cell_dynamics + '\',', file=f)
            if self.press is not None:
                print('  press=' + utils.num2str(self.press) + ',', file=f)
            if self.dpress is not None:
                print('  press_conv_thr=' + utils.num2str(self.dpress) + ',', file=f)
            if self.cell_factor is not None:
                print('  cell_factor=' + utils.num2str(self.cell_factor) + ',', file=f)
            if self.cell_dofree is not None:
                print('  cell_dofree=\'' + self.cell_dofree + '\',', file=f)

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
        """Create the input.pwi input file and start the calculation."""
        if not self.started:
            stdout = check_output(['which', 'pw.x']).decode()
            self.exedir = os.path.dirname(stdout)

            with open(self.log, 'a') as f:
                f.write('  python dir          : {}\n'.format(self.mypath))
                f.write('  espresso dir        : {}\n'.format(self.exedir))
                f.write('  pseudo dir          : {}\n'.format(self.psppath))

            self.atoms = atoms.copy()
            self.atoms2species()
            self.natoms = len(self.atoms)
            self.check_spinpol()
            self.write_input()

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
                while len(self.calculators) > 0:
                    self.calculators.pop().stop()
                self.calculators.append(self)
            if self.site.batch:
                cdir = os.getcwd()
                os.chdir(self.localtmp)
                call(self.site.perHostMpiExec + ' cp ' + self.localtmp +
                     '/input.pwi ' + self.scratch, shell=True)

                if not self.proclist:
                    self.cinp, self.cout = self.site.do_perProcMpiExec(
                        self.scratch, self.exedir + 'pw.x ' +
                        self.parflags + ' -in input.pwi')
                else:
                    (self.cinp,
                     self.cout,
                     self.cerr) = self.site.do_perSpecProcMpiExec(
                        self.mycpus, self.myncpus, self.scratch,
                        self.exedir + 'pw.x ' + self.parflags +
                        ' -in input.pwi|' + self.mypath + '/espfilter ' + str(
                            self.natoms) + ' ' + self.log + '0')
                os.chdir(cdir)
            else:
                call('cp ' + self.localtmp + '/input.pwi ' + self.scratch,
                     shell=True)
                cmd = 'cd ' + self.scratch + ' ; ' + self.exedir + 'pw.x ' + self.serflags + ' -in input.pwi'
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

    def find_max_empty_space(self, edir=3):
        """Assuming periodic boundary conditions, finds the largest
        continuous segment of free, unoccupied space and returns
        its midpoint in scaled coordinates (0 to 1) in the edir
        direction (default z).
        """
        position_array = self.atoms.get_scaled_positions()[..., edir - 1]
        position_array.sort()
        differences = np.diff(position_array)
        differences = np.append(differences, position_array[0] + 1 - position_array[-1])
        max_diff_index = np.argmax(differences)
        if max_diff_index == len(position_array) - 1:
            return (position_array[0] + 1 + position_array[-1]) / 2. % 1
        else:
            return (position_array[max_diff_index] + position_array[max_diff_index + 1]) / 2.

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

