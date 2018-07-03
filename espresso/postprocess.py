from ase.units import Rydberg, Bohr, Hartree


class PostProcess:

    def run_espressox(self,
                      binary,
                      inp,
                      log=None,
                      piperead=False,
                      parallel=True):
        """Runs one of the .x binaries of the espresso suite inp is expected to
        be in self.localtmp log will be created in self.localtmp
        """
        ll = ''
        if log is not None:
            ll += ' >> {}/{}'.format(self.localtmp, log)

        if site.batch and parallel:
            cdir = os.getcwd()
            os.chdir(self.localtmp)
            call(site.perHostMpiExec + ' cp ' + self.localtmp + '/' +
                 inp + ' ' + self.scratch, shell=True)

            if piperead:
                p = site.do_perProcMpiExec_outputonly(
                    self.scratch,
                    binary + ' ' + self.parflags + ' -in ' + inp + ll)
            else:
                site.runonly_perProcMpiExec(
                    self.scratch,
                    binary + ' ' + self.parflags + ' -in ' + inp + ll)
            os.chdir(cdir)
        else:
            call('cp {}/{} {}'.format(self.localtmp, inp, self.scratch),
                 shell=True)
            cmd = 'cd {} ; {} {} -in {}'.format(
                self.scratch, binary, self.serflags, inp + ll)
            if piperead:
                p = Popen(cmd, shell=True, stdout=PIPE).stdout
            else:
                call(cmd, shell=True)

        if piperead:
            return p

    def run_ppx(self,
                inp,
                log=None,
                inputpp=[],
                plot=[],
                output_format=5,
                iflag=3,
                piperead=False,
                parallel=True):
        if 'disk_io' in self.output:
            if self.output['disk_io'] == 'none':
                print(
                    "run_ppx requires output['disk_io'] to "
                    "be at least 'low' and avoidio=False"
                )
        self.stop()

        with open('{}/{}'.format(self.localtmp, inp), 'w') as f:
            f.write('&INPUTPP\n  prefix=\'calc\',\n  outdir=\'.\',\n')
            for a, b in inputpp:
                if isinstance(b, float):
                    c = utils.num2str(b)
                elif isinstance(b, str):
                    c = "'{}'".format(b)
                else:
                    c = str(b)
                f.write('  {}={},\n'.format(a, c))
            f.write('/\n')
            f.write('&PLOT\n  iflag={},\n  output_format={},\n'.format(
                iflag, output_format))
            for a, b in plot:
                if isinstance(b, float):
                    c = utils.num2str(b)
                elif isinstance(b, str):
                    c = "'{}'".format(b)
                else:
                    c = str(b)
                f.write('  {}={},\n'.format(a, c))
            f.write('/\n')

        if piperead:
            return self.run_espressox(
                self.exedir + 'pp.x',
                inp,
                log=log,
                piperead=piperead,
                parallel=parallel)
        else:
            self.run_espressox(
                self.exedir + 'pp.x', inp, log=log, parallel=parallel)

    def calc_pdos(self,
                  Emin=None,
                  Emax=None,
                  DeltaE=None,
                  nscf=False,
                  tetrahedra=False,
                  slab=False,
                  kpts=None,
                  kptshift=None,
                  nbands=None,
                  ngauss=None,
                  sigma=None,
                  nscf_fermilevel=False,
                  add_higher_channels=True,
                  get_overlap_integrals=False):
        """Calculate (projected) density of states.
        - Emin,Emax,DeltaE define the energy window.
        - nscf=True will cause a non-selfconsistent calculation to be performed
          on top of a previous converged scf calculation, with the advantage
          that more kpts and more nbands can be defined improving the quality/
          increasing the energy range of the DOS.
        - tetrahedra=True (in addition to nscf=True) means use tetrahedron
          (i.e. smearing-free) method for DOS
        - slab=True: use triangle method insead of tetrahedron method
          (for 2D system perp. to z-direction)
        - sigma != None sets/overrides the smearing to calculate the DOS
          (also overrides tetrahedron/triangle settings)
        - get_overlap_integrals=True: also return k-point- and band-resolved
          projections (which are summed up and smeared to obtain the PDOS)

        Returns an array containing the energy window,
        the DOS over the same range,
        and the PDOS as an array (index: atom number 0..n-1) of dictionaries.
        The dictionary keys are the angular momentum channels 's','p','d'...
        (or e.g. 'p,j=0.5', 'p,j=1.5' in the case of LS-coupling).
        Each dictionary contains an array of arrays of the total and
        m-resolved PDOS over the energy window.
        In case of spin-polarization, total up is followed by total down, by
        first m with spin up, etc...

        Quantum Espresso with the tetrahedron method for PDOS can be
        obtained here:
        svn co --username anonymous http://qeforge.qe-forge.org/svn/q-e/branches/espresso-dynpy-beef
        """
        efermi = self.get_fermi_level()

        # run a nscf calculation with e.g. tetrahedra or more k-points etc.
        if nscf:
            if not hasattr(self, 'natoms'):
                self.atoms2species()
                self.natoms = len(self.atoms)
            self.writeinputfile(
                filename='pwnscf.inp',
                mode='nscf',
                usetetrahedra=tetrahedra,
                overridekpts=kpts,
                overridekptshift=kptshift,
                overridenbands=nbands,
                suppressforcecalc=True)
            self.run_espressox(self.exedir + 'pw.x', 'pwnscf.inp',
                               'pwnscf.log')
            if nscf_fermilevel:
                cmd = 'grep -a Fermi ' + self.localtmp + '/pwnscf.log|tail -1'
                p = Popen(cmd, shell=True, stdout=PIPE).stdout
                efermi = float(p.readline().decode('utf-8').split()[-2])

        # remove old wave function projections
        call('rm -f ' + self.scratch + '/*_wfc*', shell=True)
        # create input for projwfc.x
        f = open(self.localtmp + '/pdos.inp', 'w')
        print('&PROJWFC\n  prefix=\'calc\',\n  outdir=\'.\',', file=f)
        if Emin is not None:
            print('  Emin = ' + utils.num2str(Emin + efermi) + ',', file=f)
        if Emax is not None:
            print('  Emax = ' + utils.num2str(Emax + efermi) + ',', file=f)
        if DeltaE is not None:
            print('  DeltaE = ' + utils.num2str(DeltaE) + ',', file=f)
        if slab:
            print('  lslab = .true.,', file=f)
        if ngauss is not None:
            print('  ngauss = ' + str(ngauss) + ',', file=f)
        if sigma is not None:
            print(
                '  degauss = ' + utils.num2str(sigma / Rydberg) + ',', file=f)
        print('/', file=f)
        f.close()
        # run projwfc.x
        self.run_espressox('projwfc.x', 'pdos.inp', 'pdos.log')

        # read in total density of states
        dos = np.loadtxt(self.scratch + '/calc.pdos_tot')
        if len(dos[0]) > 3:
            nspin = 2
            self.dos_total = [dos[:, 1], dos[:, 2]]
        else:
            nspin = 1
            self.dos_total = dos[:, 1]
        self.dos_energies = dos[:, 0] - efermi
        npoints = len(self.dos_energies)

        channels = {'s': 0, 'p': 1, 'd': 2, 'f': 3}
        # read in projections onto atomic orbitals
        self.pdos = [{} for i in range(self.natoms)]
        cmd = 'ls ' + self.scratch + '/calc.pdos_atm*'
        p = Popen(cmd, shell=True, stdout=PIPE).stdout
        proj = p.readlines()

        proj.sort()
        for i, inp in enumerate(proj):
            inpfile = inp.strip().decode('utf-8')
            pdosinp = np.genfromtxt(inpfile)
            spl = inpfile.split('#')
            iatom = int(spl[1].split('(')[0]) - 1
            channel = spl[2].split('(')[1].rstrip(')').replace('_j', ',j=')
            jpos = channel.find('j=')
            if jpos < 0:
                # ncomponents = 2*l+1 +1  (latter for m summed up)
                ncomponents = (2 * channels[channel[0]] + 2) * nspin
            else:
                # ncomponents = 2*j+1 +1  (latter for m summed up)
                ncomponents = int(2. * float(channel[jpos + 2:])) + 2
            if channel not in self.pdos[iatom]:
                self.pdos[iatom][channel] = np.zeros((ncomponents, npoints),
                                                     np.float)
                first = True
            else:
                first = False
            if add_higher_channels or first:
                for j in range(ncomponents):
                    self.pdos[iatom][channel][j] += pdosinp[:, (j + 1)]

        if get_overlap_integrals:
            return (self.dos_energies,
                    self.dos_total,
                    self.pdos,
                    self.__get_atomic_projections__())
        else:
            return self.dos_energies, self.dos_total, self.pdos

    def calc_bandstructure(self,
                           kptpath,
                           nbands=None,
                           atomic_projections=False):
        """Calculate bandstructure along kptpath (= array of k-points).
        If nbands is not None, override number of bands set in calculator.
        If atomic_projections is True, calculate orbital character of
        each band at each k-point.

        Returns an array of energies.
        (if spin-polarized spin is first index;
        the next index enumerates the k-points)
        """
        efermi = self.get_fermi_level()

        # run a nscf calculation
        if not hasattr(self, 'natoms'):
            self.atoms2species()
            self.natoms = len(self.atoms)
        oldnoinv = self.noinv
        oldnosym = self.nosym
        self.noinv = True
        self.nosym = True
        self.writeinputfile(
            filename='pwnscf.inp',
            mode='nscf',
            overridekpts=kptpath,
            overridenbands=nbands,
            suppressforcecalc=True)
        self.noinv = oldnoinv
        self.nosym = oldnosym
        self.run_espressox(self.exedir + 'pw.x', 'pwnscf.inp', 'pwnscf.log')

        energies = self.get_eigenvalues(efermi=efermi)

        if not atomic_projections:
            return energies
        else:
            # run pdos calculation with (tiny) E-range
            # to trigger calculation of atomic_proj.xml

            # create input for projwfc.x
            f = open(self.localtmp + '/pdos.inp', 'w')
            print('&PROJWFC\n  prefix=\'calc\',\n  outdir=\'.\',', file=f)
            print('  filpdos = \'projtmp\',', file=f)
            print('  Emin = ' + utils.num2str(-0.3 + efermi) + ',', file=f)
            print('  Emax = ' + utils.num2str(-0.2 + efermi) + ',', file=f)
            print('  DeltaE = 0.1d0,', file=f)
            print('/', file=f)
            f.close()
            # run projwfc.x
            self.run_espressox('projwfc.x', 'pdos.inp', 'pdos.log')
            # remove unneeded pdos files containing only a tiny E-range of two
            # points
            call('rm -f ' + self.scratch + '/projtmp*', shell=True)

            return energies, self.__get_atomic_projections__()

    def __get_atomic_projections__(self):
        f = open(self.scratch + '/calc.save/atomic_proj.xml', 'r')
        cmd = 'grep -a -n Giannozzi ' + self.localtmp + '/pdos.log|tail -1'
        p = Popen(cmd, shell=True, stdout=PIPE).stdout
        n = p.readline().decode('utf-8').split()[0].strip(':').strip()

        cmd = 'tail -n +' + n + ' ' + self.localtmp + '/pdos.log|grep "state #"'
        p = Popen(cmd, shell=True, stdout=PIPE).stdout
        # identify states from projwfc.x's stdout
        states = []
        for x in p.readlines():
            y = x.split('atom')[1]
            iatom = int(y.split()[0]) - 1
            z = y.replace(')\n', '').split('=')
            if y.find('m_j') < 0:
                l = int(z[1].replace('m', ''))
                m = int(z[2])
                states.append([iatom, l, m])
            else:
                j = float(z[1].replace('l', ''))
                l = int(z[2].replace('m_j', ''))
                mj = float(z[3])
                states.append([iatom, j, l, mj])

        # read in projections from atomic_proj.xml
        a = f.readline().decode('utf-8')
        while a.find('<NUMBER_OF_B') < 0:
            a = f.readline().decode('utf-8')
        nbnd = int(f.readline().decode('utf-8').strip())
        a = f.readline().decode('utf-8')
        while a.find('<NUMBER_OF_K') < 0:
            a = f.readline().decode('utf-8')
        nkp = int(f.readline().decode('utf-8').strip())
        a = f.readline().decode('utf-8')
        while a.find('<NUMBER_OF_S') < 0:
            a = f.readline().decode('utf-8')
        spinpol = int(f.readline().decode('utf-8').strip()) == 2

        if spinpol:
            proj1 = []
            proj2 = []
            proj = proj1
        else:
            proj = []

        while a.find('<ATM') < 0 and a != '':
            a = f.readline().decode('utf-8')
        if a == '':
            raise RuntimeError('no projections found')

        while True:
            while a.find('<ATM') < 0 and a != '':
                if spinpol and a.find('<SP') >= 0:
                    if a.find('N.1') > 0:
                        proj = proj1
                    else:
                        proj = proj2
                a = f.readline().decode('utf-8')
            if a == '':
                break
            pr = np.empty(nbnd, np.complex)
            for i in range(nbnd):
                b = f.readline().decode('utf-8').split(',')
                pr[i] = float(b[0]) + 1j * float(b[1])
            proj.append(pr)
            a = f.readline().decode('utf-8')

        f.close()

        if spinpol:
            projections = np.array([proj1, proj2])
            return states, np.reshape(projections,
                                      (2, nkp, len(proj1) / nkp, nbnd))
        else:
            projections = np.array(proj)
            return states, np.reshape(projections,
                                      (nkp, len(proj) / nkp, nbnd))

    def get_eigenvalues(self, kpt=None, spin=None, efermi=None):
        self.stop()

        if self.spinpol:
            cmd = "grep -a eigenval1.xml " + self.scratch + "/calc.save/data-file.xml|tr '\"' ' '|awk '{print $(NF-1)}'"
            p = Popen(cmd, shell=True, stdout=PIPE).stdout
            kptdirs1 = [x.strip() for x in p.readlines()]

            kptdirs1.sort()
            cmd = "grep -a eigenval2.xml " + self.scratch + "/calc.save/data-file.xml|tr '\"' ' '|awk '{print $(NF-1)}'"
            p = Popen(cmd, shell=True, stdout=PIPE).stdout
            kptdirs2 = [x.strip() for x in p.readlines()]

            kptdirs2.sort()
            kptdirs = kptdirs1 + kptdirs2
        else:
            cmd = "grep -a eigenval.xml " + self.scratch + "/calc.save/data-file.xml|tr '\"' ' '|awk '{print $(NF-1)}'"
            p = Popen(cmd, shell=True, stdout=PIPE).stdout
            kptdirs = [x.strip() for x in p.readlines()]
            kptdirs.sort()

        nkp2 = len(kptdirs) / 2
        if kpt is None:  # get eigenvalues at all k-points
            if self.spinpol:
                if spin == 'up' or spin == 0:
                    kp = kptdirs[:nkp2]
                if spin == 'down' or spin == 1:
                    kp = kptdirs[nkp2:]
                else:
                    kp = kptdirs
            else:
                kp = kptdirs
        else:  # get eigenvalues at specific k-point
            if self.spinpol:
                if spin == 'up' or spin == 0:
                    kp = [kptdirs[kpt]]
                if spin == 'down' or spin == 1:
                    kp = [kptdirs[kpt + nkp2]]
                else:
                    kp = [kptdirs[kpt], kptdirs[kpt + nkp2 * 2]]
            else:
                kp = [kptdirs[kpt]]

        if efermi is None:
            ef = 0.
        else:
            ef = efermi

        eig = []
        for k in kp:
            f = open(self.scratch + '/calc.save/' + k, 'r')
            a = f.readline().decode('utf-8')
            while a.upper().find('<EIG') < 0:
                a = f.readline().decode('utf-8')
            nbnd = int(a.split('"')[-2])
            eig.append(Hartree * np.fromfile(
                f, dtype=float, count=nbnd, sep=' ') - ef)
            f.close()

        spinall = spin not in ('up', 'down', 0, 1)
        if kpt is not None and spinall:
            return np.array(eig[0])
        elif kpt is None and spinall and self.spinpol:
            return np.reshape(np.array(eig), (2, nkp2, nbnd))
        else:
            return np.array(eig)

    def read_3d_grid(self, stream, log):
        with open(self.localtmp + '/' + log, 'a') as f:
            x = stream.readline().decode('utf-8')
            while x != '' and x[:11] != 'DATAGRID_3D':
                f.write(x)
                x = stream.readline().decode('utf-8')
            if x == '':
                raise RuntimeError('error reading 3D data grid')
            f.write(x)

            n = [int(_) for _ in stream.readline().split()]
            origin = np.array([float(_) for _ in stream.readline().split()])

            cell = np.empty((3, 3))
            for i in range(3):
                cell[i][:] = [float(_) for _ in stream.readline().split()]

            data = []
            ntotal = np.prod(n)
            while len(data) < ntotal:
                data += [float(_) for _ in stream.readline().split()]
            data = np.reshape(data, n, order='F')

            x = stream.readline().decode('utf-8')
            while x != '':
                f.write(x)
                x = stream.readline().decode('utf-8')

        return (origin, cell, data)

    def get_charge_density(self, spin='both', log=None):
        """Obtains the charge density as a numpy array after a DFT calculation.
        """
        if spin == 'both' or spin == 0:
            s = 0
        elif spin == 'up' or spin == 1:
            s = 1
        elif spin == 'down' or spin == 2:
            s = 2
        else:
            raise ValueError('unknown spin component')

        args = {'inp': 'charge.inp',
                'inputpp': [['plot_num', 0], ['spin_component', s]],
                'piperead': True,
                'parallel': False,
                'log': log}

        if log is not None:
            self.run_ppx(**args)
            return
        else:
            p = self.run_ppx(**args)
            origin, cell, data = self.read_3d_grid(p, 'charge.log')
            p.close()

        return (origin, cell, data)

    def get_total_potential(self, spin='both', log=None):
        """Obtains the total potential as a numpy array after a DFT calculation.
        """
        if spin == 'both' or spin == 0:
            s = 0
        elif spin == 'up' or spin == 1:
            s = 1
        elif spin == 'down' or spin == 2:
            s = 2
        else:
            raise ValueError('unknown spin component')

        args = {'inp': 'totalpot.inp',
                'inputpp': [['plot_num', 1], ['spin_component', s]],
                'piperead': True,
                'parallel': False,
                'log': log}

        if log is not None:
            self.run_ppx(**args)
            return
        else:
            p = self.run_ppx(**args)
            origin, cell, data = self.read_3d_grid(p, 'totalpot.log')
            p.close()

        return (origin, cell, data * Rydberg)

    def get_local_ionic_potential(self, log=None):
        """Obtains the local ionic potential as a numpy array after a DFT calculation.
        """
        args = {'inp': 'vbare.inp',
                'inputpp': [['plot_num', 2]],
                'piperead': True,
                'parallel': False,
                'log': log}

        if log is not None:
            self.run_ppx(**args)
            return
        else:
            p = self.run_ppx(**args)
            origin, cell, data = self.read_3d_grid(p, 'vbare.log')
            p.close()

        return (origin, cell, data * Rydberg)

    def get_local_dos_at_efermi(self):
        """Obtains the local DOS at the Fermi level as a numpy array
        after a DFT calculation.
        """
        args = {'inp': 'ldosef.inp',
                'inputpp': [['plot_num', 3]],
                'piperead': True,
                'parallel': False,
                'log': log}

        p = self.run_ppx(
            'ldosef.inp',
            inputpp=[['plot_num', 3]],
            piperead=True,
            parallel=False)
        origin, cell, data = self.read_3d_grid(p, 'ldosef.log')
        p.close()
        return (origin, cell, data)

    def xsf_local_dos_at_efermi(self, xsf):
        """
        Writes the local DOS at the Fermi level from a DFT calculation
        to an input file for xcrysden.
        """
        self.run_ppx(
            'ldosef.inp',
            inputpp=[['plot_num', 3]],
            plot=[['fileout', self.topath(xsf)]],
            parallel=False,
            log='ldosef.log')

    def extract_local_entropy_density(self):
        """Obtains the local entropy density as a numpy array after a
        DFT calculation. Returns (origin,cell,density).
        """
        p = self.run_ppx(
            'lentr.inp',
            inputpp=[['plot_num', 4]],
            piperead=True,
            parallel=False)
        origin, cell, data = self.read_3d_grid(p, 'lentr.log')
        p.close()
        return (origin, cell, data)

    def xsf_local_entropy_density(self, xsf):
        """
        Writes the local entropy density from a DFT calculation
        to an input file for xcrysden.
        """
        self.run_ppx(
            'lentr.inp',
            inputpp=[['plot_num', 4]],
            plot=[['fileout', self.topath(xsf)]],
            parallel=False,
            log='lentr.log')

    def extract_stm_data(self, bias):
        """
        Obtains STM data as a numpy array after a DFT calculation.
        Returns (origin,cell,stmdata).
        """
        p = self.run_ppx(
            'stm.inp',
            inputpp=[['plot_num', 5], ['sample_bias', bias / Rydberg]],
            piperead=True,
            parallel=False)
        origin, cell, data = self.read_3d_grid(p, 'stm.log')
        p.close()
        return (origin, cell, data)

    def xsf_stm_data(self, xsf, bias):
        """
        Writes STM data from a DFT calculation
        to an input file for xcrysden.
        """
        self.run_ppx(
            'stm.inp',
            inputpp=[['plot_num', 5], ['sample_bias', bias / Rydberg]],
            plot=[['fileout', self.topath(xsf)]],
            parallel=False,
            log='stm.log')

    def extract_magnetization_density(self):
        """Obtains the magnetization density as a numpy array after a
        DFT calculation. Returns (origin,cell,density).
        """
        p = self.run_ppx(
            'magdens.inp',
            inputpp=[['plot_num', 6]],
            piperead=True,
            parallel=False)
        origin, cell, data = self.read_3d_grid(p, 'magdens.log')
        p.close()
        return (origin, cell, data)

    def xsf_magnetization_density(self, xsf):
        """
        Writes the magnetization density from a DFT calculation
        to an input file for xcrysden.
        """
        self.run_ppx(
            'magdens.inp',
            inputpp=[['plot_num', 6]],
            plot=[['fileout', self.topath(xsf)]],
            parallel=False,
            log='magdens.log')

    def extract_wavefunction_density(self,
                                     band,
                                     kpoint=0,
                                     spin='up',
                                     gamma_with_sign=False):
        """Obtains the amplitude of a given wave function as a numpy array after
        a DFT calculation. Returns (origin,cell,amplitude).
        """
        if spin == 'up' or spin == 1:
            s = 0
        elif spin == 'down' or spin == 2:
            s = 1
        elif spin == 'charge' or spin == 0:
            s = 0
        elif spin == 'x':
            s = 1
        elif spin == 'y':
            s = 2
        elif spin == 'z':
            s = 3
        else:
            raise ValueError('unknown spin component')
        if self.spinpol:
            cmd = 'grep -a "number of k points=" ' + self.log + '|tail -1|tr \'=\' \' \''
            p = Popen(cmd, shell=True, stdout=PIPE).stdout
            nkp = int(p.readline().decode('utf-8').split()[4])
            kp = kpoint + nkp / 2 * s
        else:
            kp = kpoint
        inputpp = [['plot_num', 7], ['kpoint', kp], ['kband', band]]
        if gamma_with_sign:
            inputpp.append(['lsign', '.true.'])
        if self.noncollinear:
            inputpp.append(['spin_component', s])
        p = self.run_ppx(
            'wfdens.inp', inputpp=inputpp, piperead=True, parallel=True)
        origin, cell, data = self.read_3d_grid(p, 'wfdens.log')
        p.close()
        return (origin, cell, data)

    def xsf_wavefunction_density(self,
                                 xsf,
                                 band,
                                 kpoint=0,
                                 spin='up',
                                 gamma_with_sign=False):
        """
        Writes the amplitude of a given wave function from a DFT calculation
        to an input file for xcrysden.
        """
        if spin == 'up' or spin == 1:
            s = 0
        elif spin == 'down' or spin == 2:
            s = 1
        elif spin == 'charge' or spin == 0:
            s = 0
        elif spin == 'x':
            s = 1
        elif spin == 'y':
            s = 2
        elif spin == 'z':
            s = 3
        else:
            raise ValueError('unknown spin component')
        if self.spinpol:
            cmd = 'grep -a "number of k points=" ' + self.log + '|tail -1|tr \'=\' \' \''
            p = Popen(cmd, shell=True, stdout=PIPE).stdout
            nkp = int(p.readline().decode('utf-8').split()[4])
            kp = kpoint + nkp / 2 * s
        else:
            kp = kpoint
        inputpp = [['plot_num', 7], ['kpoint', kp], ['kband', band]]
        if gamma_with_sign:
            inputpp.append(['lsign', '.true.'])
        if self.noncollinear:
            inputpp.append(['spin_component', s])
        self.run_ppx(
            'wfdens.inp',
            inputpp=inputpp,
            plot=[['fileout', self.topath(xsf)]],
            parallel=True,
            log='wfdens.log')

    def extract_electron_localization_function(self):
        """
        Obtains the ELF as a numpy array after a DFT calculation.
        Returns (origin,cell,elf).
        """
        p = self.run_ppx(
            'elf.inp',
            inputpp=[['plot_num', 8]],
            piperead=True,
            parallel=False)
        origin, cell, data = self.read_3d_grid(p, 'elf.log')
        p.close()
        return (origin, cell, data)

    def xsf_electron_localization_function(self, xsf):
        """
        Writes the ELF from a DFT calculation
        to an input file for xcrysden.
        """
        self.run_ppx(
            'elf.inp',
            inputpp=[['plot_num', 8]],
            plot=[['fileout', self.topath(xsf)]],
            parallel=False,
            log='elf.log')

    def extract_density_minus_atomic(self):
        """Obtains the charge density minus atomic charges as a numpy array
        after a DFT calculation. Returns (origin,cell,density).
        """
        p = self.run_ppx(
            'dens_wo_atm.inp',
            inputpp=[['plot_num', 9]],
            piperead=True,
            parallel=False)
        origin, cell, data = self.read_3d_grid(p, 'dens_wo_atm.log')
        p.close()
        return (origin, cell, data)

    def xsf_density_minus_atomic(self, xsf):
        """
        Writes the charge density minus atomic charges from a DFT calculation
        to an input file for xcrysden.
        """
        self.run_ppx(
            'dens_wo_atm.inp',
            inputpp=[['plot_num', 9]],
            plot=[['fileout', self.topath(xsf)]],
            parallel=False,
            log='dens_wo_atm.log')

    def extract_int_local_dos(self, spin='both', emin=None, emax=None):
        """
        Obtains the integrated ldos as a numpy array after a DFT calculation.
        Returns (origin,cell,ldos).
        """
        if spin == 'both' or spin == 0:
            s = 0
        elif spin == 'up' or spin == 1:
            s = 1
        elif spin == 'down' or spin == 2:
            s = 2
        else:
            raise ValueError('unknown spin component')

        inputpp = [['plot_num', 10], ['spin_component', s]]
        efermi = self.get_fermi_level()
        if emin is not None:
            inputpp.append(['emin', emin - efermi])
        if emax is not None:
            inputpp.append(['emax', emax - efermi])

        p = self.run_ppx(
            'ildos.inp', inputpp=inputpp, piperead=True, parallel=False)
        origin, cell, data = self.read_3d_grid(p, 'ildos.log')
        p.close()
        return (origin, cell, data)

    def xsf_int_local_dos(self, xsf, spin='both', emin=None, emax=None):
        """
        Writes the integrated ldos from a DFT calculation
        to an input file for xcrysden.
        """
        if spin == 'both' or spin == 0:
            s = 0
        elif spin == 'up' or spin == 1:
            s = 1
        elif spin == 'down' or spin == 2:
            s = 2
        else:
            raise ValueError('unknown spin component')

        inputpp = [['plot_num', 10], ['spin_component', s]]
        efermi = self.get_fermi_level()
        if emin is not None:
            inputpp.append(['emin', emin - efermi])
        if emax is not None:
            inputpp.append(['emax', emax - efermi])

        self.run_ppx(
            'ildos.inp',
            inputpp=inputpp,
            plot=[['fileout', self.topath(xsf)]],
            parallel=False,
            log='ildos.log')

    def extract_ionic_and_hartree_potential(self):
        """Obtains the sum of ionic and Hartree potential as a numpy array
        after a DFT calculation. Returns (origin,cell,potential).
        """
        p = self.run_ppx(
            'potih.inp',
            inputpp=[['plot_num', 11]],
            piperead=True,
            parallel=False)
        origin, cell, data = self.read_3d_grid(p, 'potih.log')
        p.close()
        return (origin, cell, data * Rydberg)

    def xsf_ionic_and_hartree_potential(self, xsf):
        """
        Writes the sum of ionic and Hartree potential from a DFT calculation
        to an input file for xcrysden.
        """
        self.run_ppx(
            'potih.inp',
            inputpp=[['plot_num', 11]],
            plot=[['fileout', self.topath(xsf)]],
            parallel=False,
            log='potih.log')

    def extract_sawtooth_potential(self):
        """Obtains the saw tooth potential as a numpy array
        after a DFT calculation. Returns (origin,cell,potential).
        """
        p = self.run_ppx(
            'sawtooth.inp',
            inputpp=[['plot_num', 12]],
            piperead=True,
            parallel=False)
        origin, cell, data = self.read_3d_grid(p, 'sawtooth.log')
        p.close()
        return (origin, cell, data * Rydberg)

    def xsf_sawtooth_potential(self, xsf):
        """
        Writes the saw tooth potential from a DFT calculation
        to an input file for xcrysden.
        """
        self.run_ppx(
            'sawtooth.inp',
            inputpp=[['plot_num', 12]],
            plot=[['fileout', self.topath(xsf)]],
            parallel=False,
            log='sawtooth.log')

    def extract_noncollinear_magnetization(self, spin='all'):
        """Obtains the non-collinear magnetization as a numpy array
        after a DFT calculation. Returns (origin,cell,magnetization).
        """
        if spin == 'all' or spin == 'charge' or spin == 0:
            s = 0
        elif spin == 'x':
            s = 1
        elif spin == 'y':
            s = 2
        elif spin == 'z':
            s = 3
        else:
            raise ValueError('unknown spin component')
        p = self.run_ppx(
            'noncollmag.inp',
            inputpp=[['plot_num', 13], ['spin_component', s]],
            piperead=True,
            parallel=False)
        origin, cell, data = self.read_3d_grid(p, 'noncollmag.log')
        p.close()
        return (origin, cell, data)

    def xsf_noncollinear_magnetization(self, xsf, spin='all'):
        """
        Writes the non-collinear magnetization as from a DFT calculation
        to an input file for xcrysden.
        """
        if spin == 'all' or spin == 'charge' or spin == 0:
            s = 0
        elif spin == 'x':
            s = 1
        elif spin == 'y':
            s = 2
        elif spin == 'z':
            s = 3
        else:
            raise ValueError('unknown spin component')
        self.run_ppx(
            'noncollmag.inp',
            inputpp=[['plot_num', 13], ['spin_component', s]],
            plot=[['fileout', self.topath(xsf)]],
            parallel=False)

    def extract_ae_charge_density(self, spin='both'):
        """Obtains the all-electron (PAW) charge density as a numpy array
        after a DFT calculation. Returns (origin,cell,density)
        """
        if spin == 'both' or spin == 0:
            s = 0
        elif spin == 'up' or spin == 1:
            s = 1
        elif spin == 'down' or spin == 2:
            s = 2
        else:
            raise ValueError('unknown spin component')

        p = self.run_ppx(
            'aecharge.inp',
            inputpp=[['plot_num', 17], ['spin_component', s]],
            piperead=True,
            parallel=False)
        origin, cell, data = self.read_3d_grid(p, 'aecharge.log')
        p.close()
        return (origin, cell, data)

    def xsf_ae_charge_density(self, xsf, spin='both'):
        """
        Writes the all-electron (PAW) charge density from a DFT calculation
        to an input file for xcrysden.
        """
        if spin == 'both' or spin == 0:
            s = 0
        elif spin == 'up' or spin == 1:
            s = 1
        elif spin == 'down' or spin == 2:
            s = 2
        else:
            raise ValueError('unknown spin component')

        self.run_ppx(
            'aecharge.inp',
            inputpp=[['plot_num', 17], ['spin_component', s]],
            plot=[['fileout', self.topath(xsf)]],
            parallel=False,
            log='aecharge.log')

    def extract_noncollinear_xcmag(self):
        """Obtains the xc magnetic field for a non-collinear system as a numpy array
        after a DFT calculation. Returns (origin,cell,field).
        """
        p = self.run_ppx(
            'ncxcmag.inp',
            inputpp=[['plot_num', 18]],
            piperead=True,
            parallel=False)
        origin, cell, data = self.read_3d_grid(p, 'ncxcmag.log')
        p.close()
        return (origin, cell, data)

    def xsf_noncollinear_xcmag(self, xsf):
        """ Writes the xc magnetic field for a non-collinear system from
        a DFT calculation to an input file for xcrysden.
        """
        self.run_ppx(
            'ncxcmag.inp',
            inputpp=[['plot_num', 18]],
            plot=[['fileout', self.topath(xsf)]],
            parallel=False,
            log='ncxcmag.log')

    def extract_reduced_density_gradient(self):
        """Obtains the reduced density gradient as a numpy array after
        a DFT calculation. Returns (origin,cell,gradient).
        """
        p = self.run_ppx(
            'redgrad.inp',
            inputpp=[['plot_num', 19]],
            piperead=True,
            parallel=False)
        origin, cell, data = self.read_3d_grid(p, 'redgrad.log')
        p.close()
        return (origin, cell, data)

    def xsf_reduced_density_gradient(self, xsf):
        """
        Writes the reduced density gradient from a DFT calculation
        to an input file for xcrysden.
        """
        self.run_ppx(
            'redgrad.inp',
            inputpp=[['plot_num', 19]],
            plot=[['fileout', self.topath(xsf)]],
            parallel=False,
            log='redgrad.log')

    def extract_middle_density_hessian_eig(self):
        """Obtains the middle Hessian eigenvalue as a numpy array after
        a DFT calculation. Returns (origin,cell,density).
        """
        p = self.run_ppx(
            'mideig.inp',
            inputpp=[['plot_num', 20]],
            piperead=True,
            parallel=False)
        origin, cell, data = self.read_3d_grid(p, 'mideig.log')
        p.close()
        return (origin, cell, data)

    def xsf_middle_density_hessian_eig(self, xsf):
        """Writes the middle Hessian eigenvalue from a DFT calculation
        to an input file for xcrysden.
        """
        self.run_ppx(
            'mideig.inp',
            inputpp=[['plot_num', 20]],
            plot=[['fileout', self.topath(xsf)]],
            parallel=False,
            log='mideig.log')

    def find_max_empty_space(self, edir=3):
        """Assuming periodic boundary conditions, finds the largest
        continuous segment of free, unoccupied space and returns
        its midpoint in scaled coordinates (0 to 1) in the edir
        direction (default z).
        """
        position_array = self.atoms.get_scaled_positions()[
            ..., edir - 1]  # 0-indexed direction
        position_array.sort()
        differences = np.diff(position_array)
        differences = np.append(
            differences,
            position_array[0] + 1 - position_array[-1])  # through the PBC
        max_diff_index = np.argmax(differences)
        if max_diff_index == len(position_array) - 1:
            # should be < 1 in cell units
            return (position_array[0] + 1 + position_array[-1]) / 2. % 1
        else:
            return (position_array[max_diff_index] +
                    position_array[max_diff_index + 1]) / 2.

    def get_work_function(self, pot_filename="pot.xsf", edir=3):
        """Calculates the work function of a calculation by subtracting
        the electrostatic potential of the vacuum (from averaging the
        output of pp.x num_plot 11 in the z direction by default) from
        the Fermi energy. Values used for average.x come from the espresso
        example for work function for a surface.
        """
        self.update(self.atoms)
        self.stop()
        self.run_ppx(
            'wf_pp.in',
            log='wf_pp.log',
            inputpp=[('plot_num', 11), ('filplot', self.topath('pot.xsf'))],
            output_format=3,
            iflag=3,
            piperead=False,
            parallel=False)

        f = open(self.localtmp + '/avg.in', 'w')
        print('1', file=f)
        print(self.sdir + "/" + pot_filename, file=f)
        print('1.D0', file=f)
        print('1440', file=f)
        print(str(edir), file=f)
        print('3.835000000', file=f)
        print('', file=f)
        f.close()
        call('cp ' + self.localtmp + '/avg.in ' + self.scratch, shell=True)
        call('cd ' + self.scratch + ' ; ' + 'average.x < avg.in >>' +
             self.localtmp + '/avg.out', shell=True)

        # Pick a good place to sample vacuum level
        cell_length = self.atoms.cell[edir - 1][edir - 1] / Bohr
        vacuum_pos = self.find_max_empty_space(edir) * cell_length
        avg_out = open(self.localtmp + '/avg.out', 'r')
        record = False
        average_data = []
        lines = list(avg_out)
        for line in lines:
            if len(line.split()) == 3 and line.split()[0] == "0.000000000":
                record = True
            elif len(line.split()) == 0:
                record = False
            if record:
                average_data.append([float(i) for i in line.split()])
        # [1] is planar average [2] is macroscopic average
        vacuum_energy = average_data[np.abs(
            np.array(average_data)[..., 0] - vacuum_pos).argmin()][1]

        # Get the latest Fermi energy
        cmd = 'grep -a -n "Fermi" ' + self.log + ' | tail -1'
        fermi_data = Popen(cmd, shell=True, stdout=PIPE).stdout
        fermi_energy = float(fermi_data.readline().decode('utf-8').split()[-2])
        fermi_data.close()

        # if there's a dipole, we need to return 2 work functions - one for
        # either direction away from the slab
        if self.dipole['status']:
            eopreg = 0.025
            if 'eopreg' in self.dipole:
                eopreg = self.dipole['eopreg']
            # we use cell_length*eopreg*2.5 here since the work functions seem
            # to converge at that distance rather than *1 or *2
            vac_pos1 = (vacuum_pos - cell_length * eopreg * 2.5) % cell_length
            vac_pos2 = (vacuum_pos + cell_length * eopreg * 2.5) % cell_length
            vac_index1 = np.abs(
                np.array(average_data)[..., 0] - vac_pos1).argmin()
            vac_index2 = np.abs(
                np.array(average_data)[..., 0] - vac_pos2).argmin()
            vacuum_energy1 = average_data[vac_index1][1]
            vacuum_energy2 = average_data[vac_index2][1]
            wf = [
                vacuum_energy1 * Rydberg - fermi_energy,
                vacuum_energy2 * Rydberg - fermi_energy
            ]
        else:
            wf = vacuum_energy * Rydberg - fermi_energy

        return wf
