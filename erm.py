from math import sqrt, cos, pi

import numpy as np

from ase.parallel import world, rank, size
from ase.neb import NEB
from ase.dimer import normalize, DimerEigenmodeSearch, MinModeAtoms, perpendicular_vector
from ase.dimer import norm, parallel_vector, DimerControl

class ERM(NEB):
    def __init__(self, images, control, k=1.0, climb=False, parallel=False, minmodes=None, decouple_modes=False):
        self.control = control
        NEB.__init__(self, images, k, climb, parallel)

        self.spring_force = 'full'
        self.decouple_inidividual_modes = []

        # Set up MinModeAtoms objects for each image and make individual logfiles for each
        # NB: Shouldn't there be a ERM_Control class that takes care of this crap?
        self.images = []
        for i in range(self.nimages):
            min_control = control.copy()

            i_num = ('%0' + str(len(str(self.nimages))) + 'i') % i
            d_logfile_old = self.control.get_logfile()
            m_logfile_old = self.control.get_eigenmode_logfile()
            if d_logfile_old not in ['-', None]:
                if type(d_logfile_old) == str:
                    d_logfile_old = d_logfile_old.split('.')
                else:
                    d_logfile_old = d_logfile_old.name.split('.')
                d_logfile_old.insert(-1, i_num)
                d_logfile_new = '-'.join(['.'.join(d_logfile_old[:-2]), '.'.join(d_logfile_old[-2:])])
            else:
                d_logfile_new = d_logfile_old
            if m_logfile_old not in ['-', None]:
                if type(m_logfile_old) == str:
                    m_logfile_old = m_logfile_old.split('.')
                else:
                    m_logfile_old = m_logfile_old.name.split('.')
                m_logfile_old.insert(-1, i_num)
                m_logfile_new = '-'.join(['.'.join(m_logfile_old[:-2]), '.'.join(m_logfile_old[-2:])])
            else:
                m_logfile_new = m_logfile_old

            if i in [0, self.nimages - 1]:
                write_rank = 0
            else:
                write_rank = (i - 1) * size // (self.nimages - 2)

            min_control.set_write_rank(write_rank)
            min_control.initialize_logfiles(logfile = d_logfile_new, eigenmode_logfile = m_logfile_new)
            if minmodes is None:
                minmode = None
            else:
                minmodes = np.array(minmodes)
                if minmodes.shape == (self.nimages, self.natoms, 3):
                    # Assume one minmode for each image
                    raise NotImplementedError()
                elif minmodes.shape == (2, self.natoms, 3):
                    # Assume end images minmodes and interpolate
                    raise NotImplementedError()
                elif minmodes.shape == (self.natoms, 3):
                    minmode = [minmodes.copy()]
                else:
                    raise ValueError('ERM did not understand the minmodes given to it.')

            image = MinModeAtoms(images[i], min_control, eigenmodes = minmode)
            self.images.append(image)

        self.forces['dimer'] = np.zeros((self.nimages, self.natoms, 3))

        # Populate the tangents
        for i in range(1, self.nimages - 1):
            p_m = self.images[i - 1].get_positions()
            p_p = self.images[i + 1].get_positions()
            t = (p_p - p_m) / 2.0
            if 0.0 in t:
                # Assume a linear interpolation
                # HACK/BUG: Currently the last or first "free" image will yield p[-1] - p[0]
                t = self.images[-1].get_positions() - self.images[0].get_positions()
                t /= (self.nimages - 1.0)
            self.tangents[i] = t
        self.tangents[0] = t
        self.tangents[-1] = -t

        # Save user variables
        self.decouple_modes = decouple_modes # Release the orthogonality constraint of the minmode and tanget.

        # Testing stuff
        self.reduce_containment = False
        self.reduce_containment_tol = 0.010
        self.containment_factor = 1.0
        self.decouple_individual_modes = []
        self.containment_factors = np.ones((self.nimages))

    def calculate_image_energies_and_forces(self, i):
        self.forces['real'][i] = self.images[i].get_forces(real = True)
        self.energies[i] = self.images[i].get_potential_energy()

    def get_forces(self):
        """Evaluate and return the forces."""

        if self.parallel and self.images[1].minmode_init:
            propagate_initial_values = True
        else:
            propagate_initial_values = False

        # Update the clean forces and energies
        self.calculate_energies_and_forces()

        if self.parallel and propagate_initial_values:
            for i in range(1, self.nimages - 1):
                if self.images[i].eigenmodes is None:
                    self.images[i].eigenmodes = [np.zeros(self.images[i].get_positions().shape)]
                self.images[i].minmode_init = False
                self.images[i].rotation_required = True
                self.images[i].check_atoms = self.images[i].atoms.copy()

                root = (i - 1) * size // (self.nimages - 2)
                world.broadcast(self.images[i].eigenmodes[0], root)

        # Update the highest energy image
        self.imax = 1 + np.argsort(self.energies[1:-1])[-1]
        self.emax = self.energies[self.imax]
        if self.imax == self.nimages - 2:
            self.imax -= 1
        elif self.imax == 1:
            self.imax += 1
        # BUG: self.imax can be an endimage. Partially fixed by setting the end images energy to -np.inf (somewhere else)

        # Calculate the tangents of all the images
        self.update_tangents()

        # IDEA: If f_c_perp is small force dimer rotation?

        # Calculate the modes
        self.calculate_eigenmodes()

        # Prjoect the forces for each image
        self.invert_eigenmode_forces()
        self.project_forces(sort = 'dimer')
        if self.reduce_containment:
            self.adjust_containment_forces()
        self.control.increment_counter('optcount')
#        for k in range(1, self.nimages - 1):
#        for k in [4, 5, 6]:
#            print k, np.vdot(normalize(self.tangents[k]), self.images[k].get_eigenmode())
        return self.forces['neb'][1:self.nimages-1].reshape((-1, 3))

    def adjust_containment_forces(self):
#        self.adjust_containment_forces_ALL()
#        self.adjust_containment_forces_NO_PLANE()
        self.adjust_containment_forces_DOT()

    def adjust_containment_forces_DOT(self):
        for i in range(1, self.nimages - 1):
            if self.climb and i == self.imax:
                pass
            else:
                nt = normalize(self.tangents[i])
                nm = self.images[i].get_eigenmode()
                f_s = self.forces['spring'][i]
                f_d = self.forces['dimer'][i]
                f_s_para = np.vdot(f_s, nt) * nt
                f_s_perp = f_s - f_s_para
                f_d_para = np.vdot(f_d, nt) * nt
                f_d_perp = f_d - f_d_para
                f_s_perp_red = f_s_perp * self.containment_factors[i]
                nd = normalize(f_d_perp)
                ns = normalize(f_s_perp_red)
                dot = np.vdot(nd, ns)
                ratio = norm(f_s_perp_red) / norm(f_d_perp)
                norm_force = (((f_d_perp + f_s_para + f_s_perp_red)**2).sum(axis=1).max())**0.5
                print '%02i  % 4.3f  % 4.3f  % 4.3f' % (i, dot, ratio, self.containment_factors[i]),
                print '% 4.3f' % norm_force,
                if dot < -0.98 and dot > -1.02 and ratio > 0.98 and ratio < 1.02 and norm_force < self.reduce_containment_tol:
                    if i == 1:
                        cfp = self.containment_factors[i] / self.containment_factors[i+1]
                        if cfp > 0.40:
                            self.containment_factors[i] *= 0.70
                            print 'change'
                        else:
                            print 'no change'
                    elif i == self.nimages - 2:
                        cfm = self.containment_factors[i] / self.containment_factors[i-1]
                        if cfm > 0.40:
                            self.containment_factors[i] *= 0.70
                            print 'change'
                        else:
                            print 'no change'
                    else:
                        cfp = self.containment_factors[i] / self.containment_factors[i+1]
                        cfm = self.containment_factors[i] / self.containment_factors[i-1]
                        if cfp > 0.40 and cfm > 0.40:
                            self.containment_factors[i] *= 0.70
                            print 'change'
                        else:
                            print 'no change'
                else:
                    print ''
                f_s_new = f_s_para + f_s_perp * self.containment_factors[i]
                self.forces['spring'][i] = f_s_new
                self.forces['neb'][i] = f_d_perp + f_s_new
                self.containment_factors[self.imax] = min(self.containment_factors)
#        norm_force = (((self.forces['neb'][1 : self.nimages - 1].reshape((-1, 3)))**2).sum(axis=1).max())**0.5
#        if norm_force < self.reduce_containment_tol:
#            self.containment_factor *= (1.0 - 10.0 * (self.reduce_containment_tol - norm_force))
#            if self.containment_factor < 0.0:
#                self.containment_factor = 0.0
#        print self.containment_factor, norm_force, (1.0 - 10.0 * (self.reduce_containment_tol - norm_force))

    def adjust_containment_forces_NO_PLANE(self):
        for i in range(1, self.nimages - 1):
            if self.climb and i == self.imax:
                pass
            else:
                nt = normalize(self.tangents[i])
                nm = self.images[i].get_eigenmode()
                f_s = self.forces['spring'][i]
                f_d = self.forces['dimer'][i]
                f_s_para = np.vdot(f_s, nt) * nt
                f_s_perp = f_s - f_s_para
                f_s_perp_in_plane = np.vdot(f_s_perp, nt) * nt + np.vdot(f_s_perp, nm) * nm
                f_s_perp_out_plane = f_s_perp - f_s_perp_in_plane
                f_s_new = f_s_para + f_s_perp_in_plane + f_s_perp_out_plane * self.containment_factor
                f_d_para = np.vdot(f_d, nt) * nt
                f_d_perp = f_d - f_d_para
                self.forces['spring'][i] = f_s_new
                self.forces['neb'][i] = f_d_perp + f_s_new
        norm_force = (((self.forces['neb'][1 : self.nimages - 1].reshape((-1, 3)))**2).sum(axis=1).max())**0.5
        if norm_force < self.reduce_containment_tol:
            self.containment_factor *= (1.0 - 10.0 * (self.reduce_containment_tol - norm_force))
            if self.containment_factor < 0.0:
                self.containment_factor = 0.0
        print self.containment_factor, norm_force, (1.0 - 10.0 * (self.reduce_containment_tol - norm_force))


    def adjust_containment_forces_ALL(self):
        for i in range(1, self.nimages - 1):
            if self.climb and i == self.imax:
                pass
            else:
                nt = normalize(self.tangents[i])
                f_s = self.forces['spring'][i]
                f_d = self.forces['dimer'][i]
                f_s_para = np.vdot(f_s, nt) * nt
                f_s_perp = f_s - f_s_para
                f_d_para = np.vdot(f_d, nt) * nt
                f_d_perp = f_d - f_d_para
                f_s_new = f_s_para + f_s_perp * self.containment_factor
                self.forces['spring'][i] = f_s_new
                self.forces['neb'][i] = f_d_perp + f_s_new
#                print i, np.vdot(f_r_perp, f_s_perp), np.vdot(normalize(f_r_perp), normalize(f_s_perp))
        norm_force = (((self.forces['neb'][1 : self.nimages - 1].reshape((-1, 3)))**2).sum(axis=1).max())**0.5
        if norm_force < self.reduce_containment_tol:
            self.containment_factor *= (1.0 - 10.0 * (self.reduce_containment_tol - norm_force))
            if self.containment_factor < 0.0:
                self.containment_factor = 0.0
#        else:
#            self.containment_factor *= (1.0 + 10.0 * (norm_force - self.reduce_containment_tol))
        print self.containment_factor, norm_force, (1.0 - 10.0 * (self.reduce_containment_tol - norm_force))

    def project_forces(self, sort='dimer'):
        for i in range(1, self.nimages - 1):
            t = self.tangents[i]
            nt = t / np.vdot(t, t)**0.5
            f_r = self.forces[sort][i]
            f_r_para = np.vdot(f_r, nt) * nt
            f_r_perp = f_r - f_r_para
            if self.climb and i == self.imax:
                self.forces['neb'][i] = f_r - 2 * f_r_para
#                print 'NORMS', norm(self.forces['real'][i]), norm(self.forces['dimer'][i]), norm(self.forces['neb'][i])
            else:
                f_s = self.get_image_spring_force(i)
                self.forces['spring'][i] = f_s
                self.forces['neb'][i] = f_r_perp + f_s

    def calculate_image_eigenmode(self, i):
        img = self.images[i]
        if self.decouple_modes or i in self.decouple_individual_modes:
            img.set_basis(None)
        else:
            nm = normalize(img.get_eigenmode())
#            if (self.reduce_containment or self.spring_force == 'norm') and not (self.climb and i == self.imax):
            if True:
#            if False:
                pm = self.images[i-1].get_positions()
                pp = self.images[i+1].get_positions()
                nt = normalize(pp - pm)
            else:
                nt = normalize(self.tangents[i])
            img.set_basis(nt)
            img.set_eigenmode(normalize(perpendicular_vector(nm, nt)))
        img.get_forces()

    def calculate_eigenmodes(self):
        if self.parallel:
            i = rank * (self.nimages - 2) // size + 1
            try:
                self.calculate_image_eigenmode(i)
            except:
                # Make sure other images also fail:
                error = world.sum(1.0)
                raise
            else:
                error = world.sum(0.0)
                if error:
                    raise RuntimeError('Parallel ERM failed during eigenmode calculations.')
            for i in range(1, self.nimages - 1):
                if self.images[i].eigenmodes is None:
                    self.images[i].eigenmodes = [np.zeros(self.images[i].get_positions().shape)]
                root = (i - 1) * size // (self.nimages - 2)
                world.broadcast(self.images[i].eigenmodes[0], root)
#                world.broadcast(self.images[i : i + 1].curvatures, root)
        else:
            for i in range(1, self.nimages - 1):
                self.calculate_image_eigenmode(i)

    def invert_eigenmode_forces(self):
        for i in range(1, self.nimages - 1):
            f_r = self.forces['real'][i]
            t = self.tangents[i]
            nt = normalize(t)
            nm = self.images[i].get_eigenmode()
            self.forces['dimer'][i] = f_r - 2 * np.vdot(f_r, nm) * nm
