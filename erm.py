from math import sqrt, cos, pi

import numpy as np

from ase.parallel import world, rank, size
from ase.neb import NEB
from ase.dimer import normalize, DimerEigenmodeSearch, MinModeAtoms, perpendicular_vector
from ase.dimer import norm, parallel_vector, DimerControl

class ERM(NEB):
    def __init__(self, images, control, k=1.0, climb=False, parallel=False, minmodes=None):
        self.control = control
        NEB.__init__(self, images, k, climb, parallel)

        self.spring_force = 'full'

        # Set up MinModeAtoms objects for each image and make individual logfiles for each
        # NB: Shouldn't there be a ERM_Control class that takes care of this crap?
        self.images = []
        for i in range(self.nimages):
            min_control = control.copy()
            i_num = ('%0' + str(len(str(self.nimages))) + 'i') % i
            logfile_old = self.control.get_logfile().name.split('.')
            logfile_old.insert(-1, 'erm-%s' % (('%0' + str(len(str(self.nimages))) + 'i') % i))
            logfile_new = '-'.join(['.'.join(logfile_old[:-2]), '.'.join(logfile_old[-2:])])
            min_control.initialize_logfiles(logfile = logfile_new)
            image = MinModeAtoms(images[i], min_control)
            self.images.append(image)

        self.forces['dimer'] = np.zeros((self.nimages, self.natoms, 3))

        # Populate the tangents
        for i in range(1, self.nimages - 1):
            p_m = self.images[i - 1].get_positions()
            p_p = self.images[i + 1].get_positions()
            t = (p_p - p_m) / 2.0
            self.tangents[i] = t
        self.tangents[0] = t
        self.tangents[-1] = -t

        # Set up the initial minimum modes
        if minmodes is None:
            for i in range(self.nimages):
                m = self.images[i]
                m.initialize_eigenmodes()
        else:
            minmodes = np.array(minmodes)
            if minmodes.shape == (self.nimages, self.natoms, 3):
                # Assume one minmode for each image
                raise NotImplementedError()
            elif minmodes.shape == (2, self.natoms, 3):
                # Assume end images minmodes and interpolate
                raise NotImplementedError()
            elif minmodes.shape == (self.natoms, 3):
                # Assume the same minmode for all images
                raise NotImplementedError()
            else:
                raise ValueError('ERM did not understand the minmodes given to it.')

        # Ensure orthogonality of the minmodes
        for i in range(self.nimages):
            t = self.tangents[i]
            nt = normalize(t)
            m = self.images[i].get_eigenmode()
            m -= np.vdot(m, nt) * nt
            m = normalize(m)
            self.images[i].set_eigenmode(m)

        # These should be user variables
        self.decouple_modes = False # Release the orthogonality constraint of the minmode and tanget.

        # Development stuff
        self.plot_devplot = True
        self.plot_subplot = False
        self.plot_animate = 0
        self.plot_x = None
        self.plot_y = None
        self.plot_e = None
        self.xrange = None
        self.yrange = None

    def calculate_image_energies_and_forces(self, i):
        self.energies[i] = self.images[i].get_potential_energy()
        self.forces['real'][i] = self.images[i].get_forces(real = True)

    def get_forces(self):
        """Evaluate and return the forces."""

        # Update the clean forces and energies
        self.calculate_energies_and_forces()

        # Update the highest energy image
        self.imax = 1 + np.argsort(self.energies[1:-1])[-1]
        self.emax = self.energies[self.imax]
        # BUG: self.imax can be an endimage.

        # Calculate the tangents of all the images
        self.update_tangents()

        # IDEA: If f_c_perp is small force dimer rotation?

        # Calculate the modes
        self.calculate_eigenmodes()

        # Prjoect the forces for each image
        self.invert_eigenmode_forces()
        self.project_forces(sort = 'dimer')
        if self.plot_devplot:
            self.plot_pseudo_3d_pes()
        self.control.increment_counter('optcount')
#        print self.images[1]._calc.get_count()
        return self.forces['neb'][1:self.nimages-1].reshape((-1, 3))

    def calculate_eigenmodes(self):
        if self.parallel:
            raise NotImplementedError()
        else:
            for i in range(1, self.nimages - 1):
                img = self.images[i]
                m = img.get_eigenmode()
                t = self.tangents[i]
                nt = normalize(t)
                nm = normalize(m)
                # Does a modified mt need to be passed?
                # Otherwise a bunch of these lines can be deleted.
                if self.decouple_modes:
                    img.set_basis(None)
                    img.find_eigenmodes()
                else:
                    img.set_basis(nt)
                    img.set_eigenmode(normalize(perpendicular_vector(nm, nt)))
                    img.find_eigenmodes()

    def invert_eigenmode_forces(self):
        for i in range(1, self.nimages - 1):
            f_r = self.forces['real'][i]
            t = self.tangents[i]
            nt = normalize(t)
            nm = self.images[i].get_eigenmode()
            self.forces['dimer'][i] = f_r - 2 * np.vdot(f_r, nm) * nm

# ----------------------------------------------------------------
# --------------- Outdated and development methods ---------------
# ----------------------------------------------------------------
    def plot_pseudo_3d_pes(self):
        import pylab as plt
        from pylab import subplot, subplot2grid
        fig = plt.figure(figsize = (8,8))
        if self.plot_subplot:
            plt.axes()
            ax1 = subplot2grid((4, 1), (0, 0), rowspan = 3)
            ax2 = subplot2grid((4, 1), (3, 0), axisbg = 'y')
            ax2_base_scale = 0.00000002

        else:
            plt.axes()
            ax1 = subplot(111)
            ax2 = None

        def make_line(pos, orient, c, size=0.2, width=1, dim=[0, 1], ax=plt):
            p = pos[-1]
            o = orient[-1]
            d1 = dim[0]
            d2 = dim[1]
            ax.plot((p[d1] - o[d1] * size, p[d1] + o[d1] * size), (p[d2] - o[d2] * size, p[d2] + o[d2] * size), c, lw = width)

        def make_arrow(pos, end, c, scale=1.0, width=1, dim=[0,1], ax=plt, head_scale=0.9, base_scale=0.6):
            x = head_scale
            if ax == ax2:
                y = ax2_base_scale
            else:
                y = base_scale
            p = pos[-1]
            e = end[-1]
            d1 = dim[0]
            d2 = dim[1]

            p1 = p[d1]
            p2 = p[d2]

            e1 = e[d1] * scale + p1
            e2 = e[d2] * scale + p2

            a1 = p1 * (1 - x) + x * e1
            a2 = p2 * (1 - x) + x * e2

            b1 = a1 + (a2 - e2)
            b2 = a2 - (a1 - e1)

            c1 = a1 - (a2 - e2)
            c2 = a2 + (a1 - e1)

            b1 = a1 * (1 - y) + y * b1
            b2 = a2 * (1 - y) + y * b2

            c1 = a1 * (1 - y) + y * c1
            c2 = a2 * (1 - y) + y * c2

            ax.plot((p1, e1), (p2, e2), 'k', lw = width + 1)
            ax.plot((b1, e1, c1), (b2, e2, c2), 'k', lw = width + 1)
            ax.plot((p1, e1), (p2, e2), c, lw = width)
            ax.plot((b1, e1, c1), (b2, e2, c2), c, lw = width)

        def make_circle(pos, r, c, dim=[0, 1], ax=plt):
            p = pos[-1]
            d1 = dim[0]
            d2 = dim[1]
            ax.plot((p[d1]), (p[d2]), 'k.', markersize = r + 1)
            ax.plot((p[d1]), (p[d2]), '%s.' % c, markersize = r)

        n = self.nimages
        ts = self.tangents
        ms = []
        ps = []
        for i in range(n):
            ms.append(self.images[i].get_eigenmode())
            ps.append(self.images[i].get_positions())
        f_rs = self.forces['real']
        f_ds = self.forces['dimer']
        f_ns = self.forces['neb']

#        ax1.text(0.6, 0.6, self.phase, color = 'k')
        for i in range(n):
            p = ps[i]
            t = normalize(ts[i]) * 0.25
            m = normalize(ms[i]) * 0.25
            f_r = f_rs[i]
            f_d = f_ds[i]
            f_n = f_ns[i]
            if i in [0, n - 1]:
                make_circle(p, 20.0, 'y', ax = ax1)
            else:
                if self.climb and i == self.imax:
                    make_circle(p, 35.0, 'c', ax = ax1)
                else:
                    make_circle(p, 35.0, 'y', ax = ax1)
                make_arrow(p, f_r, 'w', ax = ax1)
                make_arrow(p, f_d, 'b', ax = ax1)
                make_arrow(p, f_n, 'k', ax = ax1)
                make_line(p, t, 'r', ax = ax1)
                make_line(p, m, 'b', ax = ax1)

        if self.plot_e is not None:
            ax1.contourf(self.plot_x, self.plot_y, self.plot_e, 30)

        if self.plot_animate < 10:
            animate = '000' + str(self.plot_animate)
        elif self.plot_animate < 100:
            animate = '00' + str(self.plot_animate)
        elif self.plot_animate < 1000:
            animate = '0' + str(self.plot_animate)
        else:
            animate = str(self.plot_animate)

        axis1 = ax1.get_axes()
        if self.plot_e is not None:
            axis1.set_xlim(xmin = min(self.plot_x), xmax = max(self.plot_x))
            axis1.set_ylim(ymin = min(self.plot_y), ymax = max(self.plot_y))
        else:
            if self.xrange is not None:
                axis1.set_xlim(xmin = self.xrange[0], xmax = self.xrange[1])
            else:
                axis1.set_xlim(xmin = 0.0, xmax = 5.0)
            if self.yrange is not None:
                axis1.set_ylim(ymin = self.yrange[0], ymax = self.yrange[1])
            else:
                axis1.set_ylim(ymin = 0.0, ymax = 5.0)

        if self.plot_subplot:
            axis2 = ax2.get_axes()
            if self.plot_e is not None:
                axis2.set_xlim(xmin = min(self.plot_x), xmax = max(self.plot_x))
            else:
                if self.xrange is not None:
                    axis2.set_xlim(xmin = self.xrange[0], xmax = self.xrange[1])
                else:
                    axis2.set_xlim(xmin = 0.0, xmax = 5.0)
#            axis2.set_ylim(ymin = -3.0e-9, ymax = 3.0e-9)

        plt.savefig('_fig-' + animate + '.png')
#        plt.savefig('_fig-' + animate + '.svg')

        plt.draw()
        plt.close()
#        plt.show()
        self.plot_animate += 1

