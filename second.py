from math import sqrt, cos, pi

import numpy as np

from numpy import cos, sin, pi
from ase.parallel import world, rank, size
from ase.neb import NEB
from ase.dimer import normalize, DimerEigenmodeSearch, MinModeAtoms, perpendicular_vector
from ase.dimer import norm, parallel_vector, DimerControl

class second(NEB):
    def __init__(self, images, control, k=0.1, climb=False, parallel=False):
        self.control = control
        NEB.__init__(self, images, k, climb, parallel)

#        self.first_modes = [None]*self.nimages
        self.first_modes = np.zeros((self.nimages, self.natoms, 3))
#        self.second_modes = [None]*self.nimages
        self.second_modes = np.zeros((self.nimages, self.natoms, 3))
#        self.first_curvatures = [None]*self.nimages
        self.first_curvatures = np.zeros(self.nimages)
#        self.second_curvatures = [None]*self.nimages
#        self.second_modes_calculated = [False]*self.nimages

        self.dimer_forces = np.zeros((self.nimages, self.natoms, 3))

        self.minmodes = []
        for k in range(self.nimages):
            m = MinModeAtoms(images[k], control)
            m.initialize_eigenmodes() # ATH, should be perp. to tangent
            self.first_modes[k] = m.get_eigenmode()
            self.minmodes += [m]

        # ATH DEV
        self.animate = 0
        self.enMat = None
        self.xVec = None
        self.yVec = None
        self.dev_plot = False # Can be turned on manually
        self.decouple_modes = False

    def get_forces(self):
        """Evaluate and return the forces."""

        # Update the clean forces and energies
        self.calculate_energies_and_forces()

        # Update the highest energy image
        self.imax = 1 + np.argsort(self.energies[1:-1])[-1]
        self.emax = self.energies[self.imax]

        # Calculate the tangents of all the images
        self.update_tangents()

        # IDEA: If f_c_perp is small force dimer rotation?

        # Calculate the modes
        self.calculate_eigenmodes()

        # Prjoect the forces for each image
        self.invert_eigenmode_forces()
        self.project_forces()
        if self.dev_plot:
            self.plot_2d_pes()
        self.control.increment_counter('optcount')
        return self.projected_forces[1:self.nimages-1].reshape((-1, 3))

    def calculate_eigenmodes(self):
        if not self.parallel:
            for k in range(1, self.nimages - 1):
                m = self.first_modes[k]
                t = self.tangents[k]
                m = normalize(perpendicular_vector(m, normalize(t)))
                if self.decouple_modes:
                    search = DimerEigenmodeSearch(self.minmodes[k], control = self.control, eigenmode = m)
                else:
                    search = DimerEigenmodeSearch(self.minmodes[k], control = self.control, eigenmode = m, basis = [normalize(t)])
                search.converge_to_eigenmode()
                self.first_modes[k] = search.get_eigenmode()
                self.first_curvatures[k] = search.get_curvature()
#                self.second_modes_calculated[k] = False
        else:
            k = self.world.rank * (self.nimages - 2) // self.world.size + 1
            m = self.first_modes[k]
            t = self.tangents[k]
            m = normalize(perpendicular_vector(m, normalize(t)))
            search = DimerEigenmodeSearch(self.minmodes[k], control = self.control, eigenmode = m, basis = [normalize(t)])
            try:
                search.converge_to_eigenmode()
            except:
                error = self.world.sum(1.0)
                raise
            else:
                error = self.world.sum(0.0)
                if error:
                    raise RuntimeError('Parallel Dimer failed!')
            self.first_modes[k] = search.get_eigenmode()
            self.first_curvatures[k] = search.get_curvature()
#            self.second_modes_calculated[k] = False # NOT USED ANYMORE
            for k in range(1, self.nimages - 1):
                root = (k - 1) * self.world.size // (self.nimages - 2)
                self.world.broadcast(self.first_modes[k], root)
                self.world.broadcast(self.first_curvatures[k:k], root)
#                self.world.broadcast(self.second_modes_calculated[k], root)
                

    def invert_eigenmode_forces(self):
        for k in range(1, self.nimages - 1):
            f_clean = self.clean_forces[k]
            t = self.tangents[k]
            m1 = self.first_modes[k]
#            m2 = self.second_modes[k]
            c1 = self.first_curvatures[k]
#            c2 = self.second_curvatures[k]
            if not self.climb or k == self.imax or True:
                self.dimer_forces[k] = f_clean - 2 * parallel_vector(f_clean, m1)
            # ATH: Maybe invert both while trying to escape the bad regions
#            if self.second_modes_calculated[k]:
#                self.dimer_forces[k] = self.dimer_forces[k] - 2 * parallel_vector(self.dimer_forces[k], m2)
#                print 'image %i, forces inverted' % k
                self.clean_forces[k] = self.dimer_forces[k]

# ----------------------------------------------------------------
# --------------- Outdated and development methods ---------------
# ----------------------------------------------------------------
    def plot_2d_pes(self):
        t  = self.tangents[1:self.nimages - 1]
        m1 = self.first_modes[1:self.nimages - 1]
#        m2 = self.second_modes[1:self.nimages - 1]

        def make_arrow(pos, orient, c):
            p = pos[-1]
            o = orient[-1]
            pylab.arrow(p[0] + o[0]*0.2, p[1] + o[1]*0.2, o[0]*-0.4, o[1]*-0.4, width = 0.001, ec = c, fc = c)
            pylab.arrow(p[0] - o[0]*0.2, p[1] - o[1]*0.2, o[0]*0.4, o[1]*0.4, width = 0.001, ec = c, fc = c)

        p = []
        for img in self.images:
            p += [img.get_positions()]
        f = p.pop()
        i = p.pop(0)

        import pylab
        fig = pylab.figure(figsize=(9, 9))
#        fig.add_axes(pylab.Axes(fig, [0.0, 0.0, 0.0, 0.0]))
        ax = pylab.axes()
        fig.set(transform=fig.transFigure, figure=fig)
        

        def adjust_spines(ax,spines):
            for loc, spine in ax.spines.iteritems():
                if loc in spines:
                    spine.set_position(('outward',10)) # outward by 10 points
                    spine.set_smart_bounds(True)
                else:
                    spine.set_color('none') # don't draw spine
        
            # turn off ticks where there is no spine
            if 'left' in spines:
                ax.yaxis.set_ticks_position('left')
            else:
                # no yaxis ticks
                ax.yaxis.set_ticks([])
        
            if 'bottom' in spines:
                ax.xaxis.set_ticks_position('bottom')
            else:
                # no xaxis ticks
                ax.xaxis.set_ticks([])

        adjust_spines(ax, [])

        
        # Plot the static stuff
        cir = pylab.Circle((i[-1][0], i[-1][1]), radius = 0.15, fc = 'y')
        pylab.gca().add_patch(cir)
        cir = pylab.Circle((f[-1][0], f[-1][1]), radius = 0.15, fc = 'y')
        pylab.gca().add_patch(cir)
#        pylab.plot(i[-1][0], f[-1][1], 'kx')

        f = self.projected_forces[1:self.nimages-1]
        cf = self.clean_forces[1:self.nimages-1] # This is the Dimer force
#        rf = self.nodimer_forces[1:self.nimages-1] # ATH, missing
        rf = cf.copy()
        for k in range(1, self.nimages - 1):
            rf[k-1] = self.images[k].get_forces()
        # Plot the band
        for k in range(len(p)):
          if k%2 == 0:
            an = np.linspace(0, 2 * pi, 100)
            can = cos(an)
            san = sin(an)
            for ka in range(len(an)):
                can[ka] *= 0.2
                can[ka] += p[k][-1][0]
                san[ka] *= 0.2
                san[ka] += p[k][-1][1]

            pylab.plot(can, san, 'k:', zorder = -1)
#            cir = pylab.Circle((p[k][-1][0], p[k][-1][1]), radius = 0.2, fc = 'y')
#            pylab.gca().add_patch(cir)
        for k in range(len(p)):
          if k%2 != 0:
            if k + 1 == self.imax and self.climb:
                cir = pylab.Circle((p[k][-1][0], p[k][-1][1]), radius = 0.2, fc = 'c')
            else:
                cir = pylab.Circle((p[k][-1][0], p[k][-1][1]), radius = 0.2, fc = 'y')
            pylab.gca().add_patch(cir)
            pylab.arrow(p[k][-1][0], p[k][-1][1], rf[k][-1][0], rf[k][-1][1], width = 0.02, ec = 'k', fc = 'w')
#            pylab.arrow(p[k][-1][0], p[k][-1][1], cf[k][-1][0], cf[k][-1][1], width = 0.02, ec = 'g', fc = 'g')
            pylab.arrow(p[k][-1][0], p[k][-1][1], f[k][-1][0], f[k][-1][1], width = 0.02, ec = 'k', fc = 'k')
            make_arrow(p[k], m1[k], 'b')
            make_arrow(p[k], t[k], 'r')
#            if m2[k] is not None and self.second_modes_calculated[k]:
#                make_arrow(p[k], m2[k], 'g')
#            if eff[k] is not None:
#                make_arrow(p[k], eff[k], 'k')
#            pylab.text(p[k][-1][0], p[k][-1][1] - 0.2, str(k + 1), color = 'k')
#            if self.climb:
#                pylab.text(2.0, 2.0, 'C', color = 'k')
        
        ax.patch.set_facecolor('green')
        ax.patch.set_transform(fig.transFigure)
        ax.patch.set_visible(False)
        # Plot the contour
        if self.enMat is not None:
#            pylab.contourf(self.xVec, self.yVec, self.enMat, 30, antialiased = True)
            pylab.contourf(self.xVec, self.yVec, self.enMat, 30, zorder = -2, transform = fig.transFigure, figure = fig)

        pylab.axis('scaled')

        if self.animate < 10:
            animate = '000' + str(self.animate)
        elif self.animate < 100:
            animate = '00' + str(self.animate)
        elif self.animate < 1000:
            animate = '0' + str(self.animate)
        else:
            animate = str(self.animate)

#        fig.canvas.print_svg('_rass' + animate + '.svg', edgecolor = 'none', facecolor = 'none')

#        fig.axes[0].set_axes_bg_color('green')

#        fig.canvas.set_visible(False)
#        print fig.artists
#        print fig.canvas
#        print ax.margins()
#        print ax.patch.get_xy()


#        ax.axes.get_xaxis().set_visible(False)
#        ax.axes.get_yaxis().set_visible(False)

#        pylab.savefig('_rass-' + animate + '.png')
#        pylab.savefig('_rass-' + animate + '.svg')

#        fig.canvas.draw()
#        pylab.draw()
        pylab.savefig('_rass' + animate + '.svg', dpi = 300, facecolor = 'none', edgecolor = 'none', bbox_inches = 'tight', pad_inches = 0.0)
#        pylab.close()
#        pylab.draw()
#        pylab.savefig('_rass' + animate + '.png', dpi = 300, facecolor = 'none', edgecolor = 'none', bbox_inches = 'tight', pad_inches = 0.0)
#        pylab.savefig('_rass' + animate + '.png', format = 'png', bbox_inches = 'tight', facecolor = 'none', pad_inches = 0.5)
#        pylab.close()

#        pylab.draw()
#        pylab.close()
#        pylab.show()

        # Plot the contour
#        if self.enMat is not None:
#            pylab.contourf(self.yVec, self.xVec, self.enMat, 30, antialiased = True)
#        pylab.axis('scaled')

#        pylab.savefig('_rass-' + animate + '.pdf')

#        pylab.draw()
#        pylab.close()

        self.animate += 1

