from math import sqrt, cos, pi

import numpy as np

from ase.parallel import world, rank, size
from ase.neb import NEB
from ase.dimer import MinModeAtoms, normalize, DimerEigenmodeSearch, perpendicular_vector
from ase.dimer import norm, parallel_vector, DimerControl

class second(NEB):
    def __init__(self, images, control, k=0.1, climb=False, parallel=False, oldtangent=False):
        self.control = control
        minmodes = []
        for image in images:
#            minmodes += [MinModeAtoms(image, control.copy())] # ATH control.copy() might not be implemented
            minmodes += [MinModeAtoms(image, control)]
        NEB.__init__(self, minmodes, k, climb, parallel, oldtangent)

        self.second_modes = [None]*self.nimages
        self.last_modes = [None]*self.nimages

        self.nodimer_forces = np.zeros((self.nimages, self.natoms, 3))

        # ATH DEV
        self.animate = 0
        self.enMat = None
        self.xVec = None
        self.yVec = None

        self.switch_min = 0.60
        self.switch_max = 0.95

    def save_nodimer_forces(self):
        for i in range(1, self.nimages - 1):
            self.nodimer_forces[i] = self.images[i].get_forces(real = True)

    def get_forces(self):
        """Evaluate and return the forces."""
#        print '------------------Begin---------------------------'

        # Update the clean forces and energies
        self.calculate_energies_and_forces()
        self.save_nodimer_forces()

        # Update the highest energy image
        self.imax = np.argsort(self.energies)[-1]
        self.emax = self.energies[self.imax]

        # Calculate the tangents of all the images
        self.update_tangents()

        # Check if calculating the second lowest mode is needed
        self.check_mode_overlap()

        # Prjoect the forces for each image
        self.project_forces()
#        for k in range(1, self.nimages - 1):
#            print k, self.clean_forces[k][-1], np.vdot(normalize(self.clean_forces[k]), normalize(self.last_modes[k]))
#            print k, self.tangents[k][-1]

        self.plot_2d_pes()
#        print '-------------------End----------------------------'
        return self.projected_forces[1:self.nimages-1].reshape((-1, 3))

    def check_mode_overlap(self):
#        self.check_mode_overlap_OLD()
        self.check_mode_overlap_NEW()

    def check_mode_overlap_NEW(self):
        self.last_second_modes = [None]*self.nimages
        self.effective_modes = [None]*self.nimages
        for i in range(1, self.nimages - 1):
            mode = self.images[i].get_eigenmode()
            self.last_modes[i] = mode
            tangent = self.tangents[i]
#            f = self.clean_forces[i] # Includes the dimer force-flip
#            f = self.nodimer_forces[i]

            mt = np.vdot(normalize(mode), normalize(tangent))
#            print i, mt, 'Should the second mode be calculated?'
            if abs(mt) > self.switch_min:
                # Detect if the old second mode is similar to the current first mode (what then?)
                if self.second_modes[i] is not None:
                    second_mode = normalize(perpendicular_vector(self.second_modes[i], mode))
                else:
                    second_mode = normalize(perpendicular_vector(mode - normalize(tangent), mode))
                second_dimer = MinModeAtoms(self.images[i].get_atoms(), self.control)
                search = DimerEigenmodeSearch(second_dimer, eigenmode = second_mode, control = self.control, basis = mode)
#                print i, np.vdot(mode, second_mode)
                search.converge_to_eigenmode() # NB: Am I re-calculating the centre image?
                second_mode = search.get_eigenmode()
                self.last_second_modes[i] = second_mode

#                print self.images[i].get_curvature()
#                print 'c', self.clean_forces[i][-1], norm(self.clean_forces[i])
#                print 'n', self.nodimer_forces[i][-1], norm(self.nodimer_forces[i])

                f_clean = self.nodimer_forces[i].copy()
                f = f_clean.copy()
                f -= 2 * parallel_vector(f, mode)
                f -= 2 * parallel_vector(f, second_mode)
                self.clean_forces[i] = f

#                print 'c', self.clean_forces[i][-1], norm(self.clean_forces[i])
#                print 'f', f_clean[-1], norm(f_clean)
#                print 'n', self.nodimer_forces[i][-1], norm(self.nodimer_forces[i])

#                rass_1 = self.images[i].get_projected_forces()
#                f_1 = f_clean - self.images[i].get_projected_forces()
#                f_stripped = f_clean - f_1
#                self.images[i].set_eigenmode(second_mode)
#                self.images[i].get_mode(mode = 2)
#                rass_2 = self.images[i].get_projected_forces()
#                f_2 = f_clean - self.images[i].get_projected_forces()
#                self.images[i].set_eigenmode(mode)
#                print 'n', 'norm_f_1      ', 'norm_f_2      ', 'norm_r_1      ', 'norm_r_2      ', 'dot_f_1_f_1  ', 'dot_modes     '
#                print i, norm(f_1), norm(f_2), norm(rass_1), norm(rass_2), np.vdot(f_1, f_2), np.vdot(mode, second_mode)
#                print f_1[-1]
#                print f_2[-1]
#                print '-'*20

                # Why not invert along both axes?
#                f -= f
#                f += self.images[i].get_forces()
                self.images[i].set_eigenmode(second_mode)
                self.second_modes[i] = mode
                self.last_second_modes[i] = second_mode



    def check_mode_overlap_OLD(self):
        self.last_second_modes = [None]*self.nimages
        self.effective_modes = [None]*self.nimages
        for i in range(1, self.nimages - 1):
            mode = self.images[i].get_eigenmode()
            self.last_modes[i] = mode
            tangent = self.tangents[i]
            f = self.clean_forces[i]

            mt = np.vdot(normalize(mode), normalize(tangent))
            if abs(mt) > self.switch_min: # or True:
                if self.second_modes[i] is not None:
                    second_mode = normalize(perpendicular_vector(self.second_modes[i], mode))
                else:
                    second_mode = normalize(perpendicular_vector(mode - normalize(tangent), mode))
                second_dimer = MinModeAtoms(self.images[i].get_atoms(), self.control)
                search = DimerEigenmodeSearch(second_dimer, eigenmode = second_mode, control = self.control, basis = mode)
                search.converge_to_eigenmode()
                second_mode = search.get_eigenmode()

                if abs(mt) > self.switch_max:
                    switch_f = 0.0
                elif abs(mt) < self.switch_min:
                    switch_f = 1.0
                else:
                    switch_f = 0.5 * (1.0 + cos(pi * (abs(mt) - self.switch_min)/(self.switch_min - self.switch_max)))

                switch_f = 0.5
                eff_mode = normalize(switch_f * mode + (1.0 - switch_f) * second_mode)

                self.images[i].set_eigenmode(eff_mode)
                # Why not invert along both axes?
                f -= f
                f += self.images[i].get_forces()
                self.images[i].set_eigenmode(second_mode)
                self.second_modes[i] = mode
                self.last_second_modes[i] = second_mode
                self.effective_modes[i] = eff_mode

    def plot_2d_pes(self):
        t = self.tangents[1:self.nimages-1]
#        o = []
#        for img in self.images:
#            o += [img.get_eigenmode()]
#        o = o[1:self.nimages-1]
        o = self.last_modes[1:self.nimages-1]
        s = self.last_second_modes[1:self.nimages-1]
        eff = self.effective_modes[1:self.nimages-1]
        def make_arrow(pos, orient, colour):
            pylab.arrow(pos[-1][0] + orient[-1][0] * 0.2, pos[-1][1] + orient[-1][1] * 0.2, orient[-1][0] * -0.4, orient[-1][1] * -0.4, width = 0.001, ec = colour, fc = colour)
            pylab.arrow(pos[-1][0] - orient[-1][0] * 0.2, pos[-1][1] - orient[-1][1] * 0.2, orient[-1][0] * 0.4, orient[-1][1] * 0.4, width = 0.001, ec = colour, fc = colour)

        p = []
        for image in self.images:
            p.append(image.get_positions())

        f = p.pop()
        i = p.pop(0)

        import pylab
        pylab.axes()

        # Plot the static stuff
        cir = pylab.Circle((i[-1][0], i[-1][1]), radius = 0.25, fc = 'y')
        pylab.gca().add_patch(cir)
        cir = pylab.Circle((f[-1][0], f[-1][1]), radius = 0.25, fc = 'y')
        pylab.gca().add_patch(cir)
#        cir = pylab.Circle((i[-3][0], i[-3][1]), radius = 0.10, fc = 'y')
#        pylab.gca().add_patch(cir)
        pylab.plot(i[-1][0], f[-1][1], 'kx')

        f = self.projected_forces[1:self.nimages-1]
        cf = self.clean_forces[1:self.nimages-1]
        rf = self.nodimer_forces[1:self.nimages-1]


        # Plot the band
        for k in range(len(p)):
            if k + 1 == self.imax:
                cir = pylab.Circle((p[k][-1][0], p[k][-1][1]), radius = 0.3, fc = 'c')
            else:
                cir = pylab.Circle((p[k][-1][0], p[k][-1][1]), radius = 0.3, fc = 'y')
            pylab.gca().add_patch(cir)
            pylab.arrow(p[k][-1][0], p[k][-1][1], rf[k][-1][0], rf[k][-1][1], width = 0.02, ec = 'r', fc = 'r')
            pylab.arrow(p[k][-1][0], p[k][-1][1], cf[k][-1][0], cf[k][-1][1], width = 0.02, ec = 'g', fc = 'g')
            pylab.arrow(p[k][-1][0], p[k][-1][1], f[k][-1][0], f[k][-1][1], width = 0.02, ec = 'k', fc = 'k')
            make_arrow(p[k], o[k], 'b')
            make_arrow(p[k], t[k], 'r')
            if s[k] is not None:
                make_arrow(p[k], s[k], 'g')
            if eff[k] is not None:
                make_arrow(p[k], eff[k], 'k')
            pylab.text(p[k][-1][0], p[k][-1][1] - 0.2, str(k + 1), color = 'k')
            if self.climb:
                pylab.text(2.0, 2.0, 'C', color = 'k')

        # Plot the contour
        if self.enMat is not None:
            pylab.contourf(self.yVec, self.xVec, self.enMat, 30, antialiased = True)

        pylab.axis('scaled')

        if self.animate < 10:
            animate = '000' + str(self.animate)
        elif self.animate < 100:
            animate = '00' + str(self.animate)
        elif self.animate < 1000:
            animate = '0' + str(self.animate)
        else:
            animate = str(self.animate)

        pylab.savefig('_rass-' + animate + '.png')

        pylab.draw()
        pylab.close()
#        pylab.show()

        # Plot the contour
#        if self.enMat is not None:
#            pylab.contourf(self.yVec, self.xVec, self.enMat, 30, antialiased = True)
#        pylab.axis('scaled')

#        pylab.savefig('_rass-' + animate + '.pdf')

#        pylab.draw()
#        pylab.close()

        self.animate += 1
