from math import sqrt

import numpy as np
from math import atan, pi

from ase.parallel import world, rank, size
from ase.calculators.singlepoint import SinglePointCalculator
from ase.io import read

class NEB:
    def __init__(self, images, k=1.0, climb=False, parallel=False):
        self.images = images

        # NB: Need support for variable spring constants
        self.k = k
        self.climb = climb
        self.parallel = parallel
        self.natoms = len(images[0])

        # Make sure all the images are of even length.
        # NB: This test should be more elaborate and include species and
        #     possible strangeness in the path.
        assert [len(images[0]) for _ in images] == \
               [len(img) for img in images]

        self.nimages = len(images)
        self.emax = np.nan
        self.imax = None

        # Set up empty arrays to store forces, energies and tangents
        self.forces = {}
        self.forces['real'] = np.zeros((self.nimages, self.natoms, 3))
        self.forces['neb'] = np.zeros((self.nimages, self.natoms, 3))
        self.forces['spring'] = np.zeros((self.nimages, self.natoms, 3))
        self.energies = np.zeros(self.nimages)
        self.tangents = np.zeros((self.nimages, self.natoms, 3))

        # Get the end point energies if they are available.
        try:
            self.energies[0] = self.images[0].get_potential_energy()
        except:
            self.energies[0] = -np.inf
        try:
            self.energies[-1] = self.images[-1].get_potential_energy()
        except:
            self.energies[-1] = -np.inf

        # Set the spring force implementation
        self.spring_force = 'norm'

    def interpolate(self, initial=0, final=-1):
        """Interpolate linearly between two images. The end images are
           used by default"""
        if final < 0:
            final = self.nimages + final
        n = final - initial
        pos1 = self.images[initial].get_positions()
        pos2 = self.images[final].get_positions()
        d = (pos2 - pos1) / n
        for i in range(1, n):
            self.images[initial + i].set_positions(pos1 + i * d)

    def get_positions(self):
        """Return the positions of all the atoms for all the images in
           a single array."""
        positions = np.zeros(((self.nimages - 2) * self.natoms, 3))
        n1 = 0
        for image in self.images[1:-1]:
            n2 = n1 + self.natoms
            positions[n1:n2] = image.get_positions()
            n1 = n2
        return positions

    def set_positions(self, positions):
        """Set the positions of the images."""
        n1 = 0
        for image in self.images[1:-1]:
            n2 = n1 + self.natoms
            image.set_positions(positions[n1:n2])
            n1 = n2

            # Parallel NEB with Jacapo needs this:
            try:
                image.get_calculator().set_atoms(image)
            except AttributeError:
                pass

    def update_tangents(self):
        """Update the tangent estimates. Only a forward difference tangent,
           towards the neighboring top energy image, is currently
           supported."""
        images = self.images
        t_m = images[1].get_positions() - images[0].get_positions()
        self.tangents[0] = t_m.copy()
        for i in range(1, self.nimages - 1):
            t_p = (images[i + 1].get_positions() - images[i].get_positions())
            e = self.energies[i]
            e_m = self.energies[i - 1]
            e_p = self.energies[i + 1]
            if (e < e_m and e > e_p) or \
               (i == self.nimages - 2 and e_p == -np.inf):
                t = t_m.copy()
            elif (e > e_m and e < e_p) or (i == 1 and e_m == -np.inf):
                t = t_p.copy()
            else:
                e_max = max(abs(e_p - e), abs(e_m - e))
                e_min = min(abs(e_p - e), abs(e_m - e))
                if e_p > e_m:
                    t = t_p * e_max + t_m * e_min
                else:
                    t = t_p * e_min + t_m * e_max
                t /= np.vdot(t, t)**0.5
                t *= (np.vdot(t_m, t_m)**0.5 + np.vdot(t_p, t_p)**0.5) / 2.0
            self.tangents[i] = t
            t_m = t_p
        self.tangents[-1] = t_m.copy()

    def calculate_image_energies_and_forces(self, i):
        """Calculate and store the force and energy for a single image."""
        self.forces['real'][i] = self.images[i].get_forces()
        self.energies[i] = self.images[i].get_potential_energy()

    def calculate_energies_and_forces(self):
        """Calculate and store the forces and energies for the band."""
        images = self.images

        if not self.parallel:
            # Do all images - one at a time:
            for i in range(1, self.nimages - 1):
                self.calculate_image_energies_and_forces(i)
        else:
            # Parallelize over images:
            i = rank * (self.nimages - 2) // size + 1
            try:
                self.calculate_image_energies_and_forces(i)
            except:
                # Make sure other images also fail:
                error = world.sum(1.0)
                raise
            else:
                error = world.sum(0.0)
                if error:
                    raise RuntimeError('Parallel NEB failed')
            for i in range(1, self.nimages - 1):
                root = (i - 1) * size // (self.nimages - 2)
                world.broadcast(self.energies[i : i + 1], root)
                world.broadcast(self.forces['real'][i], root)

    def get_forces(self):
        """Evaluate, modify and return the forces."""

        # Update the real forces and energies
        self.calculate_energies_and_forces()

        # Update the highest energy image
        self.imax = 1 + np.argsort(self.energies[1:-1])[-1]
        self.emax = self.energies[self.imax]

        # Calculate the tangents of all the images
        self.update_tangents()

        # Prjoect the forces for each image
        self.project_forces()

        return self.forces['neb'][1:self.nimages-1].reshape((-1, 3))

    def get_norm_image_spring_force(self, i):
        """Calculate the 'norm' spring force for a single image."""
        t = self.tangents[i]
        nt = t / np.vdot(t, t)**0.5
        p_m = self.images[i - 1].get_positions()
        p = self.images[i].get_positions()
        p_p = self.images[i + 1].get_positions()
        nt_m = np.vdot(p - p_m, p - p_m)**0.5
        nt_p = np.vdot(p_p - p, p_p - p)**0.5
        return (nt_p - nt_m) * self.k * t

    def get_full_image_spring_force(self, i):
        """Calculate the 'full' spring force for a single image."""
        p_m = self.images[i - 1].get_positions()
        p = self.images[i].get_positions()
        p_p = self.images[i + 1].get_positions()
        t_m = p - p_m
        t_p = p_p - p
        return (t_p - t_m) * self.k

    def get_image_spring_force(self, i):
        """Calculate the spring force for a single image."""
        if self.spring_force == 'norm':
            return self.get_norm_image_spring_force(i)
        elif self.spring_force == 'full':
            return self.get_full_image_spring_force(i)
        else:
            e = 'The only supported spring force defintions are: "norm"' + \
                ' and "full".'
            raise NotImplementedError(e)

    def project_forces(self, sort='real'):
        """Project the forces, replace the force components along the path
           with the spring force. The input variable sort is included if
           previous force manipulations have been performed."""
        for i in range(1, self.nimages - 1):
            t = self.tangents[i]
            nt = t / np.vdot(t, t)**0.5
            f_r = self.forces[sort][i]
            f_r_para = np.vdot(f_r, nt) * nt
            f_r_perp = f_r - f_r_para
            if self.climb and i == self.imax:
                self.forces['neb'][i] = f_r - 2 * f_r_para
            else:
                f_s = self.get_image_spring_force(i)
                self.forces['spring'][i] = f_s
                self.forces['neb'][i] = f_r_perp + f_s

    def get_potential_energy(self):
        """Return the energy of the top energy image."""
        return self.emax

    def __len__(self):
        return (self.nimages - 2) * self.natoms
