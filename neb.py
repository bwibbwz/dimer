from math import sqrt

''' Ideas
Save each tangent as a trajectory (like ase.vibrations) and/or textfile.
Make the barrier energy accesible (get_barrier_energy())
Make it possible to seperate optimizers for each image. (might help in order to get BFGS working with NEB).
'''

import numpy as np

from ase.parallel import world, rank, size
from ase.calculators.singlepoint import SinglePointCalculator
from ase.io import read
from ase.dimer import normalize, norm

class NEB:
    def __init__(self, images, k=1.0, climb=False, parallel=False):
        self.images = images

        # NB: Need support for variable spring constants
        self.k = k
        self.climb = climb
        self.parallel = parallel
        self.natoms = len(images[0])

        # Make sure all the images are of even length.
        # NB: This test should be more elaborate and include species and possible strangeness in the path.
        assert [len(images[0]) for _ in images] == [len(img) for img in images] 

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

        # This option should be available to the users
        self.spring_force = 'norm'

    def interpolate(self, initial=0, final=-1):
        """Interpolate linearly between initial and final images."""
        if final < 0:
            final = self.nimages + final
        n = final - initial
        pos1 = self.images[initial].get_positions()
        pos2 = self.images[final].get_positions()
        d = (pos2 - pos1) / n
        for i in range(1, n):
            self.images[initial + i].set_positions(pos1 + i * d)

    def get_positions(self):
        positions = np.zeros(((self.nimages - 2) * self.natoms, 3))
        n1 = 0
        for image in self.images[1:-1]:
            n2 = n1 + self.natoms
            positions[n1:n2] = image.get_positions()
            n1 = n2
        return positions

    def set_positions(self, positions):
        n1 = 0
        for image in self.images[1:-1]:
            n2 = n1 + self.natoms
            image.set_positions(positions[n1:n2])
            n1 = n2

    def update_tangents(self):
        images = self.images
        t_m = images[1].get_positions() - images[0].get_positions()
        for i in range(1, self.nimages - 1):
            t_p = (images[i + 1].get_positions() - images[i].get_positions())
            e = self.energies[i]
            e_m = self.energies[i - 1]
            e_p = self.energies[i + 1]
            # NB: Check the below definition, it might have been flipped to use the lower energy tangent.
            if e < e_m and e > e_p:
                t = t_m
            elif e > e_m and e < e_p:
                t = t_p
            else:
                # BUG: Possible error when end images become highest.
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

    def calculate_image_energies_and_forces(self, i):
        self.energies[i] = self.images[i].get_potential_energy()
        self.forces['real'][i] = self.images[i].get_forces()

    def calculate_energies_and_forces(self):
        images = self.images

        if not self.parallel:
            # Do all images - one at a time:
            for i in range(1, self.nimages - 1):
                self.calculate_image_energies_and_forces(i)
        else:
            # Parallelize over images:
            i = rank * (self.nimages - 2) // size + 1
            try:
                self.energies[i] = images[i].get_potential_energy()
                self.forces['real'][i] = images[i].get_forces()
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
                world.broadcast(self.energies[i:i], root) # ATH
                world.broadcast(self.forces['real'][i], root)

    def get_forces(self):
        """Evaluate and return the forces."""

        # Update the real forces and energies
        self.calculate_energies_and_forces()

        # Update the highest energy image
        self.imax = 1 + np.argsort(self.energies[1:-1])[-1]
        self.emax = self.energies[self.imax]

        # Calculate the tangents of all the images
        self.update_tangents()

        # Prjoect the forces for each image
        self.project_forces()

        print self.images[1]._calc.get_count()
        return self.forces['neb'][1:self.nimages-1].reshape((-1, 3))

    def get_norm_image_spring_force(self, i):
        t = self.tangents[i]
        nt = t / np.vdot(t, t)**0.5
        p_m = self.images[i - 1].get_positions()
        p = self.images[i].get_positions()
        p_p = self.images[i + 1].get_positions()
        nt_m = np.vdot(p - p_m, p - p_m)**0.5
        nt_p = np.vdot(p_p - p, p_p - p)**0.5
        return (nt_p - nt_m) * self.k * t

    def get_full_image_spring_force(self, i):
        p_m = self.images[i - 1].get_positions()
        p = self.images[i].get_positions()
        p_p = self.images[i + 1].get_positions()
        t_m = p - p_m
        t_p = p_p - p
        return (t_p - t_m) * self.k

    def get_image_spring_force(self, i):
        if self.spring_force == 'norm':
            return self.get_norm_image_spring_force(i)
        elif self.spring_force == 'full':
            return self.get_full_image_spring_force(i)
        else:
            raise NotImplementedError('Only "norm" and "full" are allowed.')

    def project_forces(self, sort='real'):
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
                self.forces['neb'][i] = f_r_perp + f_s

    def get_potential_energy(self):
        return self.emax

    def __len__(self):
        return (self.nimages - 2) * self.natoms

class SingleCalculatorNEB(NEB):
    def __init__(self, images, k=0.1, climb=False):
        if isinstance(images, str):
            # this is a filename
            traj = read(images, '0:')
            images = []
            for atoms in traj:
                images.append(atoms)

        NEB.__init__(self, images, k, climb, False)
        self.calculators = [None] * self.nimages
        self.energies_ok = False
 
    def refine(self, steps=1, begin=0, end=-1):
        """Refine the NEB trajectory."""
        if end < 0:
            end = self.nimages + end
        j = begin
        n = end - begin
        for i in range(n):
            for k in range(steps):
                self.images.insert(j + 1, self.images[j].copy())
                self.calculators.insert(j + 1, None)
            self.nimages = len(self.images)
            self.interpolate(j, j + steps + 1)
            j += steps + 1

    def set_positions(self, positions):
        # new positions -> new forces
        if self.energies_ok:
            # restore calculators
            self.set_calculators(self.calculators[1:-1])
        NEB.set_positions(self, positions)

    def get_calculators(self):
        """Return the original calculators."""
        calculators = []
        for i, image in enumerate(self.images):
            if self.calculators[i] is None:
                calculators.append(image.get_calculator())
            else:
                calculators.append(self.calculators[i])
        return calculators
    
    def set_calculators(self, calculators):
        """Set new calculators to the images."""
        self.energies_ok = False

        if not isinstance(calculators, list):
            calculators = [calculators] * self.nimages

        n = len(calculators)
        if n == self.nimages:
            for i in range(self.nimages):
                self.images[i].set_calculator(calculators[i])   
        elif n == self.nimages - 2:
            for i in range(1, self.nimages -1):
                self.images[i].set_calculator(calculators[i-1])   
        else:
            raise RuntimeError(
                'len(calculators)=%d does not fit to len(images)=%d'
                % (n, self.nimages))

    def get_energies_and_forces(self, all=False):
        """Evaluate energies and forces and hide the calculators"""
        if self.energies_ok:
            return

        images = self.images
        forces = np.zeros(((self.nimages - 2), self.natoms, 3))
        energies = np.zeros(self.nimages - 2)
        self.emax = -1.e32

        def calculate_and_hide(i):
            image = self.images[i]
            calc = image.get_calculator()
            if self.calculators[i] is None:
                self.calculators[i] = calc
            if calc is not None:
                if not isinstance(calc, SinglePointCalculator):
                    self.images[i].set_calculator(
                        SinglePointCalculator(image.get_potential_energy(),
                                              image.get_forces(),
                                              None,
                                              None,
                                              image))
                self.emax = min(self.emax, image.get_potential_energy())

        if all and self.calculators[0] is None:
            calculate_and_hide(0)

        # Do all images - one at a time:
        for i in range(1, self.nimages - 1):
            calculate_and_hide(i)

        if all and self.calculators[-1] is None:
            calculate_and_hide(-1)

        self.energies_ok = True
       
    def get_forces(self):
        self.get_energies_and_forces()
        return NEB.get_forces(self)

    def n(self):
        return self.nimages

    def write(self, filename):
        from ase.io.trajectory import PickleTrajectory
        traj = PickleTrajectory(filename, 'w', self)
        traj.write()
        traj.close()

    def __add__(self, other):
        for image in other:
            self.images.append(image)
        return self

def fit(images):
    E = [i.get_potential_energy() for i in images]
    F = [i.get_forces() for i in images]
    R = [i.get_positions() for i in images]
    return fit0(E, F, R)

def fit0(E, F, R):
    E = np.array(E) - E[0]
    n = len(E)
    Efit = np.empty((n - 1) * 20 + 1)
    Sfit = np.empty((n - 1) * 20 + 1)

    s = [0]
    for i in range(n - 1):
        s.append(s[-1] + sqrt(((R[i + 1] - R[i])**2).sum()))
#	print s[i], s[i] - s[i-1]

    lines = []
    for i in range(n):
        if i == 0:
            d = R[1] - R[0]
            ds = 0.5 * s[1]
        elif i == n - 1:
            d = R[-1] - R[-2]
            ds = 0.5 * (s[-1] - s[-2])
        else:
            d = R[i + 1] - R[i - 1]
            ds = 0.25 * (s[i + 1] - s[i - 1])

        d = d / sqrt((d**2).sum())
        dEds = -(F[i] * d).sum()
        x = np.linspace(s[i] - ds, s[i] + ds, 3)
        y = E[i] + dEds * (x - s[i])
        lines.append((x, y))

        if i > 0:
            s0 = s[i - 1]
            s1 = s[i]
            x = np.linspace(s0, s1, 20, endpoint=False)
            c = np.linalg.solve(np.array([(1, s0,   s0**2,     s0**3),
                                          (1, s1,   s1**2,     s1**3),
                                          (0,  1,  2 * s0, 3 * s0**2),
                                          (0,  1,  2 * s1, 3 * s1**2)]),
                                np.array([E[i - 1], E[i], dEds0, dEds]))
            y = c[0] + x * (c[1] + x * (c[2] + x * c[3]))
            Sfit[(i - 1) * 20:i * 20] = x
            Efit[(i - 1) * 20:i * 20] = y
        
        dEds0 = dEds

    Sfit[-1] = s[-1]
    Efit[-1] = E[-1]
    return s, E, Sfit, Efit, lines
