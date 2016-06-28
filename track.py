import random
from copy import copy
import numpy as np
from scipy.ndimage.filters import uniform_filter
from scipy.spatial.distance import cdist
import matplotlib.pyplot as plt
import matplotlib.animation as animation

np.seterr(all='ignore')

A = np.identity(4)
A[0,2] = 1.
A[1,3] = 1.

sigma = 1.


def likelihood_constant(positions, img):
    ratio = uniform_filter(img.astype(np.float), size=5)
    weights = np.zeros(len(positions))
    for i in xrange(len(positions)):
        try:
            if np.min(positions[i]) >= 0:
                weights[i] = ratio[positions[i, 0], positions[i, 1]]
        except IndexError:
            # particles outside the boundary should vanish
            pass

    if np.sum(weights) == 0:
        weights = np.ones_like(weights) / len(positions)
    else:
        weights = weights / np.sum(weights)

    return weights

def likelihood_velocity(positions, velocities, img_now, img_before):
    ratio_now = uniform_filter(img_now.astype(np.float), size=5)
    ratio_before = uniform_filter(img_before.astype(np.float), size=5)
    weights = np.zeros(len(positions))
    for i in xrange(len(positions)):
        try:
            if np.min(positions[i] >= 0) and np.min(positions[i] - velocities[i]) >= 0:
                weights[i] = ratio_now[positions[i, 0], positions[i, 1]] * ratio_before[positions[i, 0] - velocities[i, 0], positions[i, 1] - velocities[i, 1]]
        except IndexError:
            pass
    if np.sum(weights) == 0:
        weights = np.ones_like(weights) / len(positions)
    else:
        weights = weights / np.sum(weights)

    return weights


class Particles(object):

    def __init__(self, particle_number=100):
        self.positions = np.zeros((particle_number, 2))
        self.velocities = np.zeros((particle_number, 2)).astype(np.int)
        self.weights = np.ones(particle_number) / particle_number
        self.particle_number = particle_number

    def __getitem__(self, index):
        return self.positions[index]

    def set_initial_location(self, img):
        positions = []
        while(len(positions) < self.particle_number):
            x = random.randint(0, img.shape[0] - 1)
            y = random.randint(0, img.shape[1] - 1)
            prob = np.random.random_sample()
            if prob < img[x,y]:
                positions.append([x,y])
        assert(len(positions) == self.particle_number)
        self.positions = np.squeeze(np.asarray(positions))

    def normalize_weights(self):
        self.weights = np.nan_to_num(self.weights)
        if np.sum(self.weights) == 0.:
            self.weights = np.ones(self.particle_number) / self.particle_number
        else:
            self.weights = self.weights / np.sum(self.weights)

    def reweight(self, img):
        self.weights = likelihood_constant(self.positions, img)

    def reweight_velocity(self, img_now, img_before):
        self.weights = likelihood_velocity(self.positions, self.velocities, img_now, img_before)

    def mode(self):
        index = np.argmax(self.weights)
        position = np.copy(self.positions[index])
        if position[0] < 0:
            position[0] = 0
        elif position[0] > 29:
            position[0] = 29
        if position[1] < 0:
            position[1] = 0
        elif position[1] > 39:
            position[1] = 39

        return position

    def mean(self):
        self.weights = self.weights / np.sum(self.weights)
        position = np.sum(self.positions * self.weights[:, np.newaxis], axis=0)
        assert(position.ndim == 1)
        assert(len(position) == 2)
        position[np.where(position < 0.)] = 0.
        position = np.round(position, 0).astype(np.int)
        if position[0] < 0:
            position[0] = 0
        elif position[0] > 29:
            position[0] = 29
        if position[1] < 0:
            position[1] = 0
        elif position[1] > 39:
            position[1] = 39

        return position

    def transition_probability(self, next_particles):
        x_next = np.concatenate((next_particles.positions, next_particles.velocities), axis=1)
        x_now = np.concatenate((self.positions, self.velocities), axis=1)
        matrix = np.exp(-0.5 * np.square(cdist(x_next, x_now.dot(A.T))) / (sigma ** 2))
        summation = np.sum(matrix, axis=1)
        matrix = matrix / summation[:, np.newaxis]
        matrix[np.isnan(matrix)] = 1. / self.particle_number
        assert np.allclose(np.sum(matrix, axis=1), 1), matrix
        return matrix

    def sampling(self):
        delta_positions = np.round(sigma * np.random.randn(self.particle_number, 2), 0).astype(np.int)
        delta_velocities = np.round(sigma * np.random.randn(self.particle_number, 2), 0).astype(np.int)
        self.positions += delta_positions + self.velocities
        self.velocities += delta_velocities

    def resampling(self):
        # sample of index indicates which index of particle to be sampled
        samples = np.random.multinomial(n=self.particle_number, pvals=self.weights)

        positions = []
        velocities = []
        for i, num in enumerate(samples):
            for _ in xrange(num):
                positions.append(self.positions[i])
                velocities.append(self.velocities[i])
        positions = np.squeeze(np.array(positions))
        velocities = np.squeeze(np.array(velocities))

        particles = Particles(self.particle_number)
        particles.positions = np.copy(positions)
        particles.velocities = np.copy(velocities)
        return particles


class Trajectory(object):

    def __init__(self, path, frame_number=400, shape=(30,40)):
        assert(path.ndim == 2)
        assert(len(path) == frame_number)
        self.frame_number = frame_number
        self.shape = shape
        self.path = path

    def show(self, background=None, intensity=5):
        if background is None:
            movie = np.zeros((self.frame_number,) + self.shape)
        else:
            movie = np.copy(background.ndarray)
        for i in xrange(self.frame_number):
            movie[i, self.path[i, 0], self.path[i, 1]] = intensity

        fig = plt.figure()
        imgs = []
        for pic in movie:
            im = plt.imshow(pic, cmap='gray')
            imgs.append([im])
        ani = animation.ArtistAnimation(fig, imgs, interval=1., repeat_delay=10000)
        plt.show()

    def save(self, directory, background=None, intensity=5):
        if background is None:
            movie = np.zeros((self.frame_number,) + self.shape)
        else:
            movie = np.copy(background.ndarray)
        for i in xrange(self.frame_number):
            movie[i, self.path[i,0], self.path[i,1]] = intensity

        for i in xrange(self.frame_number):
            plt.imshow(movie[i], cmap='gray')
            plt.savefig(directory + "%03d.png" % (i))
            plt.clf()


class Trajectories(object):

    def __init__(self, frame_number=400, shape=(30,40), particle_number=100):
        self.frame_number = frame_number
        self.shape = shape
        self.particle_number = particle_number
        self.trajectories = []

    def __getitem__(self, index):
        return self.trajectories[index]

    def filtering(self, movie):
        ps = Particles(self.particle_number)
        ps.set_initial_location(movie[0])
        ps.reweight(movie[0])
        self.trajectories.append(copy(ps))

        i = 1
        while(len(self.trajectories) < self.frame_number):
            ps = ps.resampling()
            ps.sampling()
            ps.reweight_velocity(movie[i], movie[i - 1])
            self.trajectories.append(copy(ps))
            i += 1

    def mode(self):
        path = []
        for particles in self.trajectories:
            path.append(particles.mode())
        path = np.squeeze(np.asarray(path))
        assert 0 <= np.min(path[0])
        assert np.max(path[0]) < self.shape[0]
        assert 0 <= np.min(path[1])
        assert np.max(path[1]) < self.shape[1]
        t = Trajectory(path, self.frame_number, self.shape)

        return t

    def mean(self):
        path = []
        for particles in self.trajectories:
            path.append(particles.mean())
        path = np.squeeze(np.asarray(path))
        assert 0 <= np.min(path[0])
        assert np.max(path[0]) < self.shape[0]
        assert 0 <= np.min(path[1])
        assert np.max(path[1]) < self.shape[1]
        t = Trajectory(path, self.frame_number, self.shape)

        return t

    def smoothing(self):
        for i in xrange(self.frame_number - 2, -1, -1):
            matrix = self.trajectories[i].transition_probability(next_particles=self.trajectories[i + 1])
            denominator = self.trajectories[i].weights.dot(matrix)
            assert np.allclose(np.sum(denominator), 1)
            w = self.trajectories[i + 1].weights / denominator
            w[np.isnan(w)] = 0
            w[np.where(denominator == 0.)] = 0
            update = matrix.dot(w)
            self.trajectories[i].weights = self.trajectories[i].weights * update
            self.trajectories[i].normalize_weights()

    def show_image(self, index, background=None):
        if background is None:
            img = np.zeros(self.shape)
        else:
            img = np.copy(background[index])

        particles = self.trajectories[index]
        weight_max = np.max(particles.weights)
        intensities = 5. * particles.weights / weight_max
        for position, intensity in zip(particles.positions, intensities):
            try:
                if img[position[0], position[1]] < intensity:
                    img[position[0], position[1]] = intensity
            except:
                pass

        plt.imshow(img, cmap='gray')
        plt.show()

    def show_movie(self, background=None, interval=1, repeat_delay=10000):
        if background is None:
            movie = np.zeros((self.frame_number,) + self.shape)
        else:
            movie = np.copy(background.ndarray)
        for i, particles in enumerate(self.trajectories):
            weight_max = np.max(particles.weights)
            intensities = 10. * particles.weights / weight_max
            for position, intensity in zip(particles.positions, intensities):
                try:
                    if movie[i, position[0], position[1]] < intensity:
                        movie[i, position[0], position[1]] = intensity
                except:
                    pass

        fig = plt.figure()
        imgs = []
        for pic in movie:
            im = plt.imshow(pic, cmap='gray')
            imgs.append([im])
        ani = animation.ArtistAnimation(fig, imgs, interval=interval, repeat_delay=repeat_delay)
        plt.show()

    def save(self, directory, background=None, intensity=5):
        if background is None:
            movie = np.zeros((self.frame_number,) + self.shape)
        else:
            movie = np.copy(background.ndarray)
        for i, particles in enumerate(self.trajectories):
            weight_max = np.max(particles.weights)
            intensities = 5. * particles.weights / weight_max
            for position, intensity in zip(particles.positions, intensities):
                try:
                    if movie[i, position[0], position[1]] < intensity:
                        movie[i, position[0], position[1]] = intensity
                except:
                    pass

        for i in xrange(self.frame_number):
            plt.imshow(movie[i], cmap='gray')
            plt.savefig(directory + "%03d.png" % (i))
            plt.clf()


class Movie(object):

    def __init__(self, ndarray=None):
        if ndarray is None:
            ndarray = np.empty(shape=(400,30,40))
            for i in xrange(1,401,1):
                filename = 'txt/img' + '{0:03d}'.format(i) + '.txt'
                f = open(filename)
                data = f.read()
                binary_list = []
                for j in xrange(len(data)):
                    if data[j] == '0':
                        binary_list.append(0)
                    elif data[j] == '1':
                        binary_list.append(1)
                image = np.asarray(binary_list).reshape(30,40)
                ndarray[i-1] = image
            self.ndarray = ndarray
        else:
            self.ndarray = ndarray
        self.frame_number = ndarray.shape[0]
        self.frame_height = ndarray.shape[1]
        self.frame_width = ndarray.shape[2]
        self.frame_shape = ndarray.shape[1:]

    def __getitem__(self, index):
        return self.ndarray[index]

    def median_filter(self, size=5):
        uniform_blurred = uniform_filter(self.ndarray.astype(np.float), size=size)
        binary = uniform_blurred > 0.5
        binary = binary.astype(np.int)
        return Movie(ndarray=binary)

    def show(self, index=None):
        if index is None:
            fig = plt.figure()
            imgs = []
            for pic in self.ndarray:
                im = plt.imshow(pic, cmap='gray')
                imgs.append([im])
            ani = animation.ArtistAnimation(fig, imgs, interval=1, repeat_delay=10000)
        else:
            plt.imshow(self.ndarray[index], cmap='gray')
        plt.show()


def main():
    m = Movie()

    # de-noise movie using median filter
    denoised = m.median_filter()

    ts = Trajectories(particle_number=1000)

    # perform particle filter
    ts.filtering(denoised)

    # mode of distribution
    t = ts.mode()
    t.save("results/filter/mode/", m)

    # mean of distribution
    t = ts.mean()
    t.save("results/filter/mean/", m)

    # distribution itself
    ts.save("results/filter/distribution/", m)

    # perform particle smoothing
    ts.smoothing()

    t = ts.mode()
    t.save("results/smoother/mode/", m)

    t = ts.mean()
    t.save("results/smoother/mean/", m)

    ts.save("results/smoother/distribution/", m)


if __name__ == '__main__':
    main()
