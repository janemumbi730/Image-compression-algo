import numpy as np
import matplotlib.pyplot as plt
from sys import argv

APPEND, FLUSH = 0, 1
losses = []


def cluster_points(pixels, z):
    clusters = [np.empty(shape=[0, 3])] * k
    cost = 0
    for p in pixels:
        min_i, min_d = 0, float('inf')
        for i in range(k):
            d = np.linalg.norm(z[i] - p)
            if d < min_d:
                min_d, min_i = d, i
        clusters[min_i] = np.append(clusters[min_i], [p], axis=0)
        cost += min_d
    losses.append(cost / n)
    return clusters


# get means of each cluster
def get_means(clusters, z):
    means = np.array([]).reshape(0, 3)
    for i in range(k):
        cluster = clusters[i]
        num_pixels = cluster.size / 3
        if num_pixels:
            means = np.append(means, [(cluster.sum(axis=0) / num_pixels)],
                              axis=0)
        else:
            means = np.append(means,
                              np.append(cluster, [z[i]], axis=0),
                              axis=0)
    return means


def fix_center(z, means):
    for i in range(k):
        z[i] = means[i]


def logger(z):
    log_body, i = '', 0

    def log(z, cmd):
        s = ''
        nonlocal log_body, i
        for _z in z:
            s += f'{_z},'
        log_body += f"[iter {i}]:{s[:-1]}\n"
        i += 1
        if cmd == FLUSH:
            _out = open(argv[3], "w")
            _out.write(log_body)
            _out.close()
            log_body = ''

    return log


def plot(losses):
    plt.xlabel('Iteration')
    plt.ylabel('Avg. loss')
    plt.xticks(list(range(len(losses)))[0::2])
    plt.plot(list(range(len(losses))), losses)
    plt.show()


orig_pixels = plt.imread(argv[1])
pixels = (orig_pixels.astype(float) / 255).reshape(-1, 3)
n = pixels.size / 3  # Num of pixels
z = np.loadtxt(argv[2]).round(4)  # Centeroids
k = len(z)  # Num of centroids
logger = logger(z)
print('Learning, please wait . . .')
for _ in range(19):
    clusters = cluster_points(pixels, z)
    means = get_means(clusters, z).round(4)
    if not np.array_equal(z, means):
        fix_center(z, means)
    else:
        break
    logger(z, APPEND)
logger(z, FLUSH)
print('Learning done')
plot(losses)
