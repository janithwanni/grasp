#---- VECTOR DIMENSIONALITY -----------------------------------------------------------------------
# Dimensionality reduction of vectors (i.e., fewer features) can speed up distance calculations.
# This can be achieved by grouping related features (e.g., covariance) or by selecting features.

def rp(vectors=[], n=100, distribution=(-1, +1)):
    """ Returns a list of vectors, each with n features.
        (Random Projection)
    """
    # Given a m x d matrix (m vectors, d features),
    # build a d x n matrix from a Gaussian distribution.
    # The dot product is the reduced vector space m x n.
    h = features(vectors)
    h = enumerate(h)
    h = map(reversed, h)
    h = dict(h) # {feature: int} hash
    r = [[random.choice(distribution) for i in range(n)] for f in h]
    p = []
    for v in vectors:
        p.append([])
        for i in range(n):
            x = 0
            for f, w in v.items():
                x += w * r[h[f]][i] # dot product
            p[-1].append(x)
    p = map(enumerate, p)
    p = map(dict, p)
    p = list(p)
    return p

def matrix(vectors=[]):
    """ Returns a 2D numpy.ndarray of the given vectors, 
        with columns ordered by sorted(features(vectors).
    """
    import numpy

    f = features(vectors)
    f = sorted(f)
    f = enumerate(f)
    f = {v: i for i, v in f}
    m = numpy.zeros((len(vectors), len(f)))
    for v, a in zip(vectors, m):
        a.put(map(f.__getitem__, v), v.values())
    return m

class svd(list):
    """ Returns a list of vectors, each with n concepts,
        where each concept is a combination of features.
        (Singular Value Decomposition)
    """
    def __init__(self, vectors=[], n=2):
        import numpy

        f  = dict(enumerate(sorted(features(vectors))))
        m  = matrix(vectors)
        m -= numpy.mean(m, 0)
        # u:  vectors x concepts
        # v: concepts x features
        u, s, v = numpy.linalg.svd(m, full_matrices=False)

        self.extend(
            sparse({   i  : w * abs(w) for i, w in enumerate(a) }) for a in u[:,:n]
        )
        self.concepts = tuple(
            sparse({ f[i] : w * abs(w) for i, w in enumerate(a) }) for a in v[:n]
        )
        self.features = normalize(
            sparse({ f[i] : 1 * abs(w) for i, w in enumerate(numpy.dot(s[:n], v[:n]))
        }))

    @property
    def cs(self):
        return self.concepts

    @property
    def pc(self):
        return self.features

pca = svd # (Principal Component Analysis)

# data = [
#     {'x': 0.0, 'y': 1.1, 'z': 1.0},
#     {'x': 0.0, 'y': 1.0, 'z': 0.0}
# ]
# 
# print(svd(data, n=2))
# print(svd(data, n=2).cs[0])
# print(svd(data, n=1).pc)