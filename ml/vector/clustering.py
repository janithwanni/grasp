#---- VECTOR CLUSTERING ---------------------------------------------------------------------------
# The k-means clustering algorithm is an unsupervised machine learning method
# that partitions a given set of vectors into k clusters, so that each vector
# belongs to the cluster with the nearest center (mean).

euclidean = distance
spherical = cos

def ss(vectors=[], distance=euclidean):
    """ Returns the sum of squared distances to the center (variance).
    """
    v = list(vectors)
    c = centroid(v)
    return sum(distance(v, c) ** 2 for v in v)

def kmeans(vectors=[], k=3, distance=euclidean, iterations=100, n=10):
    """ Returns a list of k lists of vectors, clustered by distance.
    """
    vectors = list(vectors)
    optimum = None

    for _ in range(max(n, 1)):

        # Random initialization:
        g = list(shuffled(vectors))
        g = list(g[i::k] for i in range(k))[:len(g)]

        # Lloyd's algorithm:
        for _ in range(iterations):
            m = [centroid(v) for v in g]
            e = []
            for m1, g1 in zip(m, g):
                for v in g1:
                    d1 = distance(v, m1)
                    d2, g2 = min((distance(v, m2), g2) for m2, g2 in zip(m, g))
                    if d2 < d1:
                        e.append((g1, g2, v)) # move to nearer centroid
            for g1, g2, v in e:
                g1.remove(v)
                g2.append(v)
            if not e: # converged?
                break

        # Optimal solution = lowest within-cluster sum of squares:
        optimum = min(optimum or g, g, key=lambda g: sum(ss(g, distance) for g in g))
    return optimum

# data = [
#     {'woof': 1},
#     {'woof': 1},
#     {'meow': 1}
# ]
# 
# for cluster in kmeans(data, k=2):
#     print(cluster) # cats vs dogs