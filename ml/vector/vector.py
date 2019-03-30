#---- VECTOR --------------------------------------------------------------------------------------
# A vector is a {feature: weight} dict, with n features, or n dimensions.

# Given two points {x: 1, y: 2} and {x: 3, y: 4} in 2D,
# their distance is: sqrt((3 - 1) ** 2 + (4 - 2) ** 2).
# Distance can be calculated for points in 3D or in nD.

# Another distance metric is the angle between vectors (cosine).
# Another distance metric is the difference between vectors.
# For vectorized text cos() works well but diff() is faster.

# Vector weights are assumed to be non-negative, especially 
# when using cos(), diff(), knn(), tf(), tfidf() and freq().

def index(data=[]):
    """ Returns a dict of (id(vector), label)-items
        for the given list of (vector, label)-tuples.
    """
    return {id(v): label for v, label in data}

def distance(v1, v2):
    """ Returns the distance of the given vectors.
    """
    return sum((v1.get(f, 0) - v2.get(f, 0)) ** 2 for f in features((v1, v2))) ** 0.5

def dot(v1, v2):
    """ Returns the dot product of the given vectors.
    """
    return sum(v1.get(f, 0) * w for f, w in v2.items())

def norm(v):
    """ Returns the norm of the given vector.
    """
    return sum(w ** 2 for f, w in v.items()) ** 0.5

def cos(v1, v2):
    """ Returns the angle of the given vectors (0.0-1.0).
    """
    return 1 - dot(v1, v2) / (norm(v1) * norm(v2) or 1.0) # cosine distance

def diff(v1, v2):
    """ Returns the difference of the given vectors.
    """
    v1 = set(filter(v1.get, v1)) # non-zero
    v2 = set(filter(v2.get, v2))
    return 1 - len(v1 & v2) / float(len(v1 | v2) or 1)

def knn(v, vectors=[], k=3, distance=cos):
    """ Returns the k nearest neighbors from the given list of vectors.
    """
    nn = ((distance(v, x), random.random(), x) for x in vectors)
    nn = heapq.nsmallest(k, nn)
  # nn = sorted(nn)[:k]
    nn = [(1 - d, x) for d, r, x in nn]
    return nn

def reduce(v, features=set()):
    """ Returns a vector without the given features.
    """
    return {f: w for f, w in v.items() if f not in features}

def sparse(v, cutoff=0.00001):
    """ Returns a vector with non-zero weight features.
    """
    return {f: w for f, w in v.items() if w > cutoff}

def binary(v, cutoff=0.0):
    """ Returns a vector with binary weights (0 or 1).
    """
    return {f: int(w > cutoff) for f, w in v.items()}

def onehot(v): # {'age': '25+'} => {('age', '25+'): 1}
    """ Returns a vector with non-categorical features.
    """
    return dict((f, w) if isinstance(w, (int, float)) else ((f, w), 1) for f, w in v.items())

def scale(v, x=0.0, y=1.0):
    """ Returns a vector with normalized weights (between x and y).
    """
    a = min(v.values())
    b = max(v.values())
    return {f: float(w - a) / (b - a) * (y - x) + x for f, w in v.items()}

def unit(v):
    """ Returns a vector with normalized weights (length 1).
    """
    n = norm(v) or 1.0
    return {f: w / n for f, w in v.items()}

def normalize(v):
    """ Returns a vector with normalized weights (sum to 1).
    """
    return tf(v)

def tf(v):
    """ Returns a vector with normalized weights
        (term frequency).
    """
    n = sum(v.values())
    n = float(n or 1)
    return {f: w / n for f, w in v.items()}

def tfidf(vectors=[]):
    """ Returns an iterator of vectors with normalized weights
        (term frequency–inverse document frequency).
    """
    df = collections.Counter() # stopwords have higher df (I, the, or, ...)
    if not isinstance(vectors, list):
        vectors = list(vectors)
    for v in vectors:
        df.update(v)
    for v in vectors:
        yield {f: w / float(df[f] or 1) for f, w in v.items()}

def features(vectors=[]):
    """ Returns the set of features for all vectors.
    """
    return set().union(*vectors)

def centroid(vectors=[]):
    """ Returns the mean vector for all vectors.
    """
    v = list(vectors)
    n = float(len(v))
    return {f: sum(v.get(f, 0) for v in v) / n for f in features(v)}

def freq(a):
    """ Returns the relative frequency distribution of items in the list.
    """
    f = collections.Counter(a)
    f = collections.Counter(normalize(f))
    return f

def majority(a, default=None):
    """ Returns the most frequent item in the given list (majority vote).
    """
    f = collections.Counter(a)
    try:
        m = max(f.values())
        return random.choice([k for k, v in f.items() if v == m])
    except:
        return default

# print(majority(['cat', 'cat', 'dog']))
