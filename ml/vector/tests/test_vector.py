v1 = {'x': 1, 'y': 2}
v2 = {'x': 3, 'y': 4}
v3 = {'x': 5, 'y': 6}
v4 = {'x': 7, 'y': 0}

assert round( distance(v1, v2)                , 2) ==  2.83
assert round(      dot(v1, v2)                , 2) == 11.00
assert round(     norm(v1)                    , 2) ==  2.24
assert round(      cos(v1, v2)                , 2) ==  0.02
assert round(      knn(v1, (v2, v3))[0][0]    , 2) ==  0.98
assert             knn(v1, (v2, v3))[0][1]         ==  v2
assert          sparse(v4)                         == {'x': 7}
assert              tf(v4)                         == {'x': 1, 'y': 0}
assert       features((v1, v2))                    == set(('x', 'y'))
assert next(    tfidf((v1, v2)))['x']              ==  0.25
assert       centroid((v1, v2))                    == {'x': 2, 'y': 3}