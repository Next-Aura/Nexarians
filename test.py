from nexgml.guardians import hasinf, hasnan
from numpy import nan, inf, array

test = array([nan, inf, 2, 4, 9, 6])

print(hasinf(test))