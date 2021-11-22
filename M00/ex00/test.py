from matrix import Matrix
from matrix import Vector

matrix = Matrix((3, 2))
print(matrix.__str__())

m1 = Matrix([[0.0, 1.0], [2.0, 3.0], [4.0, 5.0]])
print(m1.__str__())

m2 = Matrix([[1.0, 1.0], [1.0, 1.0], [1.0, 1.0]])
print(m2.__str__())

m3 = Matrix(m1 - m2)
print(m3)

m4 = Matrix(m1 * Matrix([[2.0], [2.0]]))
print(m4)

m5 = Matrix([[1.0, 1.0, 1.0], [1.0, 1.0, 1.0]])

m6 = Matrix(m1 * m5)
print(m6)

m7 = Matrix(matrix.T())
print(m7)

v1 = Vector([[1, 2, 3]])
print(v1)

v2 = Vector([[1], [2], [3]])
print (v2)

v3 = Vector([[1, 2], [3, 4]])

v4 = v1.dot(Vector([[1, 2, 3]]))
print(v4)