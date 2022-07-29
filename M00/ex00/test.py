from matrix import Matrix
from matrix import Vector

print("My tests\n------")

matrix = Matrix((3, 2))
print(matrix.__str__())

m1 = Matrix([[0.0, 1.0], [2.0, 3.0], [4.0, 5.0]])
print(m1.__str__())

m2 = Matrix([[1.0, 1.0], [1.0, 1.0], [1.0, 1.0]])
print(m2.__str__())

m3 = m1 - m2
print(m3)

m4 = m1 * Matrix([[2.0], [2.0]])
print(m4)

m5 = Matrix([[1.0, 1.0, 1.0], [1.0, 1.0, 1.0]])

m6 = m1 * m5
print(m6)

m7 = matrix.T()
print(m7)

v1 = Vector([[1, 2, 3]])
print(v1)

v2 = Vector([[1], [2], [3]])
print (v2)

v3 = Vector([[1, 2], [3, 4]])

v4 = v1.dot(Vector([[1, 2, 3]]))
print(v4)

print("\nSubject test\n--------------")

m1 = Matrix([[0.0, 1.0], [2.0, 3.0], [4.0, 5.0]])
print(m1.shape)
print(m1.T())
print(m1.T().shape, end="\n\n")

m1 = Matrix([[0., 2., 4.], [1., 3., 5.]])
print(m1.shape)
print(m1.T())
print(m1.T().shape, end="\n\n")

m1 = Matrix([[0.0, 1.0, 2.0, 3.0],
[0.0, 2.0, 4.0, 6.0]])
m2 = Matrix([[0.0, 1.0],
[2.0, 3.0],
[4.0, 5.0],
[6.0, 7.0]])

print(m1 * m2, end="\n\n")

m1 = Matrix([[0.0, 1.0, 2.0],
[0.0, 2.0, 4.0]])
v1 = Vector([[1], [2], [3]])
print(m1 * v1, end="\n\n")

print("\nVector test\n-------------")
v1 = Vector([[1], [2], [3]])
v2 = Vector([[2], [4], [8]])
v3 = Vector([[1, 2, 3]])
print(v1)
print(v2)
print(v3)
print(v1 + v2)
print(v1 - v2)
print(v1 * v2)
print(v1 * v3)
print(v1 / v2)
print(1 * v1)
print(v1 / 2)
print(2 / v1)
m1 = Matrix([[1, 2, 3]])
print(v1 * m1)
print(m1 + v3)
print(v3 + m1)
