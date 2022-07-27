import copy

class Matrix():

    def __init__(self, data):
        if isinstance(data, tuple) and len(data) == 2:
            return self.__init_from_tuple__(data)
        elif isinstance(data, list):
            return self.__init_from_list__(data)
        print("Data given to matrix class is not compatible")
        return None

    def __init_from_list__(self, data):
        len_row = len(data[0])
        for row in data:
            if not isinstance(row, list) or len(row) != len_row:
                print("Data given to matrix class is not compatible")
                return None
            for elem in row:
                if not isinstance(elem, float):
                    print("Data given to matrix class is not compatible")
                    return None
        self.data = data
        self.shape = (len(data), len_row)
        return

    def __init_from_tuple__(self, data):
        matrix = []
        for row in range(data[0]):
            row = []
            for column in range(data[1]):
                row.append(0.0)
            matrix.append(row)
        self.data = matrix
        self.shape = data
        return

    def __add__(self, to_add):
        if not isinstance(to_add, Matrix) or self.shape != to_add.shape :
            print("Can't add matrix of differents dimensions")
            return None
        matrix = copy.deepcopy(self.data)
        for row in range(self.shape[0]):
            for column in range(self.shape[1]):
                matrix[row][column] += to_add.data[row][column]
        return matrix

    __radd__ = __add__

    def __sub__(self, to_sub):
        if not isinstance(to_sub, Matrix) or self.shape != to_sub.shape :
            print("Can't sub matrix of differents dimensions")
            return None
        matrix = copy.deepcopy(self.data)
        for row in range(self.shape[0]):
            for column in range(self.shape[1]):
                matrix[row][column] -= to_sub.data[row][column]
        return matrix

    __rsub__= __sub__

    def __truediv__(self, div):
        if not isinstance(div, int) and not isinstance(div, float):
            print("Can't div matrix by anothing other than scalar")
            return None
        matrix = copy.deepcopy(self.data)
        for row in range(self.shape[0]):
            for column in range(self.shape[1]):
                matrix[row][column] /= div
        return matrix

    def __rtruediv__(self, div):
        print("Can't divide by matrix")
        return None

    def __mul__(self, mul):
        if isinstance(mul, int) or isinstance(mul, float):
            matrix = copy.deepcopy(self.data)
            for row in range(self.shape[0]):
                for column in range(self.shape[1]):
                    matrix[row][column] *= mul
            return matrix
        elif isinstance(mul, Matrix) and mul.shape[1] == 1 and mul.shape[0] == self.shape[1]:
            return self.__mul_by_vector__(mul)
        elif isinstance(mul, Matrix) and self.shape[1] == mul.shape[0]:
            return self.__mul_by_matrix__(mul)
        print("Can't make this multiplication with current matrix")
        return None

    def __mul_by_vector__(self, mul):
        matrix = []
        for row in range(self.shape[0]):
            value = 0
            for column in range(self.shape[1]):
                value += self.data[row][column] * mul.data[column][0]
            sub_matrix =[value]
            matrix.append(sub_matrix)
        return matrix

    def __mul_by_matrix__(self, mul):
        matrix = []
        for row in range(self.shape[0]):
            sub_matrix = []
            for vect in range(mul.shape[1]):
                value = 0
                for column in range(self.shape[1]):
                    value += self.data[row][column] * mul.data[column][vect]
                sub_matrix.append(value)
            matrix.append(sub_matrix)
        return matrix

    def __str__(self):
        if self.data and self.shape:
            return "Matrix {} of shape {}".format(self.data, self.shape)
        return None

    def __repr__(self):
        return self

    def T(self):
        matrix = []
        for column in range(self.shape[1]):
            sub_matrix = []
            for elem in range(self.shape[0]):
                sub_matrix.append(self.data[elem][column])
            matrix.append(sub_matrix)
        return matrix


class Vector(Matrix):

    def __init__(self, data):
        self.data = []
        if isinstance(data, list) and isinstance(data[0], list):
            return self.__init_vect__(data)
        print("Data given is incompatible with vector class")
        return None

    def __init_vect__(self, data):
        if len(data) != 1 and len(data[0]) != 1:
                print("Data given is incompatible with vector class")
                return None
        for elem in data:
            if not isinstance(elem, list) or (not isinstance(elem[0], float) and not isinstance(elem[0], int)):
                print("Data given is incompatible with vector class")
                return None
            if len(elem) != len(data[0]):
                print("Data given is incompatible with vector class")
                return None
            for sub_elem in elem:
                if not isinstance(sub_elem, type(elem[0])):
                    print("Data given is incompatible with vector class")
                    return None
        self.data = data
        self.shape = (len(self.data), len(self.data[0]))

    def __str__(self):
        if self.data and self.shape:
            return "Vector {} of shape {}".format(self.data, self.shape)
        return None

    def dot(self, v):
        if not isinstance(v, Vector) or len(self.data) != len(v.data) \
            or type(self.data[0]) != type(v.data[0]):
            print("Dot product can only be made from two vectors of same dimensions.")
            return None
        res = 0
        if isinstance(self.data[0], float):
            for elem, oth_elem in zip(self.data, v.data):
                res += elem * oth_elem
        else:
            for elem, elem_ot in zip(self.data, v.data):
                for sub_elem, sub_elem_ot in zip(elem, elem_ot):
                    res += sub_elem * sub_elem_ot
        return res
