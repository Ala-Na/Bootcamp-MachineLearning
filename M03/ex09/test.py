import numpy as np
from sklearn.metrics import confusion_matrix
from confusion_matrix import confusion_matrix_

y_hat = np.array([['norminet'], ['dog'], ['norminet'], ['norminet'], ['dog'], ['bird']])
y = np.array([['dog'], ['dog'], ['norminet'], ['norminet'], ['dog'], ['norminet']])

# Example 1:
## your implementation
print(confusion_matrix_(y, y_hat))
## Output:
#array([[0 0 0]
#[0 2 1]
#[1 0 2]])
## sklearn implementation
print(confusion_matrix(y, y_hat))
## Output:
#array([[0 0 0]
#[0 2 1]
#[1 0 2]])

print('\n')

# Example 2:
## your implementation
print(confusion_matrix_(y, y_hat, labels=['dog', 'norminet']))
## Output:
#array([[2 1]
#[0 2]])
## sklearn implementation
print(confusion_matrix(y, y_hat, labels=['dog', 'norminet']))
## Output:
#array([[2 1]
#[0 2]])

print('\n')

#Example 3:
print(confusion_matrix_(y, y_hat, df_option=True))
#Output:
#bird dog norminet
#bird 0 0 0
#dog 0 2 1
#norminet 1 0 2

print('\n')

#Example 4:
print(confusion_matrix_(y, y_hat, labels=['bird', 'dog'], df_option=True))
#Output:
#bird dog
#bird 0 0
#dog 0 2

print('\n')

print(confusion_matrix_(y, y_hat, labels=['truc'], df_option=True))

print("\nCORRECTION:")

y_true=np.array(['a', 'b', 'c'])
y_hat=np.array(['a', 'b', 'c'])
print(f"{confusion_matrix_(y_true, y_hat) = }")
print("should return a numpy.array or pandas.DataFrame full of zeros except the diagonal which should be full of ones.")
print()

y_true=np.array(['a', 'b', 'c'])
y_hat=np.array(['c', 'a', 'b'])
# print(f"{confusion_matrix_(y_true, y_hat) = }")
# Should be previous but show result of following:
print(f"{confusion_matrix_(y_hat, y_true) = }")
print('should return "np.array([[0, 1, 0], [0, 0, 1], [1, 0, 0]])"')
print(f"sklearn : {confusion_matrix(y_hat, y_true) = }")
print()

y_true=np.array(['a', 'a', 'a'])
y_hat=np.array(['a', 'a', 'a'])
print(f"{confusion_matrix_(y_true, y_hat) = }")
print("should return np.array([3])")
print()

y_true=np.array(['a', 'a', 'a'])
y_hat=np.array(['a', 'a', 'a'])
print(f"{confusion_matrix_(y_true, y_hat, labels=[]) = }")
print("return None, an empty np.array or an empty pandas.Dataframe.")
print()
