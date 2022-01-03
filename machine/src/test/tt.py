import numpy as np

np.set_printoptions(precision=3)
x = np.random.random((5, 5))
x = np.array([[2,3],[4,1]])
inv = np.linalg.inv(x)
print(inv)
print(np.dot(x,inv))

# ans = np.random.random(1000)
# print(np.linalg.matrix_rank(x))
# t = np.linalg.solve(x, ans)
# print(ans)
# predict = np.dot(x, t)
# print(predict)
#
# print(np.abs(predict-ans)<0.0001)

