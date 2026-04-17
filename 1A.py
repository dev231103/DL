import tensorflow as tf
print("Matrix Multiplication Demo")
x = tf.constant([1, 2, 10, 1, 2, 6], shape=[2, 3])
print("Matrix X:\n", x.numpy())
y = tf.constant([5, 8, 12, 10, 1, 12], shape=[3, 2])
print("Matrix Y:\n", y.numpy())
z = tf.matmul(x, y)
print("Product:\n", z.numpy())
e_matrix_A = tf.random.uniform([2, 2], minval=3, maxval=10, dtype=tf.float32, name="matrixA")
e_matrix_A = (e_matrix_A + tf.transpose(e_matrix_A)) / 2
print("Matrix A:\n", e_matrix_A.numpy())
eigen_values_A, eigen_vectors_A = tf.linalg.eigh(e_matrix_A)
print("Eigen Values:\n", eigen_values_A.numpy())
print("Eigen Vectors:\n", eigen_vectors_A.numpy())