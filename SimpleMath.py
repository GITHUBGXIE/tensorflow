import tensorflow as tf

''' a = tf.constant(6.5, name='constant_a')
b = tf.constant(3.4, name='constant_b')

c = tf.constant(3.0, name='constant_c')
d = tf.constant(100.2, name='constant_d')

square = tf.square(a, name='square_a')
power = tf.pow(b,c, name='pow_b_c')
sqrt = tf.sqrt(d, name='sqrt_d')
final_sum = tf.add_n([square, power, sqrt], name='final_sum')
 '''

x=tf.placeholder(tf.int32, shape=[3], name='x')
y=tf.placeholder(tf.int32, shape=[3], name='y')
sum_x = tf.reduce_sum(x, name='sum_x')
prod_y = tf.reduce_prod(y, name='prod_y')

final_div = tf.div(sum_x, prod_y, name="final_div")
final_mean = tf.reduce_mean([sum_x, prod_y], name="final_mean")

sess = tf.Session()

''' print("square of a: ", sess.run(square))
print( "power of b ^ c: ", sess.run(power))
print( "square root d: ", sess.run(square))

print( "Sum of above : ", sess.run(final_sum)) '''

print("sum(x): ", sess.run(sum_x, feed_dict={x:[100, 200, 300]}))
print("prod(y): ", sess.run(prod_y, feed_dict={y:[1, 2, 3]}))
print( "sum(x)/prod(y): ", sess.run(final_div, feed_dict={x:[10,20,30], y:[1,2,3]}))

print( "final_mean : ",sess.run(final_mean, feed_dict={x:[1000,2000,3000], y:[10,20,30]}))

writer = tf.summary.FileWriter('./simpleMath', sess.graph)
writer.close()
sess.close()

