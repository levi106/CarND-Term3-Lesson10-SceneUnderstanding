import tensorflow as tf

def mean_iou(ground_truth, prediction, num_classes):
  iou, iou_op = tf.metrics.mean_iou(ground_truth, prediction, num_classes)
  return iou, iou_op

ground_truth = tf.constant([
  [0,0,0,0],
  [1,1,1,1],
  [2,2,2,2],
  [3,3,3,3]], dtype=tf.float32)
prediction = tf.constant([
  [0,0,0,0],
  [1,0,0,1],
  [1,2,2,1],
  [3,3,0,3]], dtype=tf.float32)

iou, iou_op = mean_iou(ground_truth, prediction, 4)

with tf.Session() as sess:
  sess.run(tf.global_variables_initializer())
  sess.run(tf.local_variables_initializer())
  sess.run(iou_op)
  print("Mean IoU =", sess.run(iou))
