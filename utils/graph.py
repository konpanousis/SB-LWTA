"""

Functions to build the graph for the CIFAR10 dataset and in the case of multigpu implementation.

@author: Currenlty anonymous
"""

import tensorflow as tf


def average_gradients(tower_grads):
    average_grads = []
    for grad_and_vars in zip(*tower_grads):
        grads = []
        for g, _ in grad_and_vars:
            # Add 0 dimension to the gradients to represent the tower.
            expanded_g = tf.expand_dims(g, 0)
            # Append on a 'tower' dimension which we will average over below.
            grads.append(expanded_g)

        # Average over the 'tower' dimension.
        grad = tf.concat(grads, 0)
        grad = tf.reduce_mean(grad, 0)

        # Keep in mind that the Variables are redundant because they are shared
        # across towers. So .. we will just return the first tower's pointer to
        # the Variable.
        v = grad_and_vars[0][1]
        grad_and_var = (grad, v)
        average_grads.append(grad_and_var)
    return average_grads



def build_graph(images_train, labels_train, images_test, labels_test, global_step, loss_function, accuracy_function,
                inference_function, learning_rate, devices):
    
    optimizer_net = tf.train.AdamOptimizer(learning_rate=learning_rate, name='adam')

    tf.summary.scalar('learning_rate', learning_rate)
    
    train_queue = tf.contrib.slim.prefetch_queue.prefetch_queue([images_train, labels_train],
                                                                capacity=3 * len(devices), num_threads=len(devices))
    test_queue = tf.contrib.slim.prefetch_queue.prefetch_queue([images_test, labels_test],
                                                               capacity=3 * len(devices), num_threads=len(devices))
    tower_grads_net = []
    train_loss_arr = []
    train_acc_arr = []
    test_loss_arr = []
    test_acc_arr = []
    with tf.variable_scope(tf.get_variable_scope()) as scope:
        for dev_id in range(len(devices)):
            with tf.device(devices[dev_id]):
                with tf.name_scope('tower_%s' % devices[dev_id][-1]) as scope:
                    # train ops
                    batch_images_train, batch_labels_train = train_queue.dequeue()
                    train_preds,_,_,_,_ = inference_function(batch_images_train, reuse=dev_id != 0, is_training=True)
                    train_loss = loss_function(train_preds, batch_labels_train)
                    train_loss_arr.append(train_loss)
                    train_acc_arr.append(accuracy_function(train_preds, batch_labels_train))
                    
                    variables = list(filter(lambda v: 'optimizer' not in v.name.lower(), tf.trainable_variables()))
                    net_variables = list(filter(lambda v: 'sb' not in v.name.lower(), variables))

                    grads_net = optimizer_net.compute_gradients(train_loss, tf.trainable_variables())
                    tower_grads_net.append(grads_net)
                    tf.get_variable_scope().reuse_variables()

                    # test ops
                    batch_images_test, batch_labels_test = test_queue.dequeue()
                    test_preds, mws, weights, variances, masks = inference_function(batch_images_test, reuse=True, is_training=False)
                    test_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=test_preds,
                                                                               labels=batch_labels_test)
                    test_loss = tf.reduce_mean(test_loss)
                    test_loss_arr.append(test_loss)
                    test_acc_arr.append(accuracy_function(test_preds, batch_labels_test))
                    tf.get_variable_scope().reuse_variables()

    grads_net = average_gradients(tower_grads_net)
    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    train_loss_op = tf.add_n(train_loss_arr)/len(devices)
    tf.summary.scalar('train_loss', train_loss_op)
    train_acc_op = tf.add_n(train_acc_arr)/len(devices)
    tf.summary.scalar('train_accuracy', train_acc_op)
    test_loss_op = tf.add_n(test_loss_arr)/len(devices)
    test_acc_op = tf.add_n(test_acc_arr)/len(devices)
    
    with tf.control_dependencies(update_ops):
        train_op = optimizer_net.apply_gradients(grads_net, global_step=global_step)
        
    return train_op, test_acc_op, test_loss_op, net_variables, mws, weights, variances, masks

def build_graph_test_only(images_test, labels_test, global_step, loss_function, accuracy_function,
                inference_function, devices):
    
    test_queue = tf.contrib.slim.prefetch_queue.prefetch_queue([images_test, labels_test],
                                                               capacity=3 * len(devices), num_threads=len(devices))
    test_loss_arr = []
    test_acc_arr = []
    
    with tf.variable_scope(tf.get_variable_scope()) as scope:
        for dev_id in range(len(devices)):
            with tf.device(devices[dev_id]):
                with tf.name_scope('tower_%s' % devices[dev_id][-1]) as scope:
        
                    variables = list(filter(lambda v: 'optimizer' not in v.name.lower(), tf.trainable_variables()))
                    net_variables = list(filter(lambda v: 'sb' not in v.name.lower(), variables))
                    
                    # test ops
                    batch_images_test, batch_labels_test = test_queue.dequeue()
                    test_preds, mws,weights, variances, masks = inference_function(batch_images_test, reuse=False, is_training=False)
                    test_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=test_preds,
                                                                               labels=batch_labels_test)
                    test_loss = tf.reduce_mean(test_loss)
                    test_loss_arr.append(test_loss)
                    test_acc_arr.append(accuracy_function(test_preds, batch_labels_test))
                    tf.get_variable_scope().reuse_variables()

    test_loss_op = tf.add_n(test_loss_arr)/len(devices)
    test_acc_op = tf.add_n(test_acc_arr)/len(devices)
    return test_acc_op, test_loss_op, net_variables, mws, weights, variances, masks
