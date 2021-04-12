# coding=utf-8

import tf_geometric as tfg
import tensorflow as tf



class MultiVariationalGCN(tf.keras.Model):

    def __init__(self, num_units_list, output_list=False, uncertain=True, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.uncertain = uncertain
        self.output_list = output_list
        self.gcns = [
            tfg.layers.GCN(num_units, activation=tf.nn.relu)
            for i, num_units in enumerate(num_units_list[:-1])
        ]

        self.mu_gcn = tfg.layers.GCN(num_units_list[-1])
        if self.uncertain:
            self.std_gcn = tfg.layers.GCN(num_units_list[-1])


    def call(self, inputs, training=None, mask=None, cache=None):
        h, edge_index, edge_weight = inputs

        outputs = []

        for i, gcn in enumerate(self.gcns):
            h = gcn([h, edge_index, edge_weight], cache=cache)
            outputs.append(h)

        mu = self.mu_gcn([h, edge_index, edge_weight], cache=cache)
        if self.uncertain:
            log_std = self.std_gcn([h, edge_index, edge_weight], cache=cache)
            std = tf.math.exp(log_std)
            rand = tf.random.truncated_normal(mu.shape)
        # if training and False:
        if self.uncertain and training:
            h = mu + rand * std
            kl = -log_std + 0.5 * (std ** 2 + mu ** 2 - 1)
        else:
            h = mu
            kl = mu * 0.0
        outputs.append(h)


        if self.output_list:
            return outputs, kl
        else:
            return h, kl



class MultiVariationalGCNWithDense(MultiVariationalGCN):
    def __init__(self, num_units_list, output_list=False, uncertain=True, *args, **kwargs):
        super().__init__(num_units_list[:-1], output_list=output_list, uncertain=uncertain)
        self.dense = tf.keras.layers.Dense(num_units_list[-1])

    def call(self, inputs, training=None, mask=None, cache=None):
        super_results, kl = super().call(inputs, training=training, mask=mask, cache=cache)
        if self.output_list:
            outputs = super_results
            h = outputs[-1]
        else:
            h = super_results

        # h = tf.nn.relu(h)
        h = self.dense(h)

        if self.output_list:
            outputs.append(h)
            return outputs, kl
        else:
            return h, kl

