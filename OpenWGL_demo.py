# coding=utf-8
import tf_geometric as tfg
from sklearn.metrics import accuracy_score, f1_score, classification_report,confusion_matrix
from tf_geometric.utils.graph_utils import negative_sampling
from sklearn.model_selection import train_test_split

from opgl.model.module import  MultiVariationalGCNWithDense

import os
import tensorflow as tf
import numpy as np
import argparse

from opgl.utils.label_utils import reassign_labels, special_train_test_split


learning_rate = 1e-3
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
base_dir = "./"

base_data_dir = os.path.join(base_dir, "data")
parser = argparse.ArgumentParser()

parser.add_argument("--dataset_name", type=str, default="cora")
parser.add_argument("--unseen_num", type=int, default=1)

# command = '--dataset_name cora --unseen_num 1'
# sys.argv = sys.argv+command.split()

args = parser.parse_args()

dataset_name = args.dataset_name
unseen_num = args.unseen_num

training_rate = 0.7
valid_rate = 0.1
unseen_label_index = -1
filter_unseen = True
learning_rate = 1e-3
drop_rate = 0.3
train_seed = 100
np.random.seed(train_seed)
tf.random.set_random_seed(train_seed)


use_softmax = True
use_class_uncertainty = True
use_VGAE = True
uncertain_num_samplings = 100 if use_VGAE else 1


if dataset_name == "cora":
    graph, _ = tfg.datasets.CoraDataset(base_data_dir).load_data()

print(training_rate)
print(valid_rate)

original_num_classes = np.max(graph.y) + 1
seen_labels = list(range(original_num_classes - unseen_num))
y_true = reassign_labels(graph.y, seen_labels, unseen_label_index)
train_indices, test_valid_indices = special_train_test_split(y_true, unseen_label_index=-1, test_size=1-training_rate)
test_indices, valid_indices = train_test_split(test_valid_indices, test_size=valid_rate / (1-training_rate))
num_classes = np.max(y_true) + 1

print('data:{}\tseen_labels:{}\tuse_softmax:{}\trandom_seed:{}\tunseen_num:{}'.format(
    dataset_name,
    seen_labels,
    use_softmax,
    train_seed,
    unseen_num))


model = MultiVariationalGCNWithDense([32, 16, num_classes],
                            uncertain=use_VGAE,
                            output_list=True)

def logits_to_probs(logits):
    if use_softmax:
        probs = tf.nn.softmax(logits)
    else:
        probs = tf.nn.sigmoid(logits)
    return probs


def compute_loss(outputs, kl, mask_indices):
    # use negative_sampling
    logits = outputs[-1]
    h = outputs[-2]

    if use_VGAE:
        neg_edge_index = negative_sampling(
            num_samples=graph.num_edges,
            num_nodes=graph.num_nodes,
            edge_index=None,
            replace=False
        )

        pos_logits = tf.reduce_sum(
            tf.gather(h, graph.edge_index[0]) * tf.gather(h, graph.edge_index[1]),
            axis=-1
        )
        neg_logits = tf.reduce_sum(
            tf.gather(h, neg_edge_index[0])  * tf.gather(h, neg_edge_index[1]),
            axis=-1
        )

        pos_losses = tf.nn.sigmoid_cross_entropy_with_logits(
            logits=pos_logits,
            labels=tf.ones_like(pos_logits)
        )
        neg_losses = tf.nn.sigmoid_cross_entropy_with_logits(
            logits=neg_logits,
            labels=tf.zeros_like(neg_logits)
        )
        gae_loss = tf.reduce_mean(pos_losses) + tf.reduce_mean(neg_losses)


    all_indices = np.arange(0, tf.shape(logits)[0])
    unmasked_indices = np.delete(all_indices, mask_indices)

    unmasked_logits = tf.gather(logits, unmasked_indices)
    #
    loss_func = tf.nn.softmax_cross_entropy_with_logits if use_softmax else tf.nn.sigmoid_cross_entropy_with_logits

    unmasked_probs = logits_to_probs(unmasked_logits)
    unmasked_probs = tf.clip_by_value(unmasked_probs, 1e-7, 1.0)

    unmasked_preds = tf.argmax(unmasked_probs, axis=-1)
    unmasked_prob = tf.gather_nd(unmasked_probs, tf.stack([tf.range(unmasked_logits.shape[0], dtype=tf.int64), unmasked_preds], axis=1))

    topk_indices = tf.where(tf.logical_and(
        tf.greater(unmasked_prob, 1.0 / num_classes),
        tf.less(unmasked_prob, 0.5)
    ))

    unmasked_probs = tf.gather(unmasked_probs, topk_indices)
    class_uncertainty_losses = unmasked_probs * tf.math.log(unmasked_probs)

    masked_logits = tf.gather(logits, mask_indices)
    masked_y_true = y_true[mask_indices]
    losses = loss_func(
        logits=masked_logits,
        labels=tf.one_hot(masked_y_true, depth=num_classes)
    )
    masked_kl = tf.gather(kl, mask_indices)

    loss = tf.reduce_mean(losses)


    if use_class_uncertainty:
        loss += tf.reduce_mean(class_uncertainty_losses) * 1.0


    if use_VGAE:
        loss = loss + gae_loss * 1.0 + tf.reduce_mean(masked_kl)*1.0

    return  loss


def evaluate(logits, mask_indices, show_matrix=False, filter_unseen=True,threshold=None):

    if isinstance(logits, list):
        logits_list = tf.stack(logits, axis=-1)
        logits = tf.reduce_mean(logits_list, axis=-1)

        if use_softmax:
            probs_list = tf.nn.softmax(logits_list, axis=-2)
        else:
            probs_list = tf.nn.sigmoid(logits_list)
        probs = tf.reduce_mean(probs_list, axis=-1)
    else:
        probs = logits_to_probs(logits)

    masked_logits = tf.gather(logits, mask_indices)
    masked_y_pred = tf.argmax(masked_logits, axis=-1)
    masked_y_true = y_true[mask_indices]

    if filter_unseen:
        probs = tf.gather(probs, mask_indices)
        probs = tf.gather_nd(probs, tf.stack([tf.range(masked_logits.shape[0], dtype=tf.int64), masked_y_pred], axis=1))
        probs = probs.numpy()
        masked_y_pred = masked_y_pred.numpy()
        print("mean: ", probs.mean())
        if threshold is None:
            threshold = (probs[masked_y_true != unseen_label_index].mean()+probs[masked_y_true == unseen_label_index].mean())/2.0
            print("auto meanS: ", threshold)
        # threshold
        masked_y_pred[probs < threshold] = unseen_label_index

    else:
        masked_y_pred = masked_y_pred.numpy()


    accuracy = accuracy_score(masked_y_true, masked_y_pred)
    macro_f_score = f1_score(masked_y_true, masked_y_pred, average="macro")

    if show_matrix:
        print(classification_report(masked_y_true, masked_y_pred))
        print(confusion_matrix(masked_y_true, masked_y_pred))

    if filter_unseen:
        return accuracy, macro_f_score, threshold
    else:
        return accuracy, macro_f_score


optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)

results = []
for step in range(3000):

    with tf.GradientTape() as tape:
        outputs, kl = model([graph.x, graph.edge_index, graph.edge_weight], cache=graph.cache, training=True)
        logits = outputs[-1]
        train_loss = compute_loss(outputs, kl, train_indices)

    vars = tape.watched_variables()
    grads = tape.gradient(train_loss, vars)
    optimizer.apply_gradients(zip(grads, vars))

    if step % 10 == 0:

        train_logits = tf.gather(logits, train_indices)
        train_probs = logits_to_probs(train_logits)

        train_accuracy, _ = evaluate(logits, train_indices, filter_unseen=False)

        test_results = [model([graph.x, graph.edge_index, graph.edge_weight], cache=graph.cache, training=True)
                        for _ in range(uncertain_num_samplings)]
        outputs_list = [test_result[0] for test_result in test_results]
        outputs = [
            tf.reduce_mean(tf.stack([item[i] for item in outputs_list], axis=-1), axis=-1)
            for i in range(len(outputs_list[0]))
        ]
        kl = tf.add_n([test_result[1] for test_result in test_results]) / len(test_results)

        logits = outputs[-1]
        valid_loss = compute_loss(outputs, kl, valid_indices)
        valid_accuracy,valid_macro_f_score,  threshold = evaluate(logits, valid_indices, filter_unseen=True)

        print(threshold)
        print('=====')
        test_loss = compute_loss(outputs, kl, test_indices)
        test_accuracy, test_macro_f_score,  _ = evaluate(logits, test_indices,show_matrix=True, filter_unseen=True,threshold=threshold)



        print(f"step = {step}\n"
              f"\ttrain_loss = {train_loss}\ttrain_accuracy = {train_accuracy}\n"
              f"\tvalid_loss = {valid_loss}\tvalid_accuracy = {valid_accuracy}\tvalid_macro_f_score = {valid_macro_f_score}\n"
              f"\ttest_loss = {test_loss}\ttest_accuracy = {test_accuracy}\ttest_macro_f_score={test_macro_f_score}\n"
             )

        results.append([test_accuracy, test_macro_f_score])
        for i, metric_name in enumerate(["accuracy", "f"]):
            values = np.array(results)[:, i]
            max_step = np.argmax(values)
            print(f"max_{metric_name} = {values[max_step]}\tmax_step = {max_step}\t{results[max_step]}")
