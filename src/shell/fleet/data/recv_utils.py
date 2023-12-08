from shell.fleet.data.data_utilize import *
import logging
import pickle

# HACK
NUM_CLASSES = {
    "mnist": 10,
    "fashionmnist": 10,
    "kmnist": 10,
    "cifar100": 100,
}


def acc(conf_mat):
    return np.diag(conf_mat).sum() / conf_mat.sum()


def add_mats(mats):
    # mats is a list of matrices
    # return the sum of all matrices
    return sum(mats)


def compute_image_search_quality(receiver_node, neighbor, neighbor_id, task_id, viz=False):
    """
    task_id: the target task that the receiver_node is requesting data from
    Return a num_classes x num_classes matrix where each entry[i, j]
    is the number of query images of class i that are given neighbors of class j.

    Return confusion_matrix, id_confusion_matrix
    id_confusion_matrix eliminate those requests that are not contained in the neighbor's current
    task pool (note that it's the responsibility of the neighbor to do OOD ect with the one it has still)
    """
    # Y_query = receiver_node.query_y[task_id]
    # # Y_query_global.shape=(num_queries) where each entry is the global label of the query
    # # range from 0 to num_classes
    # Y_query_global = get_global_labels(Y_query, [
    #                                    task_id] * len(Y_query), receiver_node.dataset.class_sequence, receiver_node.dataset.num_classes_per_task)
    Y_query_global = receiver_node.query_extra_info['query_global_y'][task_id]

    task_neighbors_prefilter = None
    Y_neighbor = receiver_node.incoming_extra_info[neighbor_id]['Y_neighbors'][task_id]

    if 'task_neighbors_prefilter' in receiver_node.incoming_extra_info[neighbor_id]:
        task_neighbors_prefilter = receiver_node.incoming_extra_info[
            neighbor_id]['task_neighbors_prefilter'][task_id]
        logging.debug("task_neighbors_prefilter %s", task_neighbors_prefilter)
    Y_neighbor_flatten = Y_neighbor.view(-1)

    qX = receiver_node.query[task_id]
    oracle_neighbors_prefilter = neighbor.prefilter_oracle_helper(
        qX, Y_query_global, n_filter_neighbors=neighbor.sharing_strategy.num_filter_neighbors)

    task_neighbor = receiver_node.incoming_extra_info[neighbor_id]['task_neighbors'][task_id]
    task_neighbor_flatten = task_neighbor.view(-1)

    logging.debug("Y_neighbor: %s", Y_neighbor)
    logging.debug("task_neighbor: %s", task_neighbor)
    Y_neighbor_global = get_global_labels(Y_neighbor_flatten, task_neighbor_flatten, neighbor.dataset.class_sequence,
                                          neighbor.dataset.num_classes_per_task).reshape(Y_neighbor.shape)
    logging.debug("Y_query_global %s", Y_query_global)
    logging.debug("Y_neighbor_global %s", Y_neighbor_global)
    # print(Y_query.shape, Y_neighbor.shape, task_neighbor.shape, Y_neighbor_global.shape)
    num_classes = NUM_CLASSES[receiver_node.dataset.name]
    confusion_matrix = np.zeros((num_classes, num_classes))
    id_confusion_matrix = np.zeros((num_classes, num_classes))
    for i in range(len(Y_query_global)):
        # Assuming Y_neighbor_global is a 2D array
        for j in range(Y_neighbor_global.shape[1]):
            if task_neighbors_prefilter is not None and task_neighbors_prefilter[i, j] == -1:
                continue
            query_label = Y_query_global[i]
            neighbor_label = Y_neighbor_global[i, j]
            confusion_matrix[query_label, neighbor_label] += 1

    for i in range(len(Y_query_global)):
        # Assuming Y_neighbor_global is a 2D array
        for j in range(Y_neighbor_global.shape[1]):
            if oracle_neighbors_prefilter[i, j] == -1:
                continue
            query_label = Y_query_global[i]
            neighbor_label = Y_neighbor_global[i, j]
            id_confusion_matrix[query_label, neighbor_label] += 1

    X_neighbor = receiver_node.incoming_data[neighbor_id][task_id]
    # if viz:
    #     viz_neighbor_data(
    #         X_neighbor, path=f"{receiver_node.save_dir}/task_{task_id}/viz/from_{neighbor_id}_prefilter_{prefilter_strategy}.pdf")

    # torch.save(
    #     X_neighbor, f"{receiver_node.save_dir}/task_{task_id}/X_neighbor_from_{neighbor_id}_prefilter_{prefilter_strategy}.pt")
    return confusion_matrix, id_confusion_matrix


def compute_recv_fleet_quality(fleet, task_id):
    for receiver in fleet.agents:
        if receiver.agent.T < receiver.agent.net.num_init_tasks:
            continue
        assert task_id == receiver.agent.T-1
        confs, id_confs = [], []

        for t in range(receiver.agent.T):
            for neighbor in receiver.neighbors:
                neighbor_id = neighbor.node_id
                conf, id_conf = compute_image_search_quality(
                    receiver, neighbor, neighbor_id, t)
                confs.append(conf)
                id_confs.append(id_conf)
        scorer = receiver.sharing_strategy.scorer

        with open(f"{receiver.save_dir}/task_{task_id}/conf_{receiver.sharing_strategy['prefilter_strategy']}_{scorer}.pkl", 'wb') as f:
            pickle.dump(confs, f)
        with open(f"{receiver.save_dir}/task_{task_id}/id_conf_{receiver.sharing_strategy['prefilter_strategy']}_{scorer}.pkl", 'wb') as f:
            pickle.dump(id_confs, f)


def load_recv_fleet_quality(fleet, task_id):
    ret_confs, ret_id_confs = [], []
    for receiver in fleet.agents:
        if receiver.agent.T < receiver.agent.net.num_init_tasks:
            continue
        assert task_id == receiver.agent.T-1
        scorer = receiver.sharing_strategy.scorer
        with open(f"{receiver.save_dir}/task_{task_id}/conf_{receiver.sharing_strategy['prefilter_strategy']}_{scorer}.pkl", 'rb') as f:
            confs = pickle.load(f)
        with open(f"{receiver.save_dir}/task_{task_id}/id_conf_{receiver.sharing_strategy['prefilter_strategy']}_{scorer}.pkl", 'rb') as f:
            id_confs = pickle.load(f)
        ret_confs += confs
        ret_id_confs += id_confs
    return ret_confs, ret_id_confs
