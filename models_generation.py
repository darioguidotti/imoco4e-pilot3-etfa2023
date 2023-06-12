import onnx
import pandas
import pynever.nodes as pyn_nodes
import pynever.networks as pyn_networks
import pynever.strategies.conversion as pyn_conv
import pynever.strategies.training as pyn_train
import datasets
import os
import logging
import torch.utils.data as pyt_data
import torch.optim as pyt_optim
import torch.nn as nn
import numpy as np

import utilities


def main():

    # ===== SET EXPERIMENT ID AND FOLDERS CREATION =====
    #
    #
    #
    experiment_id = "001"
    experiment_folder = f"exp_{experiment_id}/"
    onnx_folder = experiment_folder + "onnx_models/"
    logs_folder = experiment_folder + "logs/"
    checkpoint_folder = experiment_folder + "training_checkpoints/"
    models_data_path = experiment_folder + "models_data/"
    graphs_folder = experiment_folder + "graphs/"

    if not os.path.exists(experiment_folder):
        os.mkdir(experiment_folder)

    if not os.path.exists(onnx_folder):
        os.mkdir(onnx_folder)

    if not os.path.exists(logs_folder):
        os.mkdir(logs_folder)

    if not os.path.exists(checkpoint_folder):
        os.mkdir(checkpoint_folder)

    if not os.path.exists(models_data_path):
        os.mkdir(models_data_path)

    if not os.path.exists(graphs_folder):
        os.mkdir(graphs_folder)

    # ===== LOGGERS INSTANTIATION =====
    #
    #
    #
    logger_stream = logging.getLogger("pynever.strategies.training")
    logger_file = logging.getLogger("models_generation_file")

    file_handler = logging.FileHandler(f"{logs_folder}models_gen_logs.csv")
    stream_handler = logging.StreamHandler()

    file_handler.setLevel(logging.INFO)
    stream_handler.setLevel(logging.INFO)

    logger_file.addHandler(file_handler)
    logger_stream.addHandler(stream_handler)

    logger_file.setLevel(logging.INFO)
    logger_stream.setLevel(logging.INFO)

    # ===== PARAMETERS SELECTION =====
    #
    #
    #
    dataset_path = "data/norm_year_data.csv"
    dataset_id = "CDAD"
    activation_functions = [pyn_nodes.ReLUNode]
    network_arch = [[32, 8, 32], [50, 10, 50], [64, 16, 64], [128, 32, 128]]
    test_percentage = 0.2

    optimizer_con = pyt_optim.Adam
    opt_params = {"lr": 0.001}
    loss_function = nn.MSELoss()
    n_epochs = 50
    validation_percentage = 0.3
    train_batch_size = 512
    validation_batch_size = 128
    precision_metric = nn.MSELoss()
    device = "mps"
    checkpoints_root = checkpoint_folder
    metric_params = {}
    test_batch_size = 128

    # ===== DATASET INSTANTIATION =====
    #
    #
    #
    dataset = datasets.ComponentDegradationAD(dataset_path)
    test_len = int(np.floor(dataset.__len__() * test_percentage))
    train_len = dataset.__len__() - test_len
    training_dataset, test_dataset = pyt_data.random_split(dataset, (train_len, test_len))
    input_size = (dataset.__getitem__(0)[0].shape[0],)
    output_size = dataset.__getitem__(0)[1].shape[0]

    ad_test_dataset = datasets.ComponentDegradationAD(dataset_path, is_training=False)
    ad_batch_size = 65536

    logger_file.info("net_id,"
                     "optim,"
                     "lr,"
                     "loss_f,"
                     "n_epochs,"
                     "val_percentage,"
                     "train_b_size,"
                     "val_b_size,"
                     "precision_metric,"
                     "device,"
                     "test_b_size,"
                     "train_len,"
                     "test_len,"
                     "test_loss")

    # ===== MODEL GENERATION =====
    #
    #
    #
    for act_fun in activation_functions:

        for net_arch in network_arch:

            net_id = f"{dataset_id}_{act_fun.__name__}_{net_arch}".replace(", ", "-")
            network = pyn_networks.SequentialNetwork(identifier=net_id, input_id="X")

            node_index = 0
            in_dim = input_size
            for n_neurons in net_arch:
                fc_node = pyn_nodes.FullyConnectedNode(identifier=f"FC_{node_index}", in_dim=in_dim,
                                                       out_features=n_neurons)
                network.add_node(fc_node)
                node_index += 1

                act_node = act_fun(identifier=f"ACT_{node_index}", in_dim=fc_node.out_dim)
                network.add_node(act_node)
                in_dim = act_node.out_dim
                node_index += 1

            fc_out_node = pyn_nodes.FullyConnectedNode(identifier=f"FC_{node_index}", in_dim=in_dim,
                                                       out_features=output_size)
            network.add_node(fc_out_node)

            # == MODEL TRAINING == #

            logger_stream.info(f"NOW TRAINING MODEL: {network}")
            training_strategy = pyn_train.PytorchTraining(optimizer_con=optimizer_con, opt_params=opt_params,
                                                          loss_function=loss_function, n_epochs=n_epochs,
                                                          validation_percentage=validation_percentage,
                                                          train_batch_size=train_batch_size,
                                                          validation_batch_size=validation_batch_size,
                                                          precision_metric=precision_metric, device=device,
                                                          checkpoints_root=checkpoints_root)

            trained_network = training_strategy.train(network, training_dataset)

            # == MODEL TESTING == #

            testing_strategy = pyn_train.PytorchTesting(metric=precision_metric, metric_params=metric_params,
                                                        test_batch_size=test_batch_size, device=device)

            test_loss = testing_strategy.test(trained_network, test_dataset)
            logger_stream.info(f"TEST LOSS: {test_loss}")

            # == MODEL SAVING == #
            onnx_net = pyn_conv.ONNXConverter().from_neural_network(trained_network).onnx_network
            onnx.save(onnx_net, onnx_folder + f"{trained_network.identifier}.onnx")

            # == ANOMALY DETECTION OUTPUT (Saved as CSV for further analysis) == #
            if not os.path.exists(models_data_path + f"{trained_network.identifier}_losses.csv"):
                ad_losses = []
                ad_dataloader = pyt_data.DataLoader(ad_test_dataset, ad_batch_size)
                for index, (sample, target) in enumerate(ad_dataloader):
                    ad_loss = utilities.compute_loss(trained_network, nn.MSELoss(reduction='none'), sample, target).\
                        cpu().detach().numpy()
                    ad_losses.append(ad_loss)
                    logger_stream.info(ad_loss.shape)

                stacked_losses = np.vstack(tuple(ad_losses))
                logger_stream.info(stacked_losses.shape)
                losses_df = pandas.DataFrame(stacked_losses)
                losses_df.to_csv(models_data_path + f"{trained_network.identifier}_losses.csv", index=False)

            logger_file.info(f"{net_id},{optimizer_con.__name__},{opt_params['lr']},{loss_function.__class__.__name__},"
                             f"{n_epochs},{validation_percentage},{train_batch_size},{validation_batch_size},"
                             f"{precision_metric.__class__.__name__},{device},{test_batch_size},{train_len},{test_len},"
                             f"{test_loss}")


if __name__ == "__main__":
    main()
