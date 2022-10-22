import os
import numpy as np
from glob import glob
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# ToDo: Thicker line, less whitespace (start from 40 or 60+% accuracy), either average multiple runs or use gaussian filter to smoothen curve, add time vs test accuracy figure
# ToDo: Combine curves into subplots, 6 subplots:
#  Cifar100 and Cifar10 time, compute budget, and FLOPs vs test accuracy
# ToDo: improve table once Cifar10 results are there, multi dataset table results

def plot_time_vs_test_acc(experiments):
    timepoint_list = []
    experiment_type_list = []
    test_acc_list = []

    for expt in experiments.keys():
        # print(expt, glob(experiments[expt] + "*_time.log"))
        time_log_filepath = glob(experiments[expt] + "*_time.log")[0]
        # time_log = np.load(time_log_filepath, allow_pickle=False)
        f = open(time_log_filepath, "r")
        time_log_str = f.readlines()[0]
        time_log = time_log_str.split(" ")
        time_log = [float(elem) for elem in time_log if elem != ""]

        test_acc_filepath = glob(experiments[expt] + "*_test_acc.log")[0]
        # test_acc = np.load(test_acc_filepath, allow_pickle=True)
        f = open(test_acc_filepath, "r")
        test_acc_str = f.readlines()[0]
        test_acc = test_acc_str.split(" ")
        test_acc = [float(elem) for elem in test_acc if elem != ""]

        for idx in range(len(time_log)):
            timepoint_list.append(time_log[idx])
            test_acc_list.append(test_acc[idx])
            experiment_type_list.append(expt.split("_")[0])

    visual_df = pd.DataFrame({
        "Time (s)": timepoint_list,
        "Test Accuracy": test_acc_list,
        "Experiment": experiment_type_list
    })

    plt.rcParams.update({'font.size': 18})
    sns.lineplot(data=visual_df, x="Time (s)", y="Test Accuracy", hue="Experiment")
    plt.savefig("/home/cougarnet.uh.edu/srizvi7/Desktop/ResIST_release/runs/time_vs_test_acc_figure.png", facecolor="white", bbox_inches="tight")


def plot_compute_budget_vs_test_acc(experiments):
    compute_budget_list = []
    experiment_type_list = []
    test_acc_list = []

    localsgd_model_size = 1.325
    resist_model_size = 0.72  # GB
    repartition_iter_central_sync_ratio = 0.5154  # 50 repartition iter / 97 central training cycle

    expt_keys = experiments.keys()
    expt_keys = [expt for expt in expt_keys if "run1" in expt]

    for expt in expt_keys:
        # print(expt, glob(experiments[expt] + "*_time.log"))
        time_log_filepath = glob(experiments[expt] + "*_time.log")[0]
        # time_log = np.load(time_log_filepath, allow_pickle=False)
        f = open(time_log_filepath, "r")
        time_log_str = f.readlines()[0]
        time_log = time_log_str.split(" ")
        time_log = [float(elem) for elem in time_log if elem != ""]

        # Find index where expt stops, cut off
        end_idx = time_log.index(0.0000)
        print("{} end index: {}".format(expt, end_idx))
        time_log = time_log[:end_idx]

        test_acc_filepath = glob(experiments[expt] + "*_test_acc.log")[0]
        # test_acc = np.load(test_acc_filepath, allow_pickle=True)
        f = open(test_acc_filepath, "r")
        test_acc_str = f.readlines()[0]
        test_acc = test_acc_str.split(" ")
        test_acc = [float(elem) for elem in test_acc if elem != ""]
        test_acc = test_acc[:end_idx]

        compute_budget = 0.
        for idx in range(len(time_log)):
            if "ResIST" in expt:
                compute_budget += resist_model_size
                compute_budget_list.append(compute_budget)
            elif "PCRIST" in expt:
                # 1 sync every args["repartition_iter"] iterations, plus centralized training periods adding 1 synchronication every once awhile. ratio is defined
                compute_budget += (1 + repartition_iter_central_sync_ratio) * resist_model_size
                compute_budget_list.append(compute_budget)
            elif "LocalSGD" in expt:
                compute_budget += localsgd_model_size
                compute_budget_list.append(compute_budget)
            test_acc_list.append(test_acc[idx])
            experiment_type_list.append(expt.split("_")[0])

    print("max:", compute_budget_list[-1], test_acc_list[-1], experiment_type_list[-1])
    visual_df = pd.DataFrame({
        "Communication Budget (GB)": compute_budget_list,
        "Test Accuracy": test_acc_list,
        "Experiment": experiment_type_list
    })

    plt.rcParams.update({'font.size': 18})
    sns.lineplot(data=visual_df, x="Communication Budget (GB)", y="Test Accuracy", hue="Experiment")
    plt.savefig("/home/cougarnet.uh.edu/srizvi7/Desktop/ResIST_release/runs/compute_budget_vs_test_acc_figure_95-2.png",
                facecolor="white", bbox_inches="tight")


def plot_flops_vs_test_acc(experiments):
    epoch = 40.
    flops_list = []
    experiment_type_list = []
    test_acc_list = []

    expt_keys = experiments.keys()
    expt_keys = [expt for expt in expt_keys if "run1" in expt]

    for expt in expt_keys:
        # print(expt, glob(experiments[expt] + "*_time.log"))
        time_log_filepath = glob(experiments[expt] + "*_time.log")[0]
        # time_log = np.load(time_log_filepath, allow_pickle=False)
        f = open(time_log_filepath, "r")
        time_log_str = f.readlines()[0]
        time_log = time_log_str.split(" ")
        time_log = [float(elem) for elem in time_log if elem != ""]

        # Find index where expt stops, cut off
        end_idx = time_log.index(0.0000)
        print("{} end index: {}".format(expt, end_idx))
        time_log = time_log[:end_idx]

        test_acc_filepath = glob(experiments[expt] + "*_test_acc.log")[0]
        # test_acc = np.load(test_acc_filepath, allow_pickle=True)
        f = open(test_acc_filepath, "r")
        test_acc_str = f.readlines()[0]
        test_acc = test_acc_str.split(" ")
        test_acc = [float(elem) for elem in test_acc if elem != ""]
        test_acc = test_acc[:end_idx]

        # compute_budget = 0.
        if "ResIST" in expt:
            resist_gflops_one_epoch = 240196.4
            flops = resist_gflops_one_epoch * len(test_acc)
            print("ResIST:", flops)
            flops_list += list(np.arange(0, flops, (flops / len(test_acc))))
            print(len(flops_list))
        elif "PCRIST" in expt:
            pcrist_gflops_one_epoch = 238990.6
            flops = pcrist_gflops_one_epoch * len(test_acc)
            print("PCRIST:", flops)
            flops_list += list(np.arange(0, flops, (flops / len(test_acc))))
            print(len(flops_list))
        elif "LocalSGD" in expt:
            localsgd_gflops_one_epoch = 489478.96
            flops = localsgd_gflops_one_epoch * len(test_acc)
            print("LocalSGD:", flops)
            flops_list += list(np.arange(0, flops, (flops / len(test_acc))))
            print(len(flops_list))

        for idx in range(len(time_log)):
            test_acc_list.append(test_acc[idx])
            experiment_type_list.append(expt.split("_")[0])

    # print("max:", compute_budget_list[-1], test_acc_list[-1], experiment_type_list[-1])
    print("Lengths:", len(test_acc_list), ",", len(experiment_type_list))
    visual_df = pd.DataFrame({
        "GigaFLOPs": flops_list,
        "Test Accuracy": test_acc_list,
        "Experiment": experiment_type_list
    })

    plt.rcParams.update({'font.size': 18})
    sns.lineplot(data=visual_df, x="GigaFLOPs", y="Test Accuracy", hue="Experiment")
    # plt.xscale("log")
    plt.savefig("/home/cougarnet.uh.edu/srizvi7/Desktop/ResIST_release/runs/flops_vs_test_acc_figure_95-2.png", facecolor="white", bbox_inches="tight")


def main():
    experiments = {
        "LocalSGD_run1": "/home/cougarnet.uh.edu/srizvi7/Desktop/ResIST_release/runs/LocalSGD/2022-10-14-22_10_49/",
        "LocalSGD_run2": "/home/cougarnet.uh.edu/srizvi7/Desktop/ResIST_release/runs/LocalSGD/2022-10-15-00_19_09/",
        "LocalSGD_run3": "/home/cougarnet.uh.edu/srizvi7/Desktop/ResIST_release/runs/LocalSGD/2022-10-15-09_26_40/",
        "ResIST_run1": "/home/cougarnet.uh.edu/srizvi7/Desktop/ResIST_release/runs/ResIST/2022-10-14-17_48_18/",
        "ResIST_run2": "/home/cougarnet.uh.edu/srizvi7/Desktop/ResIST_release/runs/ResIST/2022-10-14-20_26_42/",
        "ResIST_run3": "/home/cougarnet.uh.edu/srizvi7/Desktop/ResIST_release/runs/ResIST/2022-10-14-22_06_31/",
        "PCRIST_run1": "/home/cougarnet.uh.edu/srizvi7/Desktop/ResIST_release/runs/ResIST_centralized_training/2022-10-17-21_17_24/",
        "PCRIST_run2": "/home/cougarnet.uh.edu/srizvi7/Desktop/ResIST_release/runs/ResIST_centralized_training/2022-10-17-21_21_35/",
        "PCRIST_run3": "/home/cougarnet.uh.edu/srizvi7/Desktop/ResIST_release/runs/ResIST_centralized_training/2022-10-17-21_22_39/",
    }

    # plot_time_vs_test_acc(experiments)
    # plot_compute_budget_vs_test_acc(experiments)
    plot_flops_vs_test_acc(experiments)


if __name__ == "__main__":
    main()
