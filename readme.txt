This is the ResIST code for 4 GPU distributed cifar100 experiments. "cifar100_local_iteration_test.py" is the ResIST code while "cifar100_local_sgd_test.py" is the local sgd baseline. We tested them on AWS p3.8xlarge instance for this experiment. Please use AWS AMI 18.04 image and pytorch_p36 conda enviornment.

Run "cifar100_local_iteration_test.sh" for ResIST experiment.
Run "cifar100_local_sgd_test.sh" for local sgd baseline.
Result will be stored in the "log" folder.


Training settings: 160 epochs total
LocalSGD: 70.21%, 69.97%, 70.14%                    -> 70.11 +/- 0.12 std


ResIST: 71.27%, 71.48%, 71.76%                      -> 71.50 +/- 0.25 std


PCRIST (freq 95, iter 5): 71.08%, 71.53%, 72.06%    -> 71.56 +/- 0.49 std


PCRIST (freq 80, iter 20): 71.14%, 71.49%, 70.90%


PCRIST (freq 60, iter 40): 71.45%, 70.46%
2022-10-15-14_29_10, 2022-10-15-14_29_45

PCRIST (freq 20, iter 5): 71.17%, 71.86%
2022-10-15-23_30_13, 2022-10-15-23_31_14, 2022-10-16-11_12_05

PCRIST (freq 15, iter 5):
2022-10-16-11_14_41
