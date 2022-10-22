This is the ResIST code for 4 GPU distributed cifar100 experiments. "cifar100_local_iteration_test.py" is the ResIST code while "cifar100_local_sgd_test.py" is the local sgd baseline. We tested them on AWS p3.8xlarge instance for this experiment. Please use AWS AMI 18.04 image and pytorch_p36 conda enviornment.

Run "cifar100_local_iteration_test.sh" for ResIST experiment.
Run "cifar100_local_sgd_test.sh" for local sgd baseline.
Result will be stored in the "log" folder.


Training settings: 160 epochs total
LocalSGD: 70.21%, 69.97%, 70.14%                    ***-> 70.11 +/- 0.12 std


ResIST: 71.27%, 71.48%, 71.76%, 71.17%                      ***-> 71.50 +/- 0.25 std
- 3 lower: 71.21% +/- 0.12


PCRIST (freq 95, iter 5): 71.08%, 71.53%, 72.06%*, 71.03%, 71.49%, 71.64%*, 71.78%*, 71.21%
    ***-> 71.82% +/- 0.21
Moana (4-9): 2022-10-16-13_52_33, 2022-10-16-13_53_22, 2022-10-16-13_56_34, 2022-10-16-13_59_51, 2022-10-16-14_01_48

PCRIST (freq 90, iter 10): 71.61%, 72.00%, 70.93%, 71.44%, 70.99%
Moana: 2022-10-17-03_30_43, 2022-10-17-03_31_07, 2022-10-17-03_31_37, 2022-10-17-03_32_26, 2022-10-17-03_33_06


PCRIST (freq 80, iter 20): 71.14%, 71.49%, 70.90%


PCRIST (freq 60, iter 40): 71.45%, 70.46%
2022-10-15-14_29_10, 2022-10-15-14_29_45

PCRIST (freq 20, iter 5): 71.17%, 71.86%, 71.75%, 71.32%, 71.42%
2022-10-15-23_30_13, 2022-10-15-23_31_14, 2022-10-16-11_12_05, 2022-10-16-16_52_30, 2022-10-16-16_54_09

PCRIST (freq 15, iter 5): 71.30%
2022-10-16-11_14_41



PCRIST (freq 5, iter 95): 64.96%, 65.05%
2022-10-16-14_06_16, 2022-10-16-14_07_04



PCRIST (freq 95, iter 2): 71.79%, 71.61%, 71.96%, 71.76%, 72.16%*, 71.76%, 70.77%, 72.03%*, 72.16%*
    => 72.11 +/- 0.08
    Overall mean: 71.78 += 0.42
2022-10-17-18_51_14, 2022-10-17-18_51_38, 2022-10-17-21_23_45, 2022-10-17-21_23_56
Moana: 2022-10-17-21_17_24, 2022-10-17-21_20_49, 2022-10-17-21_21_18, 2022-10-17-21_21_35, 2022-10-17-21_22_39



PCRIST (freq 95, iter 1): 71.64%, 71.69%, 72.03%*, 71.24%, 71.16%, 71.87%*, 71.77%, 71.78%, 71.89%*
    => 71.93 +/- 0.09
2022-10-18-02_14_56, 2022-10-18-02_15_14, 2022-10-18-04_03_20, 2022-10-18-04_03_45
Moana: 2022-10-18-03_56_04, 2022-10-18-03_56_24, 2022-10-18-03_56_42, 2022-10-18-03_57_12, 2022-10-18-03_57_33



FLOPs:
PCRIST - 2.389906e+14 => 238,990,600,000,000 FLOPs per epoch
ResIST - 2.401964e+14 => 240,196,400,000,000 FLOPS per epoch
LocalSGD - 4.8947896e+14 => 489,478,960,000,000 FLOPs per epoch


