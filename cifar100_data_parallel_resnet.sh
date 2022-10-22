# bash script to sweep over local iterations
for lr in 0.01 # 3e-2 = 0.03
do
    for trial in 4
    do
        # freeze when calling the second script so you only run one test at a time
        echo "Starting test: LR="$lr "trial=" $trial
        python3 resnet_data_parallel_training.py --model_name $lr"_"$trial"_data_parallel_resnet_160" --lr $lr --pytorch-seed $trial --rank=0 --cuda-id=0 &
        python3 resnet_data_parallel_training.py --model_name $lr"_"$trial"_data_parallel_resnet_160" --lr $lr --pytorch-seed $trial --rank=1 --cuda-id=1 &
        python3 resnet_data_parallel_training.py --model_name $lr"_"$trial"_data_parallel_resnet_160" --lr $lr --pytorch-seed $trial --rank=2 --cuda-id=2 &
        python3 resnet_data_parallel_training.py --model_name $lr"_"$trial"_data_parallel_resnet_160" --lr $lr --pytorch-seed $trial --rank=3 --cuda-id=3 &
        wait # wait for everything to complete before starting next test
        echo "Test complete!"
    done
done
