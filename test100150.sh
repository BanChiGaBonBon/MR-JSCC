v=90
for i in {1..6}
do 
    v=$((v+10))
    tname1="1130s16m30resv$v"
    tname2="1130s16m30ceeqotfsv$v"
    python test.py --dataset_mode 'CIFAR10' --v_min 0 --v_range 210 --v_step 10  --phase JSCCOFDM --modulation OFDM --S 16 --M 30 --pkt OFDM --K 8 --L 6 --C_channel 15 --N_pilot 2 --SNR 10 --feedforward 'EXPLICIT-RES' --epoch $tname1
    python test.py --dataset_mode 'CIFAR10' --v_min 0 --v_range 210 --v_step 10  --phase JSCCOFDM --modulation OTFS --S 16 --M 30 --pkt OFDM --K 8 --L 6 --C_channel 15 --N_pilot 30 --SNR 10 --feedforward 'EXPLICIT-CE-EQ' --epoch $tname2 --is_ga

    # python3.8 train.py --gpu_ids '0' --feedforward 'EXPLICIT-RES'  --n_downsample 2 --C_channel 15  --SNR 10 --dataset_mode 'CIFAR10' --n_epochs 50 --n_epochs_decay 600 --pkt OFDM --modulation OFDM --S 16 --M 30 --N_pilot 2 --K 8 --L 6 --v_range $v  --v_min $v --batch_size 256 --tname $tname1 --lr_policy 'cosine'
    # python3.8 train.py --gpu_ids '0' --feedforward 'EXPLICIT-CE-EQ'  --n_downsample 2 --C_channel 15  --SNR 10 --dataset_mode 'CIFAR10' --n_epochs 50 --n_epochs_decay 600 --pkt OFDM --modulation OTFS --S 16 --M 30 --N_pilot 30 --K 8 --L 6 --v_range $v --v_min $v --batch_size 256 --tname $tname2 --lr_policy 'cosine' --is_ga
done