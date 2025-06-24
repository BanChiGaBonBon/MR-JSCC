# python3.8 train.py --gpu_ids '0' --feedforward 'EXPLICIT-RES'  --n_downsample 2 --C_channel 15  --SNR 10 --dataset_mode 'CIFAR10' --n_epochs 50 --n_epochs_decay 600 --pkt OFDM --modulation OFDM --S 16 --M 30 --N_pilot 2 --K 8 --L 6 --v_range 150 --v_min 100 --batch_size 256 --tname '1130s16m30resv100150' --lr_policy 'cosine'
# v=90
# for i in {1..6}
# do 
#     v=$((v+10))
#     tname1="1130s16m30resv$v"
#     tname2="1130s16m30ceeqotfsv$v"

#     python3.8 train.py --gpu_ids '0' --feedforward 'EXPLICIT-RES'  --n_downsample 2 --C_channel 15  --SNR 10 --dataset_mode 'CIFAR10' --n_epochs 50 --n_epochs_decay 600 --pkt OFDM --modulation OFDM --S 16 --M 30 --N_pilot 2 --K 8 --L 6 --v_range $v  --v_min $v --batch_size 256 --tname $tname1 --lr_policy 'cosine'
#     python3.8 train.py --gpu_ids '0' --feedforward 'EXPLICIT-CE-EQ'  --n_downsample 2 --C_channel 15  --SNR 10 --dataset_mode 'CIFAR10' --n_epochs 50 --n_epochs_decay 600 --pkt OFDM --modulation OTFS --S 16 --M 30 --N_pilot 30 --K 8 --L 6 --v_range $v --v_min $v --batch_size 256 --tname $tname2 --lr_policy 'cosine' --is_ga
# # done
# python3.8 train.py --gpu_ids '0' --feedforward 'EXPLICIT-CE-EQ'  --n_downsample 2 --C_channel 15  --SNR 10 --dataset_mode 'CIFAR10' --n_epochs 50 --n_epochs_decay 600 --pkt OFDM --modulation OTFS --S 16 --M 30 --N_pilot 30 --K 8 --L 6 --v_range 150 --v_min 100 --batch_size 256 --tname '1130s16m30ceeqotfsv100150' --lr_policy 'cosine' --is_ga
# python3.8 train.py --gpu_ids '0' --feedforward 'EXPLICIT-EQ'  --n_downsample 2 --C_channel 15  --SNR 10 --dataset_mode 'CIFAR10' --n_epochs 50 --n_epochs_decay 600 --pkt OFDM --modulation OFDM --S 16 --M 30 --N_pilot 2 --K 8 --L 6 --v_range 50 --v_min 0 --batch_size 256 --tname '316s16m30gatcm' --lr_policy 'cosine'  --is_ga --is_t --d_model 128 --is_cm
# python3.8 train.py --gpu_ids '0' --feedforward 'EXPLICIT-EQ'  --n_downsample 2 --C_channel 15  --SNR 10 --dataset_mode 'CIFAR10' --n_epochs 50 --n_epochs_decay 600 --pkt OFDM --modulation OFDM --S 16 --M 30 --N_pilot 2 --K 8 --L 6 --v_range 50 --v_min 0 --batch_size 256 --tname '316s16m30gacm' --lr_policy 'cosine'  --is_ga --is_cm

# python3.8 train.py --gpu_ids '0' --feedforward 'EXPLICIT-EQ'  --n_downsample 2 --C_channel 15  --SNR 10 --dataset_mode 'CIFAR10' --n_epochs 100 --n_epochs_decay 100 --pkt OFDM --modulation OFDM --S 16 --M 30 --N_pilot 2 --K 8 --L 6 --v_range 150 --v_min 100 --batch_size 256 --tname '1130s16m30gat100150' --lr_policy 'linear'  --is_ga --is_t --d_model 128

# python3.8 train.py --gpu_ids '0' --feedforward 'EXPLICIT-EQ'  --n_downsample 2 --C_channel 15  --SNR 10 --dataset_mode 'CIFAR10' --n_epochs 50 --n_epochs_decay 600 --pkt OFDM --modulation OFDM --S 16 --M 30 --N_pilot 2 --K 8 --L 6 --v_range 50 --v_min 0 --batch_size 256 --tname '316s16m30tcm' --lr_policy 'cosine'  --is_t --d_model 128 --is_cm
# python3.8 train.py --gpu_ids '0' --feedforward 'EXPLICIT-EQ'  --n_downsample 2 --C_channel 15  --SNR 10 --dataset_mode 'CIFAR10' --n_epochs 50 --n_epochs_decay 600 --pkt OFDM --modulation OFDM --S 16 --M 30 --N_pilot 2 --K 8 --L 6 --v_range 50 --v_min 0 --batch_size 256 --tname '316s16m30cm' --lr_policy 'cosine'  --is_cm


# python3.8 train.py --gpu_ids '0' --feedforward 'EXPLICIT-EQ'  --n_downsample 2 --C_channel 15  --SNR 10 --dataset_mode 'CIFAR10' --n_epochs 50 --n_epochs_decay 600 --pkt OFDM --modulation OFDM --S 16 --M 30 --N_pilot 2 --K 8 --L 6 --v_range 50 --v_min 0 --batch_size 256 --tname '405s16m30gaccm' --lr_policy 'cosine' --is_ga --is_c --is_cm
# python3.8 train.py --gpu_ids '0' --feedforward 'EXPLICIT-EQ'  --n_downsample 2 --C_channel 15  --SNR 10 --dataset_mode 'CIFAR10' --n_epochs 50 --n_epochs_decay 600 --pkt OFDM --modulation OFDM --S 16 --M 30 --N_pilot 2 --K 8 --L 6 --v_range 150 --v_min 100 --batch_size 256 --tname '316s16m30gacm100' --lr_policy 'cosine'  --is_ga --is_cm

# python3.8 train.py --gpu_ids '0' --feedforward 'EXPLICIT-EQ'  --n_downsample 2 --C_channel 15  --SNR 10 --dataset_mode 'CIFAR10' --n_epochs 50 --n_epochs_decay 600 --pkt OFDM --modulation OFDM --S 16 --M 30 --N_pilot 2 --K 8 --L 6 --v_range 150 --v_min 100 --batch_size 256 --tname '405s16m30gaccm100' --lr_policy 'cosine' --is_ga --is_c --is_cm
# python3.8 train.py --gpu_ids '0' --feedforward 'EXPLICIT-EQ'  --n_downsample 2 --C_channel 15  --SNR 10 --dataset_mode 'CIFAR10' --n_epochs 50 --n_epochs_decay 600 --pkt OFDM --modulation OFDM --S 16 --M 30 --N_pilot 2 --K 8 --L 6 --v_range 150 --v_min 100 --batch_size 256 --tname '405s16m30lstmcm100' --lr_policy 'cosine' --is_ga --is_lstm --is_cm
# python3.8 train.py --gpu_ids '0' --feedforward 'EXPLICIT-EQ'  --n_downsample 2 --C_channel 15  --SNR 10 --dataset_mode 'CIFAR10' --n_epochs 50 --n_epochs_decay 600 --pkt OFDM --modulation OFDM --S 16 --M 30 --N_pilot 2 --K 8 --L 6 --v_range 150 --v_min 100 --batch_size 256 --tname '316s16m30gatcm100' --lr_policy 'cosine' --is_t --is_ga --is_cm

# python3.8 train.py --gpu_ids '0' --feedforward 'EXPLICIT-EQ'  --n_downsample 2 --C_channel 15  --SNR 10 --dataset_mode 'CIFAR10' --n_epochs 50 --n_epochs_decay 600 --pkt OFDM --modulation OFDM --S 16 --M 30 --N_pilot 2 --K 8 --L 6 --v_range 150 --v_min 100 --batch_size 256 --tname '316s16m30cm100' --lr_policy 'cosine' --is_cm

python3.8 train.py --gpu_ids '0' --feedforward 'EXPLICIT-EQ'  --n_downsample 2 --C_channel 15  --SNR 10 --dataset_mode 'CIFAR10' --n_epochs 50 --n_epochs_decay 600 --pkt OFDM --modulation OFDM --S 16 --M 30 --N_pilot 2 --K 8 --L 6 --v_range 50 --v_min 0 --batch_size 256 --tname '405s16m30ccm' --lr_policy 'cosine' --is_c --is_cm
python3.8 train.py --gpu_ids '0' --feedforward 'EXPLICIT-RES'  --n_downsample 2 --C_channel 15  --SNR 10 --dataset_mode 'CIFAR10' --n_epochs 50 --n_epochs_decay 600 --pkt OFDM --modulation OFDM --S 16 --M 30 --N_pilot 2 --K 8 --L 6 --v_range 50 --v_min 0 --batch_size 256 --tname '316s16m30res' --lr_policy 'cosine'
python3.8 train.py --gpu_ids '0' --feedforward 'EXPLICIT-CE-EQ'  --n_downsample 2 --C_channel 15  --SNR 10 --dataset_mode 'CIFAR10' --n_epochs 50 --n_epochs_decay 600 --pkt OFDM --modulation OTFS --S 16 --M 30 --N_pilot 30 --K 8 --L 6 --v_range 50 --v_min 0 --batch_size 256 --tname '316s16m30ceeqotfscm' --lr_policy 'cosine' --is_cm
# python3.8 train.py --gpu_ids '0' --feedforward 'EXPLICIT-EQ'  --n_downsample 2 --C_channel 15  --SNR 10 --dataset_mode 'CIFAR10' --n_epochs 150 --n_epochs_decay 150 --pkt OFDM --modulation OFDM --S 16 --M 30 --N_pilot 2 --K 8 --L 6 --v_range 50 --v_min 0 --batch_size 256 --tname '316s16m30cm' --lr_policy 'linear'  --is_cm
# python3.8 train.py --gpu_ids '0' --feedforward 'EXPLICIT-EQ'  --n_downsample 2 --C_channel 15  --SNR 10 --dataset_mode 'CIFAR10' --n_epochs 150 --n_epochs_decay 150 --pkt OFDM --modulation OFDM --S 16 --M 30 --N_pilot 2 --K 8 --L 6 --v_range 50 --v_min 0 --batch_size 256 --tname '316s16m30tcm' --lr_policy 'linear'  --is_cm --is_t



