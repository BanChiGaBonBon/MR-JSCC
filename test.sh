python test.py --dataset_mode 'CIFAR10' --v_min 0 --v_range 210 --v_step 10  --phase JSCCOFDM --modulation OTFS --S 16 --M 30 --pkt OFDM --K 8 --L 6 --C_channel 15 --N_pilot 30 --SNR 10 --feedforward 'EXPLICIT-CE-EQ' --epoch  1130s16m30ceeqotfsv100150 --is_ga

# python test.py --dataset_mode 'CIFAR10' --v_min 0 --v_range 110 --v_step 10  --phase JSCCOFDM --modulation OFDM --S 16 --M 30 --pkt OFDM --K 8 --L 6 --C_channel 15 --N_pilot 2 --SNR 10 --feedforward 'EXPLICIT-EQ' --epoch 1126s16m30gat --is_ga --is_t --d_model 128

python test.py --dataset_mode 'CIFAR10' --v_min 0 --v_range 210 --v_step 10  --phase JSCCOFDM --modulation OFDM --S 16 --M 30 --pkt OFDM --K 8 --L 6 --C_channel 15 --N_pilot 2 --SNR 10 --feedforward 'EXPLICIT-RES' --epoch 1130s16m30resv100150



