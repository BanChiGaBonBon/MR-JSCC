# python test.py --dataset_mode 'CIFAR10' --v_min 0 --v_range 210 --v_step 10  --phase JSCCOFDM --modulation OTFS --S 16 --M 30 --pkt OFDM --K 8 --L 6 --C_channel 15 --N_pilot 30 --SNR 10 --feedforward 'EXPLICIT-CE-EQ' --epoch  1130s16m30ceeqotfsv100150 --is_ga
python test.py --dataset_mode 'CIFAR10' --v_min 0 --v_range 110 --v_step 10  --phase JSCCOFDM --modulation OFDM --S 16 --M 30 --pkt OFDM --K 8 --L 6 --C_channel 15 --N_pilot 2 --SNR 10 --feedforward 'EXPLICIT-EQ' --epoch 316s16m30tcm  --is_t --d_model 128 --is_cm --V 75
python test.py --dataset_mode 'CIFAR10' --v_min 0 --v_range 110 --v_step 10  --phase JSCCOFDM --modulation OFDM --S 16 --M 30 --pkt OFDM --K 8 --L 6 --C_channel 15 --N_pilot 2 --SNR 10 --feedforward 'EXPLICIT-EQ' --epoch 316s16m30cm  --is_cm --V 75

python test.py --dataset_mode 'CIFAR10' --v_min 0 --v_range 110 --v_step 10  --phase JSCCOFDM --modulation OFDM --S 16 --M 30 --pkt OFDM --K 8 --L 6 --C_channel 15 --N_pilot 2 --SNR 10 --feedforward 'EXPLICIT-EQ' --epoch 316s16m30ccm --is_ga --is_c  --is_cm --V 75
# python test.py --dataset_mode 'CIFAR10' --v_min 0 --v_range 110 --v_step 10  --phase JSCCOFDM --modulation OFDM --S 16 --M 30 --pkt OFDM --K 8 --L 6 --C_channel 15 --N_pilot 2 --SNR 10 --feedforward 'EXPLICIT-EQ' --epoch 316s16m30lstmcm --is_ga --is_lstm --is_cm
# python test.py --dataset_mode 'CIFAR10' --v_min 0 --v_range 110 --v_step 10  --phase JSCCOFDM --modulation OFDM --S 16 --M 30 --pkt OFDM --K 8 --L 6 --C_channel 15 --N_pilot 2 --SNR 10 --feedforward 'EXPLICIT-EQ' --epoch 316s16m30gatcm --is_ga --is_t --d_model 128 --is_cm
# python test.py --dataset_mode 'CIFAR10' --v_min 0 --v_range 110 --v_step 10  --phase JSCCOFDM --modulation OFDM --S 16 --M 30 --pkt OFDM --K 8 --L 6 --C_channel 15 --N_pilot 2 --SNR 10 --feedforward 'EXPLICIT-EQ' --epoch 316s16m30gacm  --is_ga --is_cm 




# python test.py --dataset_mode 'CIFAR10' --v_min 0 --v_range 210 --v_step 10  --phase JSCCOFDM --modulation OFDM --S 16 --M 30 --pkt OFDM --K 8 --L 6 --C_channel 15 --N_pilot 2 --SNR 10 --feedforward 'EXPLICIT-RES' --epoch 1130s16m30resv100150
python test.py --dataset_mode 'CIFAR10' --v_min 0 --v_range 110 --v_step 10  --phase JSCCOFDM --modulation OFDM --S 16 --M 30 --pkt OFDM --K 8 --L 6 --C_channel 15 --N_pilot 2 --SNR 10 --feedforward 'EXPLICIT-EQ' --epoch 405s16m30ccm --is_c --is_cm --V 75
python test.py --dataset_mode 'CIFAR10' --v_min 0 --v_range 110 --v_step 10  --phase JSCCOFDM --modulation OFDM --S 16 --M 30 --pkt OFDM --K 8 --L 6 --C_channel 15 --N_pilot 2 --SNR 10 --feedforward 'EXPLICIT-EQ' --epoch 406s16m30lstmcm --is_lstm --is_cm --V 75


python test.py --dataset_mode 'CIFAR10' --v_min 0 --v_range 110 --v_step 10  --phase JSCCOFDM --modulation OFDM --S 16 --M 30 --pkt OFDM --K 8 --L 6 --C_channel 15 --N_pilot 2 --SNR 10 --feedforward 'EXPLICIT-RES' --epoch 316s16m30res --V 75
python test.py --dataset_mode 'CIFAR10' --v_min 0 --v_range 110 --v_step 10  --phase JSCCOFDM --modulation OTFS --S 16 --M 30 --pkt OFDM --K 8 --L 6 --C_channel 15 --N_pilot 30 --SNR 10 --feedforward 'EXPLICIT-CE-EQ' --epoch 316s16m30ceeqotfscm --is_cm --V 75


# python test.py --dataset_mode 'CIFAR10' --v_min 0 --v_range 210 --v_step 10  --phase JSCCOFDM --modulation OTFS --S 16 --M 30 --pkt OFDM --K 8 --L 6 --C_channel 15 --N_pilot 30 --SNR 10 --feedforward 'EXPLICIT-CE-EQ' --epoch  1130s16m30ceeqotfsv100150 --is_ga
python test.py --dataset_mode 'CIFAR10' --v_min 0 --v_range 110 --v_step 10  --phase JSCCOFDM --modulation OFDM --S 16 --M 30 --pkt OFDM --K 8 --L 6 --C_channel 15 --N_pilot 2 --SNR 10 --feedforward 'EXPLICIT-EQ' --epoch 316s16m30tcm  --is_t --d_model 128 --is_cm --V 25
python test.py --dataset_mode 'CIFAR10' --v_min 0 --v_range 110 --v_step 10  --phase JSCCOFDM --modulation OFDM --S 16 --M 30 --pkt OFDM --K 8 --L 6 --C_channel 15 --N_pilot 2 --SNR 10 --feedforward 'EXPLICIT-EQ' --epoch 316s16m30cm  --is_cm --V 25

python test.py --dataset_mode 'CIFAR10' --v_min 0 --v_range 110 --v_step 10  --phase JSCCOFDM --modulation OFDM --S 16 --M 30 --pkt OFDM --K 8 --L 6 --C_channel 15 --N_pilot 2 --SNR 10 --feedforward 'EXPLICIT-EQ' --epoch 316s16m30ccm --is_ga --is_c  --is_cm --V 25
# python test.py --dataset_mode 'CIFAR10' --v_min 0 --v_range 110 --v_step 10  --phase JSCCOFDM --modulation OFDM --S 16 --M 30 --pkt OFDM --K 8 --L 6 --C_channel 15 --N_pilot 2 --SNR 10 --feedforward 'EXPLICIT-EQ' --epoch 316s16m30lstmcm --is_ga --is_lstm --is_cm
# python test.py --dataset_mode 'CIFAR10' --v_min 0 --v_range 110 --v_step 10  --phase JSCCOFDM --modulation OFDM --S 16 --M 30 --pkt OFDM --K 8 --L 6 --C_channel 15 --N_pilot 2 --SNR 10 --feedforward 'EXPLICIT-EQ' --epoch 316s16m30gatcm --is_ga --is_t --d_model 128 --is_cm
# python test.py --dataset_mode 'CIFAR10' --v_min 0 --v_range 110 --v_step 10  --phase JSCCOFDM --modulation OFDM --S 16 --M 30 --pkt OFDM --K 8 --L 6 --C_channel 15 --N_pilot 2 --SNR 10 --feedforward 'EXPLICIT-EQ' --epoch 316s16m30gacm  --is_ga --is_cm 




# python test.py --dataset_mode 'CIFAR10' --v_min 0 --v_range 210 --v_step 10  --phase JSCCOFDM --modulation OFDM --S 16 --M 30 --pkt OFDM --K 8 --L 6 --C_channel 15 --N_pilot 2 --SNR 10 --feedforward 'EXPLICIT-RES' --epoch 1130s16m30resv100150
python test.py --dataset_mode 'CIFAR10' --v_min 0 --v_range 110 --v_step 10  --phase JSCCOFDM --modulation OFDM --S 16 --M 30 --pkt OFDM --K 8 --L 6 --C_channel 15 --N_pilot 2 --SNR 10 --feedforward 'EXPLICIT-EQ' --epoch 405s16m30ccm --is_c --is_cm --V 25
python test.py --dataset_mode 'CIFAR10' --v_min 0 --v_range 110 --v_step 10  --phase JSCCOFDM --modulation OFDM --S 16 --M 30 --pkt OFDM --K 8 --L 6 --C_channel 15 --N_pilot 2 --SNR 10 --feedforward 'EXPLICIT-EQ' --epoch 406s16m30lstmcm --is_lstm --is_cm --V 25


python test.py --dataset_mode 'CIFAR10' --v_min 0 --v_range 110 --v_step 10  --phase JSCCOFDM --modulation OFDM --S 16 --M 30 --pkt OFDM --K 8 --L 6 --C_channel 15 --N_pilot 2 --SNR 10 --feedforward 'EXPLICIT-RES' --epoch 316s16m30res --V 25
python test.py --dataset_mode 'CIFAR10' --v_min 0 --v_range 110 --v_step 10  --phase JSCCOFDM --modulation OTFS --S 16 --M 30 --pkt OFDM --K 8 --L 6 --C_channel 15 --N_pilot 30 --SNR 10 --feedforward 'EXPLICIT-CE-EQ' --epoch 316s16m30ceeqotfscm --is_cm --V 25




