# import os
# os.environ["CUDA_DEVICES_ORDER"] = "PCI_BUS_IS"
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"
# from tensorflow.python.client import device_lib
# print(device_lib.list_local_devices())

import train
import numpy as np
import compute_noise
import warnings
import keras
warnings.filterwarnings("ignore")
eps_1 = [32]
eps_2 = [32]

lr = 0.001



acc_list = [[] for _ in range(len(eps_1))]
for each_eps in eps_1:
    for another_eps in eps_2:
        another_eps = each_eps
        # tabtransformer_temp = keras.models.load_model('pretrained_model_'+str(each_eps),compile=False)
        noise_multiplyer = compute_noise.noise(N=195665, batch_size=64, epochs=5, eps=each_eps)
        tabtransformer_temp = train.pretrain(with_dp=True,
                                        noise=noise_multiplyer,
                                        epochs=5,
                                        )
        noise_multiplyer_temp = compute_noise.noise(N=int(35022*0.8), batch_size=64, epochs=20, eps=another_eps)
        acc = train.finetune(tabtransformer=tabtransformer_temp,
                             noise=noise_multiplyer_temp,
                             batch_size=64,
                             epochs=20,
                             lr = lr) # shallow tuning by default, to use adapter, set use_adapter=True,
                                      # to use LoRA, set use_adapter=True, to use full tuning, set full_tuning=True
        acc_list[eps_1.index(each_eps)].append(acc)
    np.savetxt('acc_shallow_tuning.txt', acc_list[:eps_1.index(each_eps)+1], fmt='%.4f')
