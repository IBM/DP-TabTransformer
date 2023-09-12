# import os
# os.environ["CUDA_DEVICES_ORDER"] = "PCI_BUS_IS"
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"
# from tensorflow.python.client import device_lib
# print(device_lib.list_local_devices())


import train
import numpy as np
import compute_noise
import warnings


# for each_epoch in epoch:
#     for each_eps in eps:
#         for each_lr in lr:
#             noise_multiplyer = compute_noise.noise(N=int(35022*0.8), batch_size=64, epochs=each_epoch, eps = each_eps)
#             # noise_multiplyer = compute_noise.noise(N=int(195665 * 0.8), batch_size=128, epochs=each_epoch, eps=each_eps)
#             acc = train.train_from_scratch(noise=noise_multiplyer, batch_size=64, epochs=each_epoch, lr = each_lr)
#             print(acc_list)
            # acc_list.append(acc)
            # np.savetxt('acc_train_from_scratch.txt', acc_list, fmt='%.4f')


noise_multiplyer = compute_noise.noise(N=int(35022 * 0.8), batch_size=64, epochs=20, eps=2)
acc = train.train_from_scratch(noise=noise_multiplyer, batch_size=64, epochs=20, lr=0.001)
print(acc)
# acc_list.append(acc)
# np.savetxt('acc_train_from_scratch.txt', acc_list, fmt='%.4f')
