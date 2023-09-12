import train
import numpy as np
import compute_noise
import warnings

eps = [32]
lr = [0.001]
epoch = [20]
for each_epoch in epoch:
    for each_eps in eps:
        for each_lr in lr:
            noise_multiplyer = compute_noise.noise(N=int(35022*0.8), batch_size=64, epochs=each_epoch, eps = each_eps)
            acc = train.train_from_scratch(noise=noise_multiplyer, batch_size=64, epochs=each_epoch, lr = each_lr)
            print(acc_list)
            acc_list.append(acc)
            np.savetxt('acc_train_from_scratch.txt', acc_list, fmt='%.4f')
