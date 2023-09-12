from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import warnings

from tensorflow_privacy.privacy.analysis.compute_dp_sgd_privacy_lib import compute_dp_sgd_privacy

# FLAGS = flags.FLAGS
#
# flags.DEFINE_integer('N', None, 'Total number of examples')
# flags.DEFINE_integer('batch_size', None, 'Batch size')
# flags.DEFINE_integer('epochs', None, 'Number of iterations')
# flags.DEFINE_float('eps', 1.0, 'Target epsilon')
#flags.DEFINE_float('delta', 1e-6, 'Target delta')
#CelebA: 182637

# +
#python compute_dp_sgd_privacy.py --N=60000 --batch_size=15 --epochs=18 --eps=5.0
#python compute_dp_sgd_privacy.py --N=182637 --batch_size=24 --epochs=40 --eps=0.5
#python compute_dp_sgd_privacy.py --N=83995 --batch_size=24 --epochs=40 --eps=0.5
#python compute_dp_sgd_privacy.py --N=62524 --batch_size=24 --epochs=40 --eps=0.5
# -

def noise(N, batch_size, epochs, eps):
	warnings.filterwarnings("ignore")
	noise_multiplier1 = 0.00001
	thread = 1.e-9
	delta = 1.e-5  #1.e-5
	print("\n"*2)



	current_eps, _ = compute_dp_sgd_privacy(N, batch_size, noise_multiplier1, epochs, delta)
	# print("_:", current_eps)
	while current_eps > eps:
		noise_multiplier1 += 1
		current_eps, _ = compute_dp_sgd_privacy(N, batch_size, noise_multiplier1, epochs, delta)

	c1 = False
	c2 = True
	noise_multiplier2 = noise_multiplier1 - 1
	while c1 or c2:
		last_eps = current_eps
		noise_multiplier = (noise_multiplier1 + noise_multiplier2)/2
		current_eps, _ = compute_dp_sgd_privacy(N, batch_size, noise_multiplier, epochs, delta)
		c1 = current_eps > eps
		if c1:
			noise_multiplier2 = noise_multiplier
		else:
			noise_multiplier1 = noise_multiplier
		c2 = abs(current_eps-eps)>thread

	return noise_multiplier
