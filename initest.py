import numpy as np
d = np.load('train_pair_0000_to_0005.npz')
print(d['D_gt_a'].shape)   # → (C, H, W)