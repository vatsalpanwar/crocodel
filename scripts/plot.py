import numpy as np
import matplotlib.pyplot as plt 

f_bb = np.load('/home/astro/phsprd/code/crocodel/examples/wasp122b/results/retrievals/TEST_GENESIS_v2__N_PCA-6_all_post_pca_mask_std__07-10-2024T10-42-28/init_forward_models.npy', allow_pickle = True).item()
f_phoenix = np.load('//home/astro/phsprd/code/crocodel/examples/wasp122b/results/retrievals/TEST_PHOENIX__N_PCA-6_all_post_pca_mask_std__07-10-2024T11-29-14/init_forward_models.npy', allow_pickle = True).item()

print(f_bb.keys())

plt.figure(figsize = (12,12))
plt.plot(f_bb['wav'], f_bb['spec'], label = 'BB', alpha = 0.7)
plt.plot(f_phoenix['wav'], f_phoenix['spec'], label = 'PHOENIX', alpha = 0.7)
plt.legend()
plt.savefig('./plot.png', dpi = 300)

