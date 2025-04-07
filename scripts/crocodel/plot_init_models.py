import numpy as np
import matplotlib.pyplot as plt 

f_bb = np.load('/home/astro/phsprd/wasp122b/results/retrievals/IGRINS_WASP-122b_free-chem_H2O-dissoc_updated-Fp_star-BB__N_PCA-6_all_post_pca_mask_std__07-10-2024T22-08-24/init_forward_models.npy', 
               allow_pickle = True).item()
f_phoenix = np.load('/home/astro/phsprd/wasp122b/results/retrievals/IGRINS_WASP-122b_free-chem_H2O-dissoc_updated-Fp_star-PHOENIX_vsini-7.7__N_PCA-6_all_post_pca_mask_std__28-10-2024T09-29-14/init_forward_models.npy', 
                    allow_pickle = True).item()

print(f_bb.keys())

plt.figure(figsize = (12,12))
plt.plot(f_bb['wav'], f_bb['spec'], label = 'BB', alpha = 0.7)
plt.plot(f_phoenix['wav'], f_phoenix['spec'], label = 'PHOENIX', alpha = 0.7)
plt.xlim(1600., 1800.)
plt.legend()
plt.savefig('./plot.png', dpi = 300)

