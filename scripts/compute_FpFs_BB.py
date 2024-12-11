from crocodel.crocodel import astro_utils as aut
from crocodel.crocodel import astro_utils as aut
import matplotlib.pyplot as plt
import numpy as np

wavelength = np.arange(1300, 2500, 10)
T_planet, T_star = 2200, 5600
RpRs = 0.1177

wav, FpFs = aut.compute_FpFs(planet_temp = T_planet, star_temp = T_star, wavelength = wavelength, rprs = RpRs)

plt.figure()
plt.plot(wav/1000, FpFs)
plt.xlabel('Wavelength [micron]')
plt.ylabel('Fp/Fs')
plt.title('T$_{planet}$ = ' + str(T_planet) + '; T$_{star}$ = ' + str(T_star) + '; Rp/Rs = ' + str(RpRs))
plt.savefig('../examples/scratch/' + 'FpFs_BB.png', dpi = 300)
