import redback
import pandas as pd
from bilby.core.prior import PriorDict
import bilby
import matplotlib.pyplot as plt
from selenium import webdriver
import numpy as np
import matplotlib as mpl
mpl.rcParams.update(mpl.rcParamsDefault)
from redback.constants import day_to_s
from redback.model_library import all_models_dict
#
COLOUR_KEY = {
    "pink": "#ff9cc2",
    "red": "#ad0000",
    "orange": "#db8f00",
    "limegreen": "#97eb10",
    "green": "#009667",
    "blue": "#1a41ed",
    "purple":"#bc46eb",
    "violet": "#b58aff",
    }
#
data = pd.read_csv('/Users/harrymccabe/Documents/PHYS498 Masters Project/160821B.csv')
df = pd.DataFrame(data=data)
print(data)
time_d = data['DeltaT'].values
time_err = data['Time Error'].values
time_e = data['ExpTime'].values
data_filter = data['Filter'].values
AB_0 = data['AB(0)'].values
flux = data['Flux'].values
flux_density = data['Flux Density'].values
flux_err = data['Flux Error'].values
flux_density_err = data["Flux Density Error"].values
frequency = data['Frequency'].values
bands = data['Filter'].values
data_mode = 'flux_density'
name = '160821B'
redshift = 0.162 # Speaks for itself really
thv = 0.0 # Observer viewing angle
loge0 = 49 # Jet energy in log10 ergs
loge0_1 = 51 # Same as above just another value to play with
loge0_2 = 100 # And again....
thc = 0.1 # Jet opening angle in radians
logn0 = -4 # ISM number density 
p = 2.2 # Electron power law index
logepse = -1 # Partition fraction in electrons
logepsb = -2 # Partition fraction in magnetic field 
g0 = 100 # Initial lorentz factor
xiN = 0.1 # Fraction of accelerated electrons
g1 = 10 # Lorentz factor of shell at start of energy injection
et = 20 # Factor kinetic energy is increased by
s1 = 12 # Index for energy injection
logtime = np.log10(time_d)
time_arr = np.logspace(-2, 2, num = 100)
freq_arr = np.ones(len(time_arr))*1e14
print(time_arr)
fig, axes = plt.subplots(1,1, figsize=(15,12))
afterglow = redback.transient.Afterglow(
    name=name, data_mode=data_mode, time=logtime,
    flux_density = flux_density, flux_density_err=flux_density_err, frequency=frequency)

ax = afterglow.plot_data(band_labels = ['g','r','F606W','i','z','F110W','H','F160W','K'], figure = fig, axes = axes, ms = 20)
afterglow.plot_multiband(band_labels = ['g','r','F606W','i','z','F110W','H','F160W','K'], axes = axes, ms = 20)
ax.set_xscale('log')
fig.show()
#
#
kwargs = dict(frequency=freq_arr, output_format = 'flux_density')
frequency = kwargs['frequency']
afterglow_tophat = redback.transient_models.afterglow_models.tophat_redback(time_arr, redshift, thv, loge0, thc, logn0, p, logepse, logepsb, g0, xiN, **kwargs)
print(afterglow_tophat)
logflux = np.log10(afterglow_tophat)
logtime = np.log10(time_arr)
plt.loglog(time_arr, afterglow_tophat)
plt.xlabel('Log Time')
plt.ylabel('Log Flux Density')
plt.title('Afterglow lightcurve Loge0 = 49')
plt.show()

kwargs = dict(frequency=freq_arr, output_format = 'flux_density')
frequency = kwargs['frequency']
afterglow_tophat = redback.transient_models.afterglow_models.tophat_redback(time_arr, redshift, thv, loge0_1, thc, logn0, p, logepse, logepsb, g0, xiN, **kwargs)
print(afterglow_tophat)
logflux = np.log10(afterglow_tophat)
logtime = np.log10(time_arr)
plt.loglog(time_arr, afterglow_tophat)
plt.xlabel('Log Time')
plt.ylabel('Log Flux Density')
plt.title('Afterglow lightcurve, Loge0 = 51')
plt.show()

kwargs = dict(frequency=freq_arr, output_format = 'flux_density')
frequency = kwargs['frequency']
afterglow_tophat = redback.transient_models.afterglow_models.tophat_redback(time_arr, redshift, thv, loge0_2, thc, logn0, p, logepse, logepsb, g0, xiN, **kwargs)
print(afterglow_tophat)
logflux = np.log10(afterglow_tophat)
logtime = np.log10(time_arr)
plt.loglog(time_arr, afterglow_tophat)
plt.xlabel('Log Time')
plt.ylabel('Log Flux Density')
plt.title('Afterglow lightcurve, Loge0 = 100')
plt.show()

kwargs = dict(frequency=freq_arr, output_format = 'flux_density')
frequency = kwargs['frequency']
afterglow_tophat = redback.transient_models.afterglow_models.tophat_redback_refreshed(time_arr, redshift, thv, loge0, thc, g1, et, s1, logn0, p, logepse, logepsb, g0, xiN, **kwargs)
print(afterglow_tophat)
logflux = np.log10(afterglow_tophat)
logtime = np.log10(time_arr)
plt.loglog(time_arr, afterglow_tophat)
plt.xlabel('Log Time')
plt.ylabel('Log Flux Density')
plt.title('Refreshed Shock Afterglow lightcurve Loge0 = 49')
plt.show()

kwargs = dict(frequency=freq_arr, output_format = 'flux_density')
frequency = kwargs['frequency']
afterglow_tophat = redback.transient_models.afterglow_models.tophat_redback_refreshed(time_arr, redshift, thv, loge0_1, thc, g1, et, s1, logn0, p, logepse, logepsb, g0, xiN, **kwargs)
print(afterglow_tophat)
logflux = np.log10(afterglow_tophat)
logtime = np.log10(time_arr)
plt.loglog(time_arr, afterglow_tophat)
plt.xlabel('Log Time')
plt.ylabel('Log Flux Density')
plt.title('Refreshed Shock Afterglow lightcurve Loge0 = 51')
plt.show()
print(len(frequency))