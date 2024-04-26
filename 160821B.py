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
# This code operates under the assumption of 100 entries in the time and frequency array, this doesn't 
# reflect reality! the actual data is used in the combined fit 
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
AB_0 = data['magnitude'].values
flux = data['flux'].values
flux_density = data['flux_density'].values
active_filters = ['g','i','r','z','H','K','F606W','F110W','F160W']
flux_density_kilo = flux_density[data['Filter'].isin(active_filters)]
print("active flux:",flux_density_kilo)
flux_err = data['Flux Error'].values
flux_density_err = data["Flux Density Error"].values
flux_density_err_kilo = flux_density_err[data['Filter'].isin(active_filters)]
frequency = data['Frequency'].values
bands = data['Filter'].values
colour = data['Colour'].values
time_d_kilo = time_d[data['Filter'].isin(active_filters)]
data_mode = 'flux_density'
name = '160821B'
redshift = 0.162
av = 0.118
thv = 0.0
loge0 = 49
loge0_1 = 51
loge0_2 = 100
thc = 0.1
logn0 = -4
p = 2.3 # Reference Gavin 
logepse = -1
logepsb = -2
ksin = 0.3
g0 = 100
mej_1 = 0.001
vej_1 = 0.2
temperature_floor_1 = 2500 
kappa_1 = 10
kappa_2 = 1
temperature_floor_2 = 2500
vej_2 = 0.1
mej_2 = 0.01
xiN = 0.1
g1 = 10
et = 20
s1 = 12
kappa = 10
logtime = np.log10(time_d)
logtimeerr = 0
logflux = np.log10(flux_density)
logfluxerr = abs(np.log10(flux_density_err))
time_arr = np.logspace(-3, 3, num = 100)
freq_arr = np.ones(len(time_arr))*1e14
time_arr_kilo = np.linspace(0.01, 10, 100)
freq_arr_kilo = np.ones(len(time_arr_kilo))*1e14
print("time",time_arr)
fig, axes = plt.subplots(1,1, figsize=(15,12))
afterglow = redback.transient.Afterglow(
    name=name, data_mode=data_mode, time=logtime,
    flux_density = flux_density, flux_density_err=flux_density_err, frequency=frequency)

ax = afterglow.plot_data(band_labels = ['g','r','F606W','i','z','F110W','H','F160W','K', 'Radio','X-Ray'], figure = fig, axes = axes, ms = 20)
afterglow.plot_multiband(band_labels = ['g','r','F606W','i','z','F110W','H','F160W','K', 'Radio','X-Ray'], axes = axes, ms = 20)
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
plt.xlabel('Log Time (Days)')
plt.ylabel('Log Flux Density (mJy)')
plt.title('Afterglow lightcurve Loge0 = 49')
plt.show()

kwargs = dict(frequency=freq_arr, output_format = 'flux_density')
frequency = kwargs['frequency']
afterglow_tophat = redback.transient_models.afterglow_models.tophat_redback(time_arr, redshift, thv, loge0_1, thc, logn0, p, logepse, logepsb, g0, xiN, **kwargs)
print(afterglow_tophat)
logflux = np.log10(afterglow_tophat)
logtime = np.log10(time_arr)
plt.plot(time_arr, afterglow_tophat, color = 'black', label = 'Model')
plt.errorbar(time_d, flux_density, flux_density_err, logtimeerr, color = 'red', label = 'Data', linestyle = '', marker = 'o')
plt.xscale('log')
plt.yscale('log')
plt.legend()
plt.xlabel('Log Time (Days)')
plt.ylabel('Log Flux Density (mJy)')
plt.title('Afterglow lightcurve, Loge0 = 51')
plt.show()

kwargs = dict(frequency=freq_arr, output_format = 'flux_density')
frequency = kwargs['frequency']
afterglow_tophat = redback.transient_models.afterglow_models.tophat_redback(time_arr, redshift, thv, loge0_2, thc, logn0, p, logepse, logepsb, g0, xiN, **kwargs)
print(afterglow_tophat)
logflux = np.log10(afterglow_tophat)
logtime = np.log10(time_arr)
plt.loglog(time_arr, afterglow_tophat)
plt.xlabel('Log Time (Days)')
plt.ylabel('Log Flux Density (mJy)')
plt.title('Afterglow lightcurve, Loge0 = 100')
plt.show()

kwargs = dict(frequency=freq_arr, output_format = 'flux_density')
frequency = kwargs['frequency']
afterglow_tophat = redback.transient_models.afterglow_models.tophat_redback_refreshed(time_arr, redshift, thv, loge0, thc, g1, et, s1, logn0, p, logepse, logepsb, g0, xiN, **kwargs)
print(afterglow_tophat)
logflux = np.log10(afterglow_tophat)
logtime = np.log10(time_arr)
plt.loglog(time_arr, afterglow_tophat)
plt.xlabel('Log Time (Days)')
plt.ylabel('Log Flux Density (mJy)')
plt.title('Refreshed Shock Afterglow lightcurve Loge0 = 49')
plt.show()

kwargs = dict(frequency=freq_arr, output_format = 'flux_density')
frequency = kwargs['frequency']
afterglow_tophat = redback.transient_models.afterglow_models.tophat_redback_refreshed(time_arr, redshift, thv, loge0_1, thc, g1, et, s1, logn0, p, logepse, logepsb, g0, xiN, **kwargs)
print(afterglow_tophat)
logflux = np.log10(afterglow_tophat)
logtime = np.log10(time_arr)
plt.loglog(time_arr, afterglow_tophat)
plt.xlabel('Log Time (Days)')
plt.ylabel('Log Flux Density (mJy)')
plt.title('Refreshed Shock Afterglow lightcurve Loge0 = 51')
plt.show()
print(len(frequency))

kwargs = dict(frequency=freq_arr, output_format = 'flux_density')
frequency = kwargs['frequency']
afterglow_tophat = redback.transient_models.afterglow_models.tophat_redback_refreshed(time_arr, redshift, thv, loge0_1, thc, g1, et, s1, logn0, p, logepse, logepsb, g0, xiN, **kwargs)
print(afterglow_tophat)
plt.plot(time_arr, afterglow_tophat, color = 'black', label = 'Model')
plt.errorbar(time_d, flux_density, flux_density_err, logtimeerr, color = 'red', label = 'Data', linestyle = '', marker = 'o')
plt.xscale('log')
plt.yscale('log')
plt.legend()
plt.xlabel('Log Time (Days)')
plt.ylabel('Log Flux Density (mJy)')
plt.title('Refreshed Shock Afterglow lightcurve Loge0 = 51')
plt.show()

kwargs = dict(frequency=freq_arr_kilo, output_format = 'flux_density')
frequency = kwargs['frequency']
two_component = redback.transient_models.kilonova_models.two_component_kilonova_model(time_arr_kilo, redshift, mej_1, vej_1, temperature_floor_1, kappa_1, mej_2, vej_2, temperature_floor_2, kappa_2, **kwargs)
plt.plot(time_arr_kilo, (two_component))
plt.xlabel('Time (Days)')
plt.ylabel('Intensity (mJy) [Log Scale]')
plt.yscale('log')
plt.title('Two Component Kilonova Model Of GRB 160821B')
plt.show()

# Kilonovae need the radio and x-ray filtering out
kwargs = dict(frequency=freq_arr_kilo, output_format = 'flux_density')
frequency = kwargs['frequency']
two_component = redback.transient_models.kilonova_models.two_component_kilonova_model(time_arr_kilo, redshift, mej_1, vej_1, temperature_floor_1, kappa_1, mej_2, vej_2, temperature_floor_2, kappa_2, **kwargs)
plt.plot(time_arr_kilo, (two_component), color = 'black', label = 'model')
plt.errorbar(time_d_kilo, flux_density_kilo, flux_density_err_kilo, logtimeerr, color = 'red', label = 'Data', linestyle = '', marker = 'o')
plt.xlabel('Time (Days)')
plt.ylabel('Intensity (mJy) [Log Scale]')
plt.yscale('log')
plt.title('Two Component Kilonova Model Of GRB 160821B')
plt.show()