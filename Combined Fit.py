# Some experimenting with the raw data and importing and formatting important for the sampling done here
import redback
import pandas as pd
from bilby.core.prior import LogUniform, Uniform
import bilby
import matplotlib.pyplot as plt
import numpy as np
import matplotlib as mpl
mpl.rcParams.update(mpl.rcParamsDefault)

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
flux_err = data['Flux Error'].values
flux_density_err = data["Flux Density Error"].values
frequency = data['Frequency'].values
bands = data['Filter'].values
colour = data['Colour'].values
data_mode = 'flux_density'
name = '160821B'
active_filters = ['g','i','r','z','H','K','F606W','F110W','F160W']
flux_density_kilo = flux_density[data['Filter'].isin(active_filters)]
flux_density_err_kilo = flux_density_err[data['Filter'].isin(active_filters)]
time_d_kilo = time_d[data['Filter'].isin(active_filters)]
freq_arr_kilo = np.ones(len(time_d_kilo))*1e14

g_band_error = flux_density_err[data["Filter"] == 'g']
i_band_error = flux_density_err[data["Filter"] == 'i']
r_band_error = flux_density_err[data["Filter"] == 'r']
z_band_error = flux_density_err[data["Filter"] == 'z']
H_band_error = flux_density_err[data["Filter"] == 'H']
K_band_error = flux_density_err[data["Filter"] == 'K']
F606W_band_error = flux_density_err[data["Filter"] == 'F606W']
F110W_band_error = flux_density_err[data["Filter"] == 'F110W']
F160W_band_error = flux_density_err[data["Filter"] == 'F160W']
Radio_band_error = flux_density_err[data["Filter"] == 'Radio']
Xray_band_error = flux_density_err[data["Filter"] == 'X-Ray']
sigma = 1.35e-29
redshift = 0.162
av = 0.118
thv = 0.0
loge0 = 51
thc = 0.1
logn0 = -4
p = 2.3  
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
logtime = np.log10(time_d)
time_arr = np.logspace(-2, 2, num = 37)
freq_arr = np.ones(len(time_arr))*1e14
print(time_arr)
time_arr_two = np.linspace(0.01, 30, 37)
time_arr_four = 10**(np.linspace(-2, 1.5, num = 37))
print('time:', time_arr_four)
afterglow = redback.transient.Afterglow(
    name=name, data_mode=data_mode, time=time_d,
    flux_density = flux_density, flux_density_err=flux_density_err, frequency=frequency)

g_band_data = data[data["Filter"] == 'g']
print("g-band:", g_band_data)
time_arr_g = (g_band_data["DeltaT"].values)
print(time_arr_g)
freq_arr_g = np.ones(len(time_arr_g))*6.32e14
print(freq_arr_g)
i_band_data = data[data["Filter"] == 'i']
time_arr_i = (i_band_data["DeltaT"].values)
freq_arr_i = np.ones(len(time_arr_i))*3.83e14
r_band_data = data[data["Filter"] == 'r'] 
time_arr_r = (r_band_data["DeltaT"].values)
freq_arr_r = np.ones(len(time_arr_r))*4.86e14
z_band_data = data[data["Filter"] == 'z']
time_arr_z = (z_band_data["DeltaT"].values)
freq_arr_z = np.ones(len(time_arr_z))*3.46e14
H_band_data = data[data["Filter"] == 'H']
time_arr_H = (H_band_data["DeltaT"].values)
freq_arr_H = np.ones(len(time_arr_H))*1.88e14
K_band_data = data[data["Filter"] == 'K']
time_arr_K = (K_band_data["DeltaT"].values)
freq_arr_K = np.ones(len(time_arr_K))*1.36e14
F606W_band_data = data[data["Filter"] == 'F606W']
time_arr_F606W = (F606W_band_data["DeltaT"].values)
freq_arr_F606W = np.ones(len(time_arr_F606W))*4.95e14
F110W_band_data = data[data["Filter"] == 'F110W']
time_arr_F110W = (F110W_band_data["DeltaT"].values)
freq_arr_F110W = np.ones(len(time_arr_F110W))*2.73e14
F160W_band_data = data[data["Filter"] == 'F160W']
time_arr_F160W = (F160W_band_data["DeltaT"].values)
freq_arr_F160W = np.ones(len(time_arr_F160W))*1.88e14
Radio_band_data = data[data["Filter"] == 'Radio']
time_arr_Radio = (Radio_band_data["DeltaT"].values)
freq_arr_Radio = np.ones(len(time_arr_Radio))*9.8e9
Xray_band_data = data[data["Filter"] == 'X-Ray']
time_arr_Xray = (Xray_band_data["DeltaT"].values)
freq_arr_Xray = np.ones(len(time_arr_Xray))*2.4e17
print("xray data:", Xray_band_data)

freq_arr_2 = np.ones(len(time_arr_two))*1e14
kwargs = dict(frequency=freq_arr_2, output_format = 'flux_density')
frequency = kwargs['frequency']
two_component_model = redback.transient_models.kilonova_models.two_component_kilonova_model(time_d, redshift, mej_1, vej_1, temperature_floor_1, kappa_1, mej_2, vej_2, temperature_floor_2, kappa_2, **kwargs)
kwargs = dict(frequency=freq_arr_g, output_format = 'flux_density', data_mode = 'flux_density')
frequency = freq_arr_g
two_component_g = redback.transient_models.kilonova_models.two_component_kilonova_model(time_arr_g, redshift, mej_1, vej_1, temperature_floor_1, kappa_1, mej_2, vej_2, temperature_floor_2, kappa_2, **kwargs)
kwargs = dict(frequency=freq_arr_i, output_format = 'flux_density')
frequency = freq_arr_i
two_component_i = redback.transient_models.kilonova_models.two_component_kilonova_model(time_arr_i, redshift, mej_1, vej_1, temperature_floor_1, kappa_1, mej_2, vej_2, temperature_floor_2, kappa_2, **kwargs)
kwargs = dict(frequency=freq_arr_r, output_format = 'flux_density')
frequency = freq_arr_r
two_component_r = redback.transient_models.kilonova_models.two_component_kilonova_model(time_arr_r, redshift, mej_1, vej_1, temperature_floor_1, kappa_1, mej_2, vej_2, temperature_floor_2, kappa_2, **kwargs)
kwargs = dict(frequency=freq_arr_z, output_format = 'flux_density')
frequency = freq_arr_z
two_component_z = redback.transient_models.kilonova_models.two_component_kilonova_model(time_arr_z, redshift, mej_1, vej_1, temperature_floor_1, kappa_1, mej_2, vej_2, temperature_floor_2, kappa_2, **kwargs)
kwargs = dict(frequency=freq_arr_H, output_format = 'flux_density')
frequency = freq_arr_H
two_component_H = redback.transient_models.kilonova_models.two_component_kilonova_model(time_arr_H, redshift, mej_1, vej_1, temperature_floor_1, kappa_1, mej_2, vej_2, temperature_floor_2, kappa_2, **kwargs)
kwargs = dict(frequency=freq_arr_K, output_format = 'flux_density')
frequency = freq_arr_K
two_component_K = redback.transient_models.kilonova_models.two_component_kilonova_model(time_arr_K, redshift, mej_1, vej_1, temperature_floor_1, kappa_1, mej_2, vej_2, temperature_floor_2, kappa_2, **kwargs)
kwargs = dict(frequency=freq_arr_F606W, output_format = 'flux_density')
frequency = freq_arr_F606W
two_component_F606W = redback.transient_models.kilonova_models.two_component_kilonova_model(time_arr_F606W, redshift, mej_1, vej_1, temperature_floor_1, kappa_1, mej_2, vej_2, temperature_floor_2, kappa_2, **kwargs)
kwargs = dict(frequency=freq_arr_F110W, output_format = 'flux_density')
frequency = freq_arr_F110W
two_component_F110W = redback.transient_models.kilonova_models.two_component_kilonova_model(time_arr_F110W, redshift, mej_1, vej_1, temperature_floor_1, kappa_1, mej_2, vej_2, temperature_floor_2, kappa_2, **kwargs)
kwargs = dict(frequency=freq_arr_F160W, output_format = 'flux_density')
frequency = freq_arr_F160W
two_component_F160W = redback.transient_models.kilonova_models.two_component_kilonova_model(time_arr_F160W, redshift, mej_1, vej_1, temperature_floor_1, kappa_1, mej_2, vej_2, temperature_floor_2, kappa_2, **kwargs)
kwargs = dict(frequency=freq_arr_Radio, output_format = 'flux_density')
frequency = freq_arr_Radio
two_component_Radio = redback.transient_models.kilonova_models.two_component_kilonova_model(time_arr_Radio, redshift, mej_1, vej_1, temperature_floor_1, kappa_1, mej_2, vej_2, temperature_floor_2, kappa_2, **kwargs)
wargs = dict(frequency=freq_arr_Xray, output_format = 'flux_density')
frequency = freq_arr_Xray
two_component_Xray = redback.transient_models.kilonova_models.two_component_kilonova_model(time_arr_Xray, redshift, mej_1, vej_1, temperature_floor_1, kappa_1, mej_2, vej_2, temperature_floor_2, kappa_2, **kwargs)

plt.plot(time_d, two_component_model)
plt.xlabel('Time (Days)')
plt.ylabel('Intensity (mJy)')
plt.title('Two Component Kilonova Model Of 160821B')
plt.show()
df.loc[df["Filter"] == "X-Ray" , "flux_density"] = 0
df.loc[df["Filter"] == "Radio" , "flux_density"] = 0
print(df)
kwargs = dict(frequency=freq_arr, output_format = 'flux_density')
frequency2 = kwargs['frequency']
afterglow_tophat_model = redback.transient_models.afterglow_models.tophat_redback_refreshed(time_d, redshift, thv, loge0, thc, g1, et, s1, logn0, p, logepse, logepsb, g0, xiN, **kwargs)
kwargs = dict(frequency=freq_arr_g, output_format = 'flux_density', data_mode = 'flux_density')
frequency = freq_arr_g
afterglow_tophat_g = redback.transient_models.afterglow_models.tophat_redback_refreshed(time_arr_g, redshift, thv, loge0, thc, g1, et, s1, logn0, p, logepse, logepsb, g0, xiN, **kwargs)
kwargs = dict(frequency=freq_arr_i, output_format = 'flux_density')
frequency = freq_arr_i
afterglow_tophat_i = redback.transient_models.afterglow_models.tophat_redback_refreshed(time_arr_i, redshift, thv, loge0, thc, g1, et, s1, logn0, p, logepse, logepsb, g0, xiN, **kwargs)
kwargs = dict(frequency=freq_arr_r, output_format = 'flux_density')
frequency = freq_arr_r
afterglow_tophat_r = redback.transient_models.afterglow_models.tophat_redback_refreshed(time_arr_r, redshift, thv, loge0, thc, g1, et, s1, logn0, p, logepse, logepsb, g0, xiN, **kwargs)
kwargs = dict(frequency=freq_arr_z, output_format = 'flux_density')
frequency = freq_arr_z
afterglow_tophat_z = redback.transient_models.afterglow_models.tophat_redback_refreshed(time_arr_z, redshift, thv, loge0, thc, g1, et, s1, logn0, p, logepse, logepsb, g0, xiN, **kwargs)
kwargs = dict(frequency=freq_arr_H, output_format = 'flux_density')
frequency = freq_arr_H
afterglow_tophat_H = redback.transient_models.afterglow_models.tophat_redback_refreshed(time_arr_H, redshift, thv, loge0, thc, g1, et, s1, logn0, p, logepse, logepsb, g0, xiN, **kwargs)
kwargs = dict(frequency=freq_arr_K, output_format = 'flux_density')
frequency = freq_arr_K
afterglow_tophat_K = redback.transient_models.afterglow_models.tophat_redback_refreshed(time_arr_K, redshift, thv, loge0, thc, g1, et, s1, logn0, p, logepse, logepsb, g0, xiN, **kwargs)
kwargs = dict(frequency=freq_arr_F606W, output_format = 'flux_density')
frequency = freq_arr_F606W
afterglow_tophat_F606W = redback.transient_models.afterglow_models.tophat_redback_refreshed(time_arr_F606W, redshift, thv, loge0, thc, g1, et, s1, logn0, p, logepse, logepsb, g0, xiN, **kwargs)
kwargs = dict(frequency=freq_arr_F110W, output_format = 'flux_density')
frequency = freq_arr_F110W
afterglow_tophat_F110W = redback.transient_models.afterglow_models.tophat_redback_refreshed(time_arr_F110W, redshift, thv, loge0, thc, g1, et, s1, logn0, p, logepse, logepsb, g0, xiN, **kwargs)
kwargs = dict(frequency=freq_arr_F160W, output_format = 'flux_density')
frequency = freq_arr_F160W
afterglow_tophat_F160W = redback.transient_models.afterglow_models.tophat_redback_refreshed(time_arr_F160W, redshift, thv, loge0, thc, g1, et, s1, logn0, p, logepse, logepsb, g0, xiN, **kwargs)
kwargs = dict(frequency=freq_arr_Radio, output_format = 'flux_density')
frequency = freq_arr_Radio
afterglow_tophat_Radio = redback.transient_models.afterglow_models.tophat_redback_refreshed(time_arr_Radio, redshift, thv, loge0, thc, g1, et, s1, logn0, p, logepse, logepsb, g0, xiN, **kwargs)
kwargs = dict(frequency=freq_arr_Xray, output_format = 'flux_density')
frequency = freq_arr_Xray
afterglow_tophat_Xray = redback.transient_models.afterglow_models.tophat_redback_refreshed(time_arr_Xray, redshift, thv, loge0, thc, g1, et, s1, logn0, p, logepse, logepsb, g0, xiN, **kwargs)

print("model data:", afterglow_tophat_model)
logflux = np.log10(afterglow_tophat_model)
logtime = np.log10(time_arr)
plt.loglog(time_d, afterglow_tophat_model)
plt.xlabel('Log Time')
plt.ylabel('Log Flux Density')
plt.title('Refreshed Shock Afterglow Lightcurve Model Loge0 = 51')
plt.show()

condition = ([data["Filter"] == 'X-Ray'])
print(condition)
combine_model = afterglow_tophat_model + two_component_model 
combine_g = afterglow_tophat_g + two_component_g 
combine_i = afterglow_tophat_i + two_component_i 
combine_r = afterglow_tophat_r + two_component_r
combine_z = afterglow_tophat_z + two_component_z
combine_H = afterglow_tophat_H + two_component_H 
combine_K = afterglow_tophat_K + two_component_K
combine_F606W = afterglow_tophat_F606W + two_component_F606W 
combine_F110W = afterglow_tophat_F110W + two_component_F110W
combine_F160W = afterglow_tophat_F160W + two_component_F160W
combine_Radio = afterglow_tophat_Radio + two_component_Radio
combine_Xray = afterglow_tophat_Xray + two_component_Xray

logflux = np.log10(combine_model)
logtime = np.log10(time_arr)
plt.loglog(time_d, two_component_model, label = 'Two Component', color = 'black')
plt.loglog(time_d, afterglow_tophat_model, label = 'Tophat', color = 'grey')
plt.loglog(time_d, combine_model, linestyle = ':', label = 'Combined', color = 'black')
plt.loglog(time_arr_g, combine_g, linestyle = '--', label = 'g', color = 'purple', marker = 'o')
plt.loglog(time_arr_i, combine_i, linestyle = '--', label = 'i', color = 'green', marker = 'o')
plt.loglog(time_arr_r, combine_r, linestyle = '--', label = 'r', color = 'cyan', marker = 'o')
plt.loglog(time_arr_z, combine_z, linestyle = '--', label = 'z', color = 'yellow', marker = 'o')
plt.loglog(time_arr_H, combine_H, linestyle = '--', label = 'H', color = 'red', marker = 'o')
plt.loglog(time_arr_K, combine_K, linestyle = '--', label = 'K', color = 'magenta', marker = 'o')
plt.loglog(time_arr_F606W, combine_F606W, linestyle = '--', label = 'F606W', color = 'orange', marker = 'o')
plt.loglog(time_arr_F110W, combine_F110W, linestyle = '--', label = 'F110W', color = 'blue', marker = 'o')
plt.loglog(time_arr_F160W, combine_F160W, linestyle = '--', label = 'F160W', color = 'violet', marker = 'o')
plt.loglog(time_arr_Radio, combine_Radio, linestyle = '--', label = 'Radio', color = 'brown', marker = 'x')
plt.loglog(time_arr_Xray, combine_Xray, linestyle = '--', label = 'X-Ray', color = 'lime', marker = 'x')
plt.legend(loc = 'lower left')
plt.xlabel('Log Time (Days)')
plt.ylabel('Log Flux Density (mJy)')
plt.title('Combined Refreshed Tophat Two Component Model For 160821B (All Bands)')
plt.show()

plt.loglog(time_d, two_component_model, label = 'Two Component', color = 'black')
plt.loglog(time_d, afterglow_tophat_model, label = 'Tophat', color = 'grey')
plt.loglog(time_d, combine_model, linestyle = ':', label = 'Combined', color = 'black')
plt.loglog(time_arr_g, combine_g, linestyle = '--', label = 'g', color = 'purple', marker = 'o')
plt.loglog(time_arr_i, combine_i, linestyle = '--', label = 'i', color = 'green', marker = 'o')
plt.loglog(time_arr_r, combine_r, linestyle = '--', label = 'r', color = 'cyan', marker = 'o')
plt.loglog(time_arr_z, combine_z, linestyle = '--', label = 'z', color = 'yellow', marker = 'o')
plt.loglog(time_arr_H, combine_H, linestyle = '--', label = 'H', color = 'red', marker = 'o')
plt.loglog(time_arr_K, combine_K, linestyle = '--', label = 'K', color = 'magenta', marker = 'o')
plt.loglog(time_arr_F606W, combine_F606W, linestyle = '--', label = 'F606W', color = 'orange', marker = 'o')
plt.loglog(time_arr_F110W, combine_F110W, linestyle = '--', label = 'F110W', color = 'blue', marker = 'o')
plt.loglog(time_arr_F160W, combine_F160W, linestyle = '--', label = 'F160W', color = 'violet', marker = 'o')
plt.legend(loc = 'lower left')
plt.xlabel('Log Time (Days)')
plt.ylabel('Log Flux Density (mJy)')
plt.title('Combined Refreshed Tophat Two Component Model For 160821B (Excluding Radio And X-Ray)')
plt.show()

plt.loglog(time_d, two_component_model, label = 'Two Component', color = 'black')
plt.loglog(time_d, afterglow_tophat_model, label = 'Tophat', color = 'grey')
plt.loglog(time_d, combine_model, linestyle = ':', label = 'Combined', color = 'black')
plt.loglog(time_arr_Radio, combine_Radio, linestyle = '--', label = 'Radio', color = 'brown', marker = 'x')
plt.loglog(time_arr_Xray, combine_Xray, linestyle = '--', label = 'X-Ray', color = 'lime', marker = 'x')
plt.legend(loc = 'lower left')
plt.xlabel('Log Time (Days)')
plt.ylabel('Log Flux Density (mJy)')
plt.title('Combined Refreshed Tophat Two Component Model For 160821B (Radio And X-Ray)')
plt.show()
#%%
# Quick visualisation of model and raw data
data = pd.read_csv('/Users/harrymccabe/Documents/PHYS498 Masters Project/160821B.csv')
z = data['Frequency'].values
plt.scatter(time_d, flux_density, label = 'Data', c = z, marker = 'o', cmap = 'Blues')
plt.scatter(time_d, combine_model, label = 'Combined Model', c = z, marker = 'x', cmap = 'Blues')

plt.legend()
plt.xlabel('Time (Days)')
plt.ylabel('Flux Density (mJy)')
plt.title('Model Vs Raw Flux Data')
plt.show()
#%%
# THIS TAKES A VERY LONG TIME TO RUN!!! #
# THIS TAKES A VERY LONG TIME TO RUN!!! #
# THIS TAKES A VERY LONG TIME TO RUN!!! #
import redback
import pandas as pd
from bilby.core.prior import LogUniform, Uniform
import bilby
import matplotlib as plt
import matplotlib as mpl
mpl.rcParams.update(mpl.rcParamsDefault)
import numpy as np

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



def model_func(time, redshift, thv, loge0, thc, g1, et, s1, logn0, p, logepse, logepsb, g0, xiN, mej_1, vej_1, temperature_floor_1, kappa_1, mej_2, vej_2, temperature_floor_2, kappa_2, **kwargs):
    
    AG_model = redback.transient_models.afterglow_models.tophat_redback_refreshed(time, redshift, thv, loge0, thc, g1, et, s1, logn0, p, logepse, logepsb, g0, xiN, **kwargs)
    KN_model = redback.transient_models.kilonova_models.two_component_kilonova_model(time, redshift, mej_1, vej_1, temperature_floor_1, kappa_1, mej_2, vej_2, temperature_floor_2, kappa_2, **kwargs)
    combine = AG_model + KN_model
    return combine #, AG_model, KN_model

data = pd.read_csv('/Users/harrymccabe/Documents/PHYS498 Masters Project/160821B copy.csv') #this is a pandas DataFrame
df = pd.DataFrame(data=data)
time_d = data['DeltaT'].values #creates an array of the values in time column
time_err = data['Time Error'].values
data_filter = data['Filter'].values
AB_mag = data['magnitude'].values
flux_density = data['flux_density'].values #unit is mJy
flux_density_err = data['Flux Density Error'].values
frequency = redback.utils.bands_to_frequency(data_filter)

kwargs = dict(frequency=frequency, output_format='flux_density')
name = '160821B'

afterglow = redback.transient.Afterglow(name=name, data_mode='flux_density', time=time_d, flux_density=flux_density, flux_density_err=flux_density_err, frequency=frequency)

model_data = model_func(time_d, 0.162, 0.0, 51.3, 0.1, 10, 20, 12, -4, 2.3, -1, -2, 100, 0.1, 0.001, 0.2, 2500, 1, 0.01, 0.1, 2500, 10, **kwargs)


nlive = 10000
sampler = 'emcee'

priors = bilby.core.prior.PriorDict()   
priors['redshift'] = 0.162
priors['thv'] = 0.0
priors['loge0'] = Uniform(49, 53, 'loge0', latex_label='loge0')
priors['thc'] = Uniform(0.02, 0.2, 'thc', latex_label='thc')
priors['logn0'] = Uniform(-6, 1, 'logn0', latex_label='logn0')
priors['p'] = Uniform(1.5, 3.1, 'p', latex_label='p')
priors['logepse'] = Uniform(-6, -0.5, 'logepse', latex_label='logepse')
priors['logepsb'] = Uniform(-6, -0.5, 'logepsb', latex_label='logepsb')
priors['g0'] = Uniform(50, 150, 'g0', latex_label='g0')
priors['xiN'] = Uniform(0.01, 1, 'xiN', latex_label='xiN')
priors['mej_1'] = Uniform(0.0005, 0.0015, 'mej_1', latex_label='mej_1')
priors['vej_1'] = Uniform(0.1, 0.3, 'vej_1', latex_label='vej_1')
priors['temperature_floor_1'] = LogUniform(1500, 3500, 'temperature_floor_1', latex_label='temperature_floor_1')
priors['kappa_1'] = Uniform(1, 20, 'kappa_1', latex_label='kappa_1')
priors['mej_2'] = Uniform(0.005, 0.015, 'mej_2', latex_label='mej_2')
priors['vej_2'] = Uniform(0.01, 0.2, 'vej_2', latex_label='vej_2')
priors['temperature_floor_2'] = LogUniform(1500, 3500, 'temperature_floor_2', latex_label='temperature_floor_2')
priors['kappa_2'] = Uniform(0.1, 2, 'kappa_2', latex_label='kappa_2')
priors['g1'] = Uniform(5, 15, 'g1', latex_label='g1')
priors['et'] = Uniform(15, 25, 'et', latex_label='et')
priors['s1'] = Uniform(9, 15, 's1', latex_label='s1')

model = model_func
 
model_kwargs = kwargs 

result = redback.fit_model(model=model, sampler=sampler, nlive=nlive, transient=afterglow,
                           model_kwargs=model_kwargs, prior=priors, nburn = 1000, resume=True, clean = True)
result.plot_lightcurve(random_models=100, model=model_func)
result.plot_multiband_lightcurve(model = model_func)
result.plot_corner(save=True)
plt.show()
#%%
# THIS TAKES A VERY LONG TIME TO RUN!!! #
# THIS TAKES A VERY LONG TIME TO RUN!!! #
# THIS TAKES A VERY LONG TIME TO RUN!!! #
# Sampling code #
import redback
import pandas as pd
from bilby.core.prior import LogUniform, Uniform
import bilby
import matplotlib as mpl
mpl.rcParams.update(mpl.rcParamsDefault)
import numpy as np

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



def model_func(time, redshift, thv, loge0, thc, logn0, p, logepse, logepsb, g0, xiN, mej_1, vej_1, temperature_floor_1, kappa_1, mej_2, vej_2, temperature_floor_2, kappa_2, **kwargs):
    
    AG_model = redback.transient_models.afterglow_models.tophat_redback(time, redshift, thv, loge0, thc, logn0, p, logepse, logepsb, g0, xiN, **kwargs)
    KN_model = redback.transient_models.kilonova_models.two_component_kilonova_model(time, redshift, mej_1, vej_1, temperature_floor_1, kappa_1, mej_2, vej_2, temperature_floor_2, kappa_2, **kwargs)
    #print(KN_model)
    combine = AG_model + KN_model
    return combine #, AG_model, KN_model

data = pd.read_csv('/Users/harrymccabe/Documents/PHYS498 Masters Project/160821B.csv') #Change as needed to your path
time_d = data['DeltaT'].values #creates an array of the values in time column
time_err = data['Time Error'].values
data_filter = data['Filter'].values
AB_mag = data['magnitude'].values
flux_density = data['flux_density'].values #unit is mJy
flux_density_err = data['Flux Density Error'].values
frequency = data['Frequency'].values


thv = 0.0
redshift = 0.162

kwargs = dict(frequency=frequency, output_format='flux_density')
data_mode = 'flux_density'
name = '160821B'

afterglow = redback.transient.Afterglow(name=name, data_mode='flux_density', time=time_d, flux_density=flux_density, flux_density_err=flux_density_err, frequency=frequency)

model_data = model_func(time_d, 0.162, 0.0, 51.3, 0.1, -4, 2.3, -1, -2, 100, 0.1, 0.001, 0.2, 2500, 10, 0.01, 0.1, 2500, 1, **kwargs)


nlive = 10000
sampler = 'emcee'

priors = bilby.core.prior.PriorDict()   
priors['redshift'] = 0.162
priors['thv'] = 0.0
priors['loge0'] = Uniform(49, 53, 'loge0', latex_label='loge0')
priors['thc'] = Uniform(0.02, 0.2, 'thc', latex_label='thc')
priors['logn0'] = Uniform(-6, 1, 'logn0', latex_label='logn0')
priors['p'] = Uniform(1.5, 3.1, 'p', latex_label='p')
priors['logepse'] = Uniform(-6, -0.5, 'logepse', latex_label='logepse')
priors['logepsb'] = Uniform(-6, -0.5, 'logepsb', latex_label='logepsb')
priors['g0'] = Uniform(50, 150, 'g0', latex_label='g0')
priors['xiN'] = Uniform(0.01, 1, 'xiN', latex_label='xiN')
priors['mej_1'] = Uniform(0.0005, 0.0015, 'mej_1', latex_label='mej_1')
priors['vej_1'] = Uniform(0.1, 0.3, 'vej_1', latex_label='vej_1')
priors['temperature_floor_1'] = LogUniform(1500, 3500, 'temperature_floor_1', latex_label='temperature_floor_1')
priors['kappa_1'] = Uniform(1, 20, 'kappa_1', latex_label='kappa_1')
priors['mej_2'] = Uniform(0.005, 0.015, 'mej_2', latex_label='mej_2')
priors['vej_2'] = Uniform(0.01, 0.2, 'vej_2', latex_label='vej_2')
priors['temperature_floor_2'] = LogUniform(1500, 3500, 'temperature_floor_2', latex_label='temperature_floor_2')
priors['kappa_2'] = Uniform(0.1, 2, 'kappa_2', latex_label='kappa_2')


model = model_func

model_kwargs = kwargs 

result = redback.fit_model(name = name, model=model, sampler=sampler, nlive=nlive, transient=afterglow,
                           model_kwargs=model_kwargs, prior=priors, nburn = 1000, live_dangerously = True, sample='rslice', resume=True, clean = True, data_mode = 'flux_density')
result.plot_lightcurve(random_models=100, model=model_func)
#%%
# TOP HAT SAMPLING PLOTTER
final = redback.result.read_in_result('/Users/harrymccabe/Documents/PHYS498 Masters Project/GRB160821B_result_TH.json')
def model_func(time, redshift, thv, loge0, thc, logn0, p, logepse, logepsb, g0, xiN, mej_1, vej_1, temperature_floor_1, kappa_1, mej_2, vej_2, temperature_floor_2, kappa_2, **kwargs):
    
    AG_model = redback.transient_models.afterglow_models.tophat_redback(time, redshift, thv, loge0, thc, logn0, p, logepse, logepsb, g0, xiN, **kwargs)
    KN_model = redback.transient_models.kilonova_models.two_component_kilonova_model(time, redshift, mej_1, vej_1, temperature_floor_1, kappa_1, mej_2, vej_2, temperature_floor_2, kappa_2, **kwargs)
    combine = AG_model + KN_model
    return combine 

fig1, ax1 = plt.subplots(1,1)
ax1.set_xscale('log')
final.plot_lightcurve(figure = fig1, axes = ax1, random_models = 100, model = model_func, priors = False, show = True, fontsize_legend = 9, legend_location = 'upper right')
final.plot_multiband_lightcurve(model = model_func, axes = ax1)
#%%
# REFRESHED SHOCK SAMPLING PLOTTER
final = redback.result.read_in_result('/Users/harrymccabe/Documents/PHYS498 Masters Project/GRB160821B_result_rTH.json')
def model_func(time, redshift, thv, loge0, thc, g1, et, s1, logn0, p, logepse, logepsb, g0, xiN, mej_1, vej_1, temperature_floor_1, kappa_1, mej_2, vej_2, temperature_floor_2, kappa_2, **kwargs):
    
    AG_model = redback.transient_models.afterglow_models.tophat_redback_refreshed(time, redshift, thv, loge0, thc, g1, et, s1, logn0, p, logepse, logepsb, g0, xiN, **kwargs)
    KN_model = redback.transient_models.kilonova_models.two_component_kilonova_model(time, redshift, mej_1, vej_1, temperature_floor_1, kappa_1, mej_2, vej_2, temperature_floor_2, kappa_2, **kwargs)
    combine = AG_model + KN_model
    return combine 

fig1, ax1 = plt.subplots(1,1)
ax1.set_xscale('log')
final.plot_lightcurve(figure = fig1, axes = ax1, random_models = 100, model = model_func, prior = False, show = True, fontsize_legend = 9, legend_location = 'upper right')
final.plot_multiband_lightcurve(model = model_func, axes = ax1)
#%%
# THIS IS FOR THE REFRESHED SHOCK 100 PLOTS

import json
import numpy as np
import redback
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
time_arr = np.logspace(-2, 1, num = 100)
freq_arr = np.ones(len(time_arr))*4.86e14
freq_arr_2 = np.ones(len(time_arr))*2.4e17
freq_arr_3 = np.ones(len(time_arr))*2.73e14
freq_arr_4 = np.ones(len(time_arr))*1.88e14
freq_arr_5 = np.ones(len(time_arr))*4.95e14
freq_arr_6 = np.ones(len(time_arr))*6.32e14
freq_arr_7 = np.ones(len(time_arr))*1.88e14
freq_arr_8 = np.ones(len(time_arr))*1.36e14
freq_arr_9 = np.ones(len(time_arr))*9.80e9
freq_arr_10 = np.ones(len(time_arr))*3.46e14
freq_arr_11 = np.ones(len(time_arr))*3.83e14

redshift = 0.162
thv = 0.0
data = pd.read_csv('/Users/harrymccabe/Documents/PHYS498 Masters Project/160821B.csv') #Change as needed to your path
time_d = data['DeltaT'].values #creates an array of the values in time column
time_err = data['Time Error'].values
data_filter = data['Filter'].values
AB_mag = data['magnitude'].values
flux_density = data['flux_density'].values #unit is mJy
flux_density_err = data['Flux Density Error'].values
frequency = data['Frequency'].values
# Load the JSON file containing the sampled parameters
final = redback.result.read_in_result('/Users/harrymccabe/Documents/PHYS498 Masters Project/GRB160821B_result_rTH.json')
kwargs = dict(frequency=freq_arr, output_format='flux_density')
kwargs2 = dict(frequency=freq_arr_2, output_format='flux_density')
kwargs3 = dict(frequency=freq_arr_3, output_format='flux_density')
kwargs4 = dict(frequency=freq_arr_4, output_format='flux_density')
kwargs5 = dict(frequency=freq_arr_5, output_format='flux_density')
kwargs6 = dict(frequency=freq_arr_6, output_format='flux_density')
kwargs7 = dict(frequency=freq_arr_7, output_format='flux_density')
kwargs8 = dict(frequency=freq_arr_8, output_format='flux_density')
kwargs9 = dict(frequency=freq_arr_9, output_format='flux_density')
kwargs10 = dict(frequency=freq_arr_10, output_format='flux_density')
kwargs11 = dict(frequency=freq_arr_11, output_format='flux_density')
pos = final.samples
col = final.parameter_labels
df = pd.DataFrame(pos, columns = col)
print(df)
def model_func(time, redshift, thv, loge0, thc, g1, et, s1, logn0, p, logepse, logepsb, g0, xiN, mej_1, vej_1, temperature_floor_1, kappa_1, mej_2, vej_2, temperature_floor_2, kappa_2, **kwargs):
 
    AG_model = redback.transient_models.afterglow_models.tophat_redback_refreshed(time, redshift, thv, loge0, thc, g1, et, s1, logn0, p, logepse, logepsb, g0, xiN, **kwargs)
    KN_model = redback.transient_models.kilonova_models.two_component_kilonova_model(time, redshift, mej_1, vej_1, temperature_floor_1, kappa_1, mej_2, vej_2, temperature_floor_2, kappa_2, **kwargs)
    combine = AG_model + KN_model
    return combine

# Step 1: Randomly select 100 rows from the DataFrame
selected_rows = df.sample(n=10, random_state = 69)  # Can fix random_state if needed

# Step 2: Extract parameter values from the selected rows
parameter_sets = selected_rows.to_dict(orient='records')

# Step 3: Compute model output for each parameter set and store in a list
model_outputs = []
model_outputs2 = []
model_outputs3 = []
model_outputs4 = []
model_outputs5 = []
model_outputs6 = []
model_outputs7 = []
model_outputs8 = []
model_outputs9 = []
model_outputs10 = []
model_outputs11 = []

for params in parameter_sets:
    params.update(kwargs)
    output = model_func(time_arr, redshift, thv, **params)
    model_outputs.append(output)
for params in parameter_sets:
    params.update(kwargs2)
    output2 = model_func(time_arr, redshift, thv, **params)
    model_outputs2.append(output2)
for params in parameter_sets:
    params.update(kwargs3)
    output3 = model_func(time_arr, redshift, thv, **params)
    model_outputs3.append(output3)
for params in parameter_sets:
    params.update(kwargs4)
    output4 = model_func(time_arr, redshift, thv, **params)
    model_outputs4.append(output4)
for params in parameter_sets:
    params.update(kwargs5)
    output5 = model_func(time_arr, redshift, thv, **params)
    model_outputs5.append(output5)
for params in parameter_sets:
    params.update(kwargs6)
    output6 = model_func(time_arr, redshift, thv, **params)
    model_outputs6.append(output6)
for params in parameter_sets:
    params.update(kwargs7)
    output7 = model_func(time_arr, redshift, thv, **params)
    model_outputs7.append(output7)
for params in parameter_sets:
    params.update(kwargs8)
    output8 = model_func(time_arr, redshift, thv, **params)
    model_outputs8.append(output8)
for params in parameter_sets:
    params.update(kwargs9)
    output9 = model_func(time_arr, redshift, thv, **params)
    model_outputs9.append(output9)
for params in parameter_sets:
    params.update(kwargs10)
    output10 = model_func(time_arr, redshift, thv, **params)
    model_outputs2.append(output10)
for params in parameter_sets:
    params.update(kwargs11)
    output11 = model_func(time_arr, redshift, thv, **params)
    model_outputs11.append(output11)
# Step 4: Plot the results on the same graph
custom_handles = [
    Line2D([0], [0], color='violet', lw=2),
    Line2D([0], [0], color='grey', lw=2),
    Line2D([0], [0], color='green', lw=2),
    Line2D([0], [0], color='orange', lw=2),
    Line2D([0], [0], color='darkviolet', lw=2),
    Line2D([0], [0], color='purple', lw=2),
    Line2D([0], [0], color='yellow', lw=2),
    Line2D([0], [0], color='red', lw=2),
    Line2D([0], [0], color='grey', linestyle='--', lw=2),
    Line2D([0], [0], color='indigo', lw=2),
    Line2D([0], [0], color='blue', lw=2),
    Line2D([0], [0], color='black', lw=2, marker = 'o')
]

plt.figure(figsize=(12, 8))
for output in model_outputs:
    plt.loglog(time_arr, output, alpha=0.2, color = 'violet')
for output2 in model_outputs2:
    plt.loglog(time_arr, output2, alpha=0.2, color = 'grey')
for output3 in model_outputs3:
    plt.loglog(time_arr, output3, alpha=0.2, color = 'green')
for output4 in model_outputs4:
    plt.loglog(time_arr, output4, alpha=0.2, color = 'orange')
for output5 in model_outputs5:
    plt.loglog(time_arr, output5, alpha=0.2, color = 'darkviolet')
for output6 in model_outputs6:
    plt.loglog(time_arr, output6, alpha=0.2, color = 'purple')
for output7 in model_outputs7:
    plt.loglog(time_arr, output7, alpha=0.2, color = 'yellow')
for output8 in model_outputs8:
    plt.loglog(time_arr, output8, alpha=0.2, color = 'red')
for output9 in model_outputs9:
    plt.loglog(time_arr, output9, alpha=0.2, color = 'grey', linestyle = '--')
for output10 in model_outputs10:
    plt.loglog(time_arr, output10, alpha=0.2, color = 'indigo')
for output11 in model_outputs11:
    plt.loglog(time_arr, output11, alpha=0.2, color = 'blue')
plt.errorbar(time_d, flux_density, flux_density_err, color = 'black', linestyle = '', marker = 'o')
plt.xlabel('Time In Days [Log Scale]')
plt.ylabel('Flux In mJy [Log Scale]')
plt.title('Model Outputs for 100 Random Parameter Sets (Refreshed Top Hat & Two Component Kilonova)')
plt.ylim(1e-7, 1)# Cut out x-ray lines that are way off
plt.xlim(0.01, 10)
plt.legend(handles = custom_handles ,labels = ['r','X-Ray','F110W','F160W','F606W','g','H','K','Radio','z','i', 'Data'], loc = 'lower left', fontsize = 11, ncol = 2)
plt.show()
#%%

# THIS IS FOR THE TOP HAT SHOCK 100 PLOTS

import json
import numpy as np
import redback
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
time_arr = np.logspace(-2, 1, num = 100)
freq_arr = np.ones(len(time_arr))*4.86e14
freq_arr_2 = np.ones(len(time_arr))*2.4e17
freq_arr_3 = np.ones(len(time_arr))*2.73e14
freq_arr_4 = np.ones(len(time_arr))*1.88e14
freq_arr_5 = np.ones(len(time_arr))*4.95e14
freq_arr_6 = np.ones(len(time_arr))*6.32e14
freq_arr_7 = np.ones(len(time_arr))*1.88e14
freq_arr_8 = np.ones(len(time_arr))*1.36e14
freq_arr_9 = np.ones(len(time_arr))*9.80e9
freq_arr_10 = np.ones(len(time_arr))*3.46e14
freq_arr_11 = np.ones(len(time_arr))*3.83e14

redshift = 0.162
thv = 0.0
data = pd.read_csv('/Users/harrymccabe/Documents/PHYS498 Masters Project/160821B.csv') #Change as needed to your path
time_d = data['DeltaT'].values #creates an array of the values in time column
time_err = data['Time Error'].values
data_filter = data['Filter'].values
AB_mag = data['magnitude'].values
flux_density = data['flux_density'].values #unit is mJy
flux_density_err = data['Flux Density Error'].values
frequency = data['Frequency'].values
# Load the JSON file containing the sampled parameters
final = redback.result.read_in_result('/Users/harrymccabe/Documents/PHYS498 Masters Project/GRB160821B_result_TH.json')
kwargs = dict(frequency=freq_arr, output_format='flux_density')
kwargs2 = dict(frequency=freq_arr_2, output_format='flux_density')
kwargs3 = dict(frequency=freq_arr_3, output_format='flux_density')
kwargs4 = dict(frequency=freq_arr_4, output_format='flux_density')
kwargs5 = dict(frequency=freq_arr_5, output_format='flux_density')
kwargs6 = dict(frequency=freq_arr_6, output_format='flux_density')
kwargs7 = dict(frequency=freq_arr_7, output_format='flux_density')
kwargs8 = dict(frequency=freq_arr_8, output_format='flux_density')
kwargs9 = dict(frequency=freq_arr_9, output_format='flux_density')
kwargs10 = dict(frequency=freq_arr_10, output_format='flux_density')
kwargs11 = dict(frequency=freq_arr_11, output_format='flux_density')
pos = final.samples
col = final.parameter_labels
df = pd.DataFrame(pos, columns = col)
print(df)
def model_func(time, redshift, thv, loge0, thc, logn0, p, logepse, logepsb, g0, xiN, mej_1, vej_1, temperature_floor_1, kappa_1, mej_2, vej_2, temperature_floor_2, kappa_2, **kwargs):
 
    AG_model = redback.transient_models.afterglow_models.tophat_redback(time, redshift, thv, loge0, thc, logn0, p, logepse, logepsb, g0, xiN, **kwargs)
    KN_model = redback.transient_models.kilonova_models.two_component_kilonova_model(time, redshift, mej_1, vej_1, temperature_floor_1, kappa_1, mej_2, vej_2, temperature_floor_2, kappa_2, **kwargs)
    combine = AG_model + KN_model
    return combine

# Step 1: Randomly select 100 rows from the DataFrame
selected_rows = df.sample(n=10)  # Can fix random_state if needed

# Step 2: Extract parameter values from the selected rows
parameter_sets = selected_rows.to_dict(orient='records')

# Step 3: Compute model output for each parameter set and store in a list
model_outputs = []
model_outputs2 = []
model_outputs3 = []
model_outputs4 = []
model_outputs5 = []
model_outputs6 = []
model_outputs7 = []
model_outputs8 = []
model_outputs9 = []
model_outputs10 = []
model_outputs11 = []

for params in parameter_sets:
    params.update(kwargs)
    output = model_func(time_arr, redshift, thv, **params)
    model_outputs.append(output)
for params in parameter_sets:
    params.update(kwargs2)
    output2 = model_func(time_arr, redshift, thv, **params)
    model_outputs2.append(output2)
for params in parameter_sets:
    params.update(kwargs3)
    output3 = model_func(time_arr, redshift, thv, **params)
    model_outputs3.append(output3)
for params in parameter_sets:
    params.update(kwargs4)
    output4 = model_func(time_arr, redshift, thv, **params)
    model_outputs4.append(output4)
for params in parameter_sets:
    params.update(kwargs5)
    output5 = model_func(time_arr, redshift, thv, **params)
    model_outputs5.append(output5)
for params in parameter_sets:
    params.update(kwargs6)
    output6 = model_func(time_arr, redshift, thv, **params)
    model_outputs6.append(output6)
for params in parameter_sets:
    params.update(kwargs7)
    output7 = model_func(time_arr, redshift, thv, **params)
    model_outputs7.append(output7)
for params in parameter_sets:
    params.update(kwargs8)
    output8 = model_func(time_arr, redshift, thv, **params)
    model_outputs8.append(output8)
for params in parameter_sets:
    params.update(kwargs9)
    output9 = model_func(time_arr, redshift, thv, **params)
    model_outputs9.append(output9)
for params in parameter_sets:
    params.update(kwargs10)
    output10 = model_func(time_arr, redshift, thv, **params)
    model_outputs2.append(output10)
for params in parameter_sets:
    params.update(kwargs11)
    output11 = model_func(time_arr, redshift, thv, **params)
    model_outputs11.append(output11)
# Step 4: Plot the results on the same graph
custom_handles = [
    Line2D([0], [0], color='violet', lw=2),
    Line2D([0], [0], color='grey', lw=2),
    Line2D([0], [0], color='green', lw=2),
    Line2D([0], [0], color='orange', lw=2),
    Line2D([0], [0], color='darkviolet', lw=2),
    Line2D([0], [0], color='purple', lw=2),
    Line2D([0], [0], color='yellow', lw=2),
    Line2D([0], [0], color='red', lw=2),
    Line2D([0], [0], color='grey', linestyle='--', lw=2),
    Line2D([0], [0], color='indigo', lw=2),
    Line2D([0], [0], color='blue', lw=2),
    Line2D([0], [0], color='black', lw=2, marker = 'o')
]

plt.figure(figsize=(12, 8))
for output in model_outputs:
    plt.loglog(time_arr, output, alpha=0.2, color = 'violet')
for output2 in model_outputs2:
    plt.loglog(time_arr, output2, alpha=0.2, color = 'grey')
for output3 in model_outputs3:
    plt.loglog(time_arr, output3, alpha=0.2, color = 'green')
for output4 in model_outputs4:
    plt.loglog(time_arr, output4, alpha=0.2, color = 'orange')
for output5 in model_outputs5:
    plt.loglog(time_arr, output5, alpha=0.2, color = 'darkviolet')
for output6 in model_outputs6:
    plt.loglog(time_arr, output6, alpha=0.2, color = 'purple')
for output7 in model_outputs7:
    plt.loglog(time_arr, output7, alpha=0.2, color = 'yellow')
for output8 in model_outputs8:
    plt.loglog(time_arr, output8, alpha=0.2, color = 'red')
for output9 in model_outputs9:
    plt.loglog(time_arr, output9, alpha=0.2, color = 'grey', linestyle = '--')
for output10 in model_outputs10:
    plt.loglog(time_arr, output10, alpha=0.2, color = 'indigo')
for output11 in model_outputs11:
    plt.loglog(time_arr, output11, alpha=0.2, color = 'blue')
plt.errorbar(time_d, flux_density, flux_density_err, color = 'black', linestyle = '', marker = 'o')
plt.xlabel('Time In Days [Log Scale]')
plt.ylabel('Flux In mJy [Log Scale]')
plt.title('Model Outputs for 100 Random Parameter Sets (Top Hat & Two Component Kilonova)')
plt.ylim(1e-7, 1)# Cut out x-ray lines that are way off
plt.xlim(0.01, 10)
plt.legend(handles = custom_handles ,labels = ['r','X-Ray','F110W','F160W','F606W','g','H','K','Radio','z','i', 'Data'], loc = 'lower left', fontsize = 11, ncol = 2)
plt.show()
#%%

# Corner plots, can be a slow takes a few mins

import redback
import pandas as pd
import numpy as np
from bilby.core.result import Result
from bilby.core.result import _determine_file_name # noqa
import matplotlib.pyplot as plt

def model_func_TH(time, redshift, thv, loge0, thc, logn0, p, logepse, logepsb, g0, xiN, mej_1, vej_1, temperature_floor_1, kappa_1, mej_2, vej_2, temperature_floor_2, kappa_2, **kwargs):
    
    AG_model = redback.transient_models.afterglow_models.tophat_redback(time, redshift, thv, loge0, thc, logn0, p, logepse, logepsb, g0, xiN, **kwargs)
    KN_model = redback.transient_models.kilonova_models.two_component_kilonova_model(time, redshift, mej_1, vej_1, temperature_floor_1, kappa_1, mej_2, vej_2, temperature_floor_2, kappa_2, **kwargs)
    combine = AG_model + KN_model
    return combine

def model_func_rTH(time, redshift, thv, loge0, thc, g1, et, s1, logn0, p, logepse, logepsb, g0, xiN, mej_1, vej_1, temperature_floor_1, kappa_1, mej_2, vej_2, temperature_floor_2, kappa_2, **kwargs):
    
    AG_model = redback.transient_models.afterglow_models.tophat_redback_refreshed(time, redshift, thv, loge0, thc, g1, et, s1, logn0, p, logepse, logepsb, g0, xiN, **kwargs)
    KN_model = redback.transient_models.kilonova_models.two_component_kilonova_model(time, redshift, mej_1, vej_1, temperature_floor_1, kappa_1, mej_2, vej_2, temperature_floor_2, kappa_2, **kwargs)
    combine = AG_model + KN_model
    return combine 

result_TH = redback.result.read_in_result("/Users/harrymccabe/Documents/PHYS498 Masters Project/GRB160821B_result_TH.json")
print('Done TH')

result_rTH = redback.result.read_in_result("/Users/harrymccabe/Documents/PHYS498 Masters Project/GRB160821B_result_rTH.json")
print('Done rTH')

result_TH.plot_corner(burn = 1000, thin = 100, smooth = True,  show=True, outdir = '/Users/harrymccabe/Documents/PHYS498 Masters Project')

result_rTH.plot_corner(burn = 1000, thin = 100, smooth = True,  show=True, outdir = '/Users/harrymccabe/Documents/PHYS498 Masters Project')

