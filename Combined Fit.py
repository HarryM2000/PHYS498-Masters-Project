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
sigma = 1.35e-29
redshift = 0.162
av = 0.118
thv = 0.0
loge0 = 51
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
logtime = np.log10(time_d)
time_arr = np.logspace(-2, 2, num = 39)
freq_arr = np.ones(len(time_arr))*1e14
print(time_arr)
time_arr_two = np.linspace(0.01, 30, 39)
time_arr_four = 10**(np.linspace(-2, 1.5, num = 39))
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
freq_arr_Radio = np.ones(len(time_arr_Radio))*7.5e9
Xray_band_data = data[data["Filter"] == 'X-Ray']
time_arr_Xray = (Radio_band_data["DeltaT"].values)
freq_arr_Xray = np.ones(len(time_arr_Radio))*1.2e18

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
plt.legend()
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
plt.legend()
plt.xlabel('Log Time (Days)')
plt.ylabel('Log Flux Density (mJy)')
plt.title('Combined Refreshed Tophat Two Component Model For 160821B (Excluding Radio And X-Ray)')
plt.show()

plt.loglog(time_d, two_component_model, label = 'Two Component', color = 'black')
plt.loglog(time_d, afterglow_tophat_model, label = 'Tophat', color = 'grey')
plt.loglog(time_d, combine_model, linestyle = ':', label = 'Combined', color = 'black')
plt.loglog(time_arr_Radio, combine_Radio, linestyle = '--', label = 'Radio', color = 'brown', marker = 'x')
plt.loglog(time_arr_Xray, combine_Xray, linestyle = '--', label = 'X-Ray', color = 'lime', marker = 'x')
plt.legend()
plt.xlabel('Log Time (Days)')
plt.ylabel('Log Flux Density (mJy)')
plt.title('Combined Refreshed Tophat Two Component Model For 160821B (Radio And X-Ray)')
plt.show()
#%%
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
import redback
import pandas as pd
from bilby.core.prior import LogUniform, Uniform
import bilby
import matplotlib.pyplot as plt
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

data = pd.read_csv('/Users/harrymccabe/Documents/PHYS498 Masters Project/160821B.csv') #this is a pandas DataFrame
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

model_data = model_func(time_d, 0.162, 0.0, 51.3, 0.1, 10, 10, 6, -4, 2.3, -1, -2, 100, 0.1, 0.001, 0.2, 2500, 10, 0.01, 0.1, 2500, 1, **kwargs)

plt.loglog(time_d, flux_density, 'o', color='blue', markersize=6, alpha=0.2)
plt.loglog(time_d, model_data, 's', color='red', markersize=6, alpha=0.2)

plt.show()

nlive = 200 
sampler = 'dynesty'


priors = bilby.core.prior.PriorDict()   
priors['redshift'] = 0.162
priors['thv'] = 0.0
priors['thc'] = Uniform(0, 0.2, 'thc', latex_label=r'$\thc$')
priors['loge0'] = Uniform(46, 54, 'loge0', latex_label=r'$\loge0$')
priors['logn0'] = Uniform(-8, 0, 'logn0', latex_label=r'$\logn0$')
priors['p'] = Uniform(1.3, 3.3, 'p', latex_label=r'$\p$')
priors['logepse'] = Uniform(-2, 0, 'logepse', latex_label=r'$\logepse$')
priors['logepsb'] = Uniform(-3, -1, 'logepsb', latex_label=r'$\logepsb$')
priors['ksin'] = Uniform(0, 0.6, 'ksin', latex_label=r'$\ksin$')
priors['g0'] = Uniform(50, 150, 'g0', latex_label=r'$\g0$')
priors['mej_1'] = Uniform(0.0005, 0.0015, 'mej_1', latex_label=r'$\mej_1$')
priors['vej_1'] = Uniform(0.1, 0.3, 'vej_1', latex_label=r'$\vej_1$')
priors['temperature_floor_1'] = LogUniform(1500, 3500, 'temperature_floor_1', latex_label=r'$\temperature_floor_1$')
priors['kappa_1'] = Uniform(0, 20, 'kappa_1', latex_label=r'$\kappa_1$')
priors['mej_2'] = Uniform(0.005, 0.015, 'mej_2', latex_label=r'$\mej_2$')
priors['vej_2'] = Uniform(0.0, 0.2, 'vej_2', latex_label=r'$\vej_2$')
priors['temperature_floor_2'] = LogUniform(1500, 3500, 'temperature_floor_2', latex_label=r'$\temperature_floor_2$')
priors['kappa_2'] = Uniform(0, 2, 'kappa_2', latex_label=r'$\kappa_2$')
priors['xiN'] = Uniform(0, 0.3, 'xiN', latex_label=r'$\xiN$')
priors['g1'] = Uniform(5, 15, 'g1', latex_label=r'$\g1$')
priors['et'] = Uniform(10, 30, 'et', latex_label=r'$\et$')
priors['s1'] = Uniform(6, 18, 's1', latex_label=r'$\s1$')

model = model_func

model_kwargs = kwargs 

result = redback.fit_model(model=model, sampler=sampler, nlive=nlive, transient=afterglow,
                           model_kwargs=model_kwargs, prior=priors, sample='rslice', resume=True)
result.plot_lightcurve(random_models=100, model=model_func)