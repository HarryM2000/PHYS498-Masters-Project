# https://link.aps.org/accepted/10.1103/PhysRevD.97.083013
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

redshift = 0.0099 # Speaks for itself really
mej_dyn = 0.001 # Dynamical mass of ejecta in solar massees
mej_disk = 0.01 # Disk mass of ejecta in solar masses
phi = 30 # Half opening angle of the lanthanide rich tidal dynamic ejecta in degrees
costheta_obs = 1 # Cosine of the observers viewing angle   
vej = 0.1
kappa = 10
sampler = 'nessai'
model = 'one_component_kilonova_model'

kne = 'at2017gfo'
# gets the magnitude data for AT2017gfo, the KN associated with GW170817
data = redback.get_data.get_kilonova_data_from_open_transient_catalog_data(transient=kne)
print(data)
time_d = data[('time (days)')].values

# creates a GRBDir with GRB
kilonova = redback.kilonova.Kilonova.from_open_access_catalogue(
    name=kne, data_mode="flux_density")
kilonova.plot_data(show=False)
fig, axes = plt.subplots(3, 2, sharex=True, sharey=True, figsize=(12, 8))
kilonova.plot_multiband(figure=fig, axes=axes, filters=["g", "r", "i", "z", "y", "J"], show=True)

# use default priors
priors = redback.priors.get_priors(model=model)
priors['redshift'] = redshift

model_kwargs = dict(frequency=kilonova.filtered_frequencies, output_format='flux_density')
result = redback.fit_model(transient=kilonova, model=model, sampler=sampler, model_kwargs=model_kwargs,
                           prior=priors, sample='rslice', nlive=1000, resume=True)
result.plot_corner(show=True)
plt.show()
# returns a Kilonova result object
result.plot_lightcurve(show=True)
# Even though we only fit the 'g' band, we can still plot the fit for different bands.
result.plot_multiband_lightcurve(filters=["g", "r", "i", "z", "y", "J"], show=True)
   


model_kwargs = dict(frequency=2e14, output_format='flux_density', bands='sdssi')
time = np.linspace(0.1, 30, 50)
for x in range(100):
    ss = priors.sample()
    ss.update(model_kwargs)
    out = redback.transient_models.kilonova_models.one_component_kilonova_model(time, **ss)
    plt.semilogy(time, out, alpha=0.1, color='red')
plt.title("One Component Kilonova")
plt.xlabel('Time [days]')
plt.ylabel('Flux density [mJy]')
plt.tight_layout()
plt.show()

time_arr = np.logspace(-2.8, 2.8, num = 100)
time_arr_2 = np.linspace(-10, 10, num = 620)
time_arr_bns = np.linspace(0, 30, 100)
freq_arr = np.ones(len(time_arr))*1e14
freq_arr_2 = np.ones(len(time))*1e14
lambda_array = np.linspace(3000, 30000, 50)
print(freq_arr)
kwargs = dict(frequency=freq_arr, output_format = 'flux_density')
frequency = freq_arr
kilonova_bns = redback.transient_models.kilonova_models.bulla_bns_kilonova(time_arr_bns, redshift, mej_dyn, mej_disk, phi, costheta_obs, **kwargs)
print("Array:", kilonova_bns)
plt.semilogy(time_arr_bns, kilonova_bns)
plt.xlabel('Time (Days)')
plt.ylabel('Log Flux Density (mJy)')
plt.title(' Bulla Model For Binary Neutron Star Merger Kilonova For GW 170817')
plt.show()

g_band_data = data[data["band"] == 'g']
print(g_band_data)
time_arr_g = (g_band_data["time (days)"].values)
print(time_arr_g)
freq_arr_g = np.ones(len(time_arr_g))*6.32e14
print(freq_arr_g)
i_band_data = data[data["band"] == 'i']
time_arr_i = (i_band_data["time (days)"].values)
freq_arr_i = np.ones(len(time_arr_i))*3.83e14
r_band_data = data[data["band"] == 'r']
time_arr_r = (r_band_data["time (days)"].values)
freq_arr_r = np.ones(len(time_arr_r))*4.86e14
z_band_data = data[data["band"] == 'z']
time_arr_z = (z_band_data["time (days)"].values)
freq_arr_z = np.ones(len(time_arr_z))*3.46e14
y_band_data = data[data["band"] == 'y']
time_arr_y = (y_band_data["time (days)"].values)
freq_arr_y = np.ones(len(time_arr_y))*3.12e14
J_band_data = data[data["band"] == 'J']
time_arr_J = (J_band_data["time (days)"].values)
freq_arr_J = np.ones(len(time_arr_J))*2.40e14
print(time_arr_J)
print(freq_arr_J)


kwargs = dict(frequency=freq_arr_g, output_format = 'flux_density')
frequency = freq_arr_g
kilonova_bns_g = redback.transient_models.kilonova_models.bulla_bns_kilonova(time_arr_g, redshift, mej_dyn, mej_disk, phi, costheta_obs, **kwargs)
kwargs = dict(frequency=freq_arr_i, output_format = 'flux_density')
frequency = freq_arr_i
kilonova_bns_i = redback.transient_models.kilonova_models.bulla_bns_kilonova(time_arr_i, redshift, mej_dyn, mej_disk, phi, costheta_obs, **kwargs)
kwargs = dict(frequency=freq_arr_r, output_format = 'flux_density')
frequency = freq_arr_r
kilonova_bns_r = redback.transient_models.kilonova_models.bulla_bns_kilonova(time_arr_r, redshift, mej_dyn, mej_disk, phi, costheta_obs, **kwargs)
kwargs = dict(frequency=freq_arr_z, output_format = 'flux_density')
frequency = freq_arr_z
kilonova_bns_z = redback.transient_models.kilonova_models.bulla_bns_kilonova(time_arr_z, redshift, mej_dyn, mej_disk, phi, costheta_obs, **kwargs)
kwargs = dict(frequency=freq_arr_y, output_format = 'flux_density')
frequency = freq_arr_y
kilonova_bns_y = redback.transient_models.kilonova_models.bulla_bns_kilonova(time_arr_y, redshift, mej_dyn, mej_disk, phi, costheta_obs, **kwargs)
kwargs = dict(frequency=freq_arr_J, output_format = 'flux_density')
frequency = freq_arr_J
kilonova_bns_J = redback.transient_models.kilonova_models.bulla_bns_kilonova(time_arr_J, redshift, mej_dyn, mej_disk, phi, costheta_obs, **kwargs)

plt.semilogy(time_arr_g, kilonova_bns_g, label = 'g-band 475nm', marker = 'o')
plt.semilogy(time_arr_i, kilonova_bns_i, label ='i-band 783nm', marker = 'o')
plt.semilogy(time_arr_r, kilonova_bns_r, label = 'r-band 617nm', marker = 'o')
plt.semilogy(time_arr_z, kilonova_bns_z, label ='z-band 867nm', marker = 'o')
plt.semilogy(time_arr_y, kilonova_bns_y, label = 'y-band 962nm', marker = 'o')
plt.semilogy(time_arr_J, kilonova_bns_J, label ='J-band 1.25μm', marker = 'o')
plt.legend()
plt.xlabel('Time (Days)')
plt.ylabel('Log Flux Density (mJy)')
plt.title('Binary Neutron Star Merger Kilonova For GW170817')
plt.show()
time_arr_two = np.linspace(0.01, 30, 50)
kwargs = dict(frequency=freq_arr_2, lambda_array = lambda_array, output_format = 'flux_density')
frequency = kwargs['frequency']
lambda_array = lambda_array
one_component = redback.transient_models.kilonova_models.one_component_kilonova_model(time, redshift, mej_dyn, vej, kappa, **kwargs)
plt.plot(time_arr_two, (one_component))
plt.xlabel('Time (Days)')
plt.ylabel('Intensity (mJy)')
plt.title('One Component Kilonova Model Of GW170817')
plt.show()
time_arr_three = np.linspace(0.01, 30, 50)
sampler = 'nessai'
model = 'two_component_kilonova_model'
mej_1 = 0.001
vej_1 = 0.6
temperature_floor_1 = 4500 # this looks low?
kappa_1 = 100
mej_2 = 0.01
vej_2 = 0.075
temperature_floor_2 = 3300
kappa_2 = 10

freq_arr_2 = np.ones(len(time_arr_two))*1e14
freq = 1/(data['time (days)'].values*86400)
time_arr_3 = (data['time (days)'].values)
print(freq)
lambda_array = np.linspace(3000, 30000, 100)
kwargs = dict(frequency=freq_arr_2, lambda_array = lambda_array, output_format = 'flux_density')
frequency = kwargs['frequency']
lambda_array = lambda_array
two_component = redback.transient_models.kilonova_models.two_component_kilonova_model(time_arr_three, redshift, mej_1, vej_1, temperature_floor_1, kappa_1, mej_2, vej_2, temperature_floor_2, kappa_2, **kwargs)
plt.plot(time_arr_three, (two_component))
plt.xlabel('Time (Days)')
plt.ylabel('Intensity (mJy)')
plt.title('Two Component Kilonova Model Of GW170817')
plt.show()

plt.plot(time_arr_3, freq)  
plt.xlabel('Time (Days)')
plt.ylabel('Frequency (Hz)')
plt.title('GW170817 Frequency Post Merger')
plt.show()