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

data = pd.read_csv('160821B.csv') #this is a pandas DataFrame
time_d = data['DeltaT'].values #creates an array of the values in time column
time_err = data['Time Error'].values
data_filter = data['Filter'].values
AB_mag = data['magnitude'].values
flux_density = data['flux_density'].values #unit is mJy
flux_density_err = 1e3 * data['Flux Density Error'].values
frequency = redback.utils.bands_to_frequency(data_filter)
#data['Frequency'].values

thv = 0.0
redshift = 0.162

kwargs = dict(frequency=frequency, output_format='flux_density')
data_mode = 'flux_density'
name = '160821B'



afterglow = redback.transient.Afterglow(name=name, data_mode='flux_density', time=time_d, flux_density=flux_density, flux_density_err=flux_density_err, frequency=frequency)

#print(afterglow.data_mode)

model_data = model_func(time_d, 0.162, 0.0, 51.3, 0.1, 10, 10, 6, -4, 2.3, -1, -2, 100, 0.1, 0.001, 0.2, 2500, 10, 0.01, 0.1, 2500, 1, **kwargs)


plt.loglog(time_d, flux_density, 'o', color='blue', markersize=6, alpha=0.2)
plt.loglog(time_d, model_data, 's', color='red', markersize=6, alpha=0.2)

plt.show()

nlive = 200 
sampler = 'dynesty'


priors = bilby.core.prior.PriorDict()   
priors['thc'] = Uniform(0, 0.2, 'thc', latex_label=r'$\thc$')
priors['loge0'] = Uniform(46, 54, 'loge0', latex_label=r'$\loge0$')
priors['logn0'] = Uniform(-6, 1, 'logn0', latex_label=r'$\logn0$')
priors['p'] = Uniform(1.3, 3.3, 'p', latex_label=r'$\p$')
priors['logepse'] = Uniform(-6, -0.5, 'logepse', latex_label=r'$\logepse$')
priors['logepsb'] = Uniform(-6, -0.5, 'logepsb', latex_label=r'$\logepsb$')
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
priors['et'] = Uniform(2, 40, 'et', latex_label=r'$\et$')
priors['s1'] = Uniform(1, 10, 's1', latex_label=r'$\s1$')
priors['redshift'] = 0.162
priors['thv'] = 0.0

model = model_func

model_kwargs = kwargs 

result = redback.fit_model(model=model, sampler=sampler, nlive=nlive, transient=afterglow,
                           model_kwargs=model_kwargs, prior=priors, sample='rslice', resume=True)
result.plot_lightcurve(random_models=100, model=model_func)
