# -*- coding: utf-8 -*-

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 
import pymc3 as pm
import arviz as az
import seaborn as sns


'''Data obtained from: https://es.wikipedia.org/wiki/Anexo:Sondeos_de_intenci%C3%B3n_de_voto_para_la_elecci%C3%B3n_presidencial_de_Chile_de_2021
correspond to voting intention polls only, see first graph in link above.'''

fecha = ["2021/09/03", "2021/09/16", "2021/10/01", "2021/10/07", "2021/10/14", 
         "2021/10/15", "2021/10/29", "2021/10/31", "2021/11/02", "2021/11/02", 
         "2021/11/03", "2021/11/04", "2021/11/06", "2021/11/19", "2021/11/20", 
         "2021/11/26", "2021/11/26", "2021/11/27", "2021/11/29", "2021/11/29", 
         "2021/11/29", "2021/12/02", "2021/12/02", "2021/12/04", "2021/12/04", 
         "2021/12/10", "2021/12/10"]

boric = [66.2, 64.9, 62.5, 56.1, 55, 58.4, 50.1, 46.8, 45.6, 46.3, 50, 47.6, 
         47.1, 52.9, 50, 62.2, 54.2, 51.8, 49.8, 54, 58, 53, 53.3, 59.9, 51.5, 49, 52]

kast = [33.7, 35.1, 37.5, 43.9, 45, 41.6, 49.9, 53.2, 54.4, 53.7, 50, 52.4, 52.9, 47.1, 
        50, 37.8, 45.8, 48.2, 50.2, 46, 42, 47, 46.7, 40.1, 48.5, 51, 48]

boric = np.array(boric)/100
kast = np.array(kast)/100

fj = pd.Categorical(fecha).codes
x_obs = fj[:,None]
y_obs_b = boric
y_obs_k = kast
    
with pm.Model() as model:
    # Polling period_b
    ls_b = pm.Gamma(name='ls_b', alpha=2.0, beta=1.0)
    period_b = pm.Gamma(name='period_b', alpha=100, beta=2)
    
    # Gaussian process
    gp_b = pm.gp.Marginal(cov_func=pm.gp.cov.Periodic(input_dim=1, period=period_b, ls=ls_b))

    # Error
    sigma_b = pm.HalfNormal(name='sigma_b', sigma=0.5)
    # Likelihood
    y_pred_b = gp_b.marginal_likelihood('y_pred_b', X=x_obs, y=y_obs_b, noise=sigma_b)
    
    # Polling period_k
    ls_k = pm.Gamma(name='ls_k', alpha=2.0, beta=1.0)
    period_k = pm.Gamma(name='period_k', alpha=100, beta=2)
    
    # Gaussian process 
    gp_k = pm.gp.Marginal(cov_func=pm.gp.cov.Periodic(input_dim=1, period=period_k, ls=ls_k))

    # Error
    sigma_k = pm.HalfNormal(name='sigma_k', sigma=0.5)
    # Likelihood
    y_pred_k = gp_k.marginal_likelihood('y_pred_k', X=x_obs, y=y_obs_k, noise=sigma_k)


with model:
    preds = pm.sample_prior_predictive(samples=1000, var_names=["y_pred_b", "y_pred_k"], random_seed=18)

fig, ax = plt.subplots()
samps = np.random.randint(0,1000,100)
for s in samps:
    ax = sns.kdeplot(preds['y_pred_b'][s], gridsize=1000, color=(0.8, 0.2, 0.1, 0.1))
sns.kdeplot(y_obs_b, gridsize=1000, color='r', label='Boric')
for s in samps:
    ax = sns.kdeplot(preds['y_pred_k'][s], gridsize=1000, color=(0.1, 0.3, 0.8, 0.1))
sns.kdeplot(y_obs_k, gridsize=1000, color='b', label='Kast')
plt.xlabel('Average Voting Preference')
plt.title('Prior Predictive Checks')
plt.gca().spines['right'].set_visible(False)
plt.gca().spines['top'].set_visible(False)
plt.legend()
plt.tight_layout()
plt.savefig('prior_pc.png',dpi=300)
plt.close()
plt.close()

####Sampling
with model:
    trace = pm.sample(1000, tune=1000, chains=4, random_seed=18)


fecha_new = fecha + ['2021/12/19', '2021/12/nn']
time_new = pd.Categorical(fecha_new).codes[:,None]
x_pred = pd.Categorical(fecha_new).codes[:,None]
x_pred = x_pred[27:]

#predictions
with model:
    x_b_obs_conditional = gp_b.conditional('x_b_obs_cond', x_obs)
    y_b_obs_pred_samples = pm.sample_posterior_predictive(trace, var_names=["x_b_obs_cond"], random_seed=18)
    x_b_pred_conditional = gp_b.conditional('x_b_pred_cond', x_pred)
    y_b_pred_pred_samples = pm.sample_posterior_predictive(trace, var_names=["x_b_pred_cond"], random_seed=18)
    x_k_obs_conditional = gp_k.conditional('x_k_obs_cond', x_obs)
    y_k_obs_pred_samples = pm.sample_posterior_predictive(trace, var_names=["x_k_obs_cond"], random_seed=18)
    x_k_pred_conditional = gp_k.conditional('x_k_pred_cond', x_pred)
    y_k_pred_pred_samples = pm.sample_posterior_predictive(trace, var_names=["x_k_pred_cond"], random_seed=18)
                    
                                                           
### Boric data for plotting
boric_obs_mean = y_obs_b
pred_old_b = [m.mean() for m in y_b_obs_pred_samples['x_b_obs_cond'].T] 
pred_new_b = [m.mean() for m in y_b_pred_pred_samples['x_b_pred_cond'].T] 
boric_pred_mean = pred_old_b + pred_new_b
h5_pred_b = [az.hdi(m, hdi_prob=0.9)[0] for m in y_b_pred_pred_samples['x_b_pred_cond'].T] 
h95_pred_b = [az.hdi(m, hdi_prob=0.9)[1] for m in y_b_pred_pred_samples['x_b_pred_cond'].T] 
h5_old_b = [az.hdi(m, hdi_prob=0.9)[0] for m in y_b_obs_pred_samples['x_b_obs_cond'].T] + h5_pred_b
h95_old_b = [az.hdi(m, hdi_prob=0.9)[1] for m in y_b_obs_pred_samples['x_b_obs_cond'].T] + h95_pred_b 

### kast data for plotting
kast_obs_mean = y_obs_k
pred_old_k = [m.mean() for m in y_k_obs_pred_samples['x_k_obs_cond'].T] 
pred_new_k = [m.mean() for m in y_k_pred_pred_samples['x_k_pred_cond'].T] 
kast_pred_mean = pred_old_k + pred_new_k
h5_pred_k = [az.hdi(m, hdi_prob=0.9)[0] for m in y_k_pred_pred_samples['x_k_pred_cond'].T] 
h95_pred_k = [az.hdi(m, hdi_prob=0.9)[1] for m in y_k_pred_pred_samples['x_k_pred_cond'].T] 
h5_old_k = [az.hdi(m, hdi_prob=0.9)[0] for m in y_k_obs_pred_samples['x_k_obs_cond'].T] + h5_pred_k
h95_old_k = [az.hdi(m, hdi_prob=0.9)[1] for m in y_k_obs_pred_samples['x_k_obs_cond'].T] + h95_pred_k 


#variables for plots
time_new = pd.Categorical(fecha_new).codes[:,None]
X = time_new
fecha_lab = ['03 Sep', '15 Oct', '11 Nov', '27 Nov', '19 Dic']
x_lab = [X.flatten()[0],X.flatten()[5],X.flatten()[11],X.flatten()[17],X.flatten()[27]]
y_tick = np.array([0.30,0.35,0.40,0.45,0.50,0.55,0.60,0.65,0.70])
y_lab = np.array(['30%','35%','40%','45%','50%','55%','60%','65%','70%'])


########## Plot data
plt.plot(time_new, boric_pred_mean, color='coral', label='Boric: observed preference')
plt.fill_between(time_new.flatten(), h5_old_b, h95_old_b, alpha=0.3, color='coral', label="Boric: 90% HDI")
plt.plot(x_obs.flatten(), pred_old_b, color='coral')
plt.plot([19]+list(x_pred.flatten()), [pred_old_b[-1]]+pred_new_b, color='crimson', label='Boric: predicted preference')
plt.fill_between([19]+list(x_pred.flatten()), [h5_old_b[-3]]+h5_pred_b, [h95_old_b[-3]]+h95_pred_b, alpha=0.3, color='crimson')
plt.scatter(time_new.flatten(), list(y_obs_b)+pred_new_b, color='red', s=10)
plt.axhline(pred_new_b[0], color='r', linestyle=":")
plt.text(0, pred_new_b[0]+0.005, str(round(pred_new_b[0]*100,2))+'%', color='r')

plt.plot(time_new, kast_pred_mean, color='skyblue', label='Kast: observed preference')
plt.fill_between(time_new.flatten(), h5_old_k, h95_old_k, alpha=0.3, color='skyblue', label="Kast: 90% HDI")
plt.plot(x_obs.flatten(), pred_old_k, color='skyblue')
plt.plot([19]+list(x_pred.flatten()), [pred_old_k[-1]]+pred_new_k, color='navy', label='Kast: predicted preference')
plt.fill_between([19]+list(x_pred.flatten()), [h5_old_k[-3]]+h5_pred_k, [h95_old_k[-3]]+h95_pred_k, alpha=0.3, color='navy')
plt.scatter(time_new.flatten(), list(kast_obs_mean)+pred_new_k, color='blue', s=10)
plt.axhline(pred_new_k[0], color='b', linestyle=":")
plt.text(0, pred_new_k[0]-0.022, str(round(pred_new_k[0]*100,2))+'%', color='b')
plt.axvline(X[27], color='dimgray', linestyle=":")

plt.text(X[22]+0.3, 0.65, "Election Day", color='dimgray')
plt.xticks(ticks=x_lab, labels=fecha_lab, size=12)
plt.yticks(y_tick, y_lab, size=12)
plt.xlabel('Date', size=16)
plt.ylabel('Voting Preference (itention)', size=16)
plt.legend(fontsize=8)
plt.gca().spines['top'].set_visible(False)
plt.gca().spines['right'].set_visible(False)
plt.grid(alpha=0.3)
plt.title('Predicciones Presidenciales Chile 2021')
plt.tight_layout()
plt.savefig('preds_model.png', dpi=300)


#model checks
az.plot_energy(trace)
plt.savefig('energy_model.png')

summ = az.summary(trace, hdi_prob=0.9)
summ.to_csv('summary_model.csv', index=False)


