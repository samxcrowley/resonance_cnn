#!/usr/bin/env python
# coding: utf-8

# In[3]:


import numpy as np;
import matplotlib.pyplot as plt;


# In[4]:

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'twobody'))
import twobody.exfor;
import twobody.nucleus;
import twobody.models.partialwaves;
import twobody.channel;
import twobody.grid;


# # Calculation for creating artificial data sets for training machine learning models
# 
# 

# In[12]:


# For simplicity assumes a baseline potential of BG at 10 MeV with complex parts zeroed.

# Everything in CM frame.

def continuous_phase_shift(phases) :
    for i in range(1,len(phases)) :
        nochange=False;
        while not nochange :
            nochange=True;
            if phases[i-1]*phases[i] < 0 :
                if phases[i-1]-phases[i] > 0.3 :
                    phases[i]+=np.pi;
                    nochange=False;
                elif phases[i]-phases[i-1] > 0.3 :
                    phases[i]-=np.pi;
                    nochange=False;

a1=4;
z1=2;
a2=12;
z2=6;

# Define the collision and potential
p=twobody.nucleus.Nucleus(a1,z1);
t=twobody.nucleus.Nucleus(a2,z2);
pp=twobody.channel.ParticlePair(p,t);
pp.set_energy(10.0);

a13=a1**0.333333333+a2**0.333333333;

pot=twobody.potential.CompositePotential();

# Note that any realistic potential tends to exhibit genuine, wide single-particle-like resonances.
# To keep things simple, we should neglect these for now and use a hard-sphere.
# But ultimately we will want to be able to interpret exactly this sort of data.
if False :
    # Quasi-realistic potential
    pot.add_potential(twobody.potential.SphereCoulomb(1.25*a13));
    pot.add_potential(twobody.potential.Centrifugal());
    pot.add_potential(twobody.potential.WoodsSaxon(-45.9,1.16*a13,0.69,0.0,1.0,1.0));
else :
    # Quasi-hard sphere
    pot.add_potential(twobody.potential.SphereCoulomb(1.25*a13));
    pot.add_potential(twobody.potential.Centrifugal());
    pot.add_potential(twobody.potential.WoodsSaxon(10000,1.16*a13,0.1,0.0,1.0,1.0));

    
#np.seterr(all='raise');
g=twobody.grid.GridLinear(0,150.0,1501);
pw=twobody.models.partialwaves.PartialWaves(pp,pot,l_max=6,r=g);
ruth=twobody.models.partialwaves.PointCoulomb(pp);

#print(pot.get(g._x,twobody.channel.Channel(pp,l=0)));

# Evaluate the cross section

emin=0.1;
emax=6.0;
nestep=200;
thetamin=40.0;
thetamax=150.0;
ntheta=30;
uncertainty=0.05,

thetas=np.linspace(thetamin*np.pi/180.0,thetamax*np.pi/180,endpoint=True,num=ntheta);
energies=np.linspace(emin,emax,endpoint=True,num=nestep);

rng = np.random.default_rng();

import copy;

phase_shift_base=[];
for e in energies :
    pp.set_energy(e);
    pw.calculate_phase_shifts();
    phase_shift_base.append(copy.deepcopy(pw._phase_shift));

@twobody.profile.LineProfiler                    
def create_dataset() :

    n_res=rng.choice([1,2,3,4,5]);
    l_res=[];
    E_res=[];
    Gamma_res=[];
    info={ "n_res":n_res, "res":[] }
    for i in range(n_res) :
        l_res.append(rng.choice([0,1,2,3,4]));
        E_res.append(0.5+rng.random()*5);
        Gamma_res.append(0.05+rng.random()/(l_res[i]+1)*E_res[i]/4);
        info["res"].append({ "E_res":E_res[i], "Gamma_res":Gamma_res[i], "l_res":l_res[i] });

    xsetheta=np.zeros((len(energies),len(thetas)),dtype="float16");

    for ie,e in enumerate(energies) :
        
        e=energies[ie];
        
        # Calculate hs
        pw._phase_shift = copy.deepcopy(phase_shift_base[ie]);
        pw.calculate_phase_shift_derived();
        hsxs=pw.dsigma_dOmega_cm(thetas)

        # Add artifical resonance
        for i in range(n_res) :
            pw._phase_shift[l_res[i]] += np.arctan(Gamma_res[i]/2/(E_res[i]-e));
        
        # Complete the phase shift calculation, calculate cross sections
        pw.calculate_phase_shift_derived();
        ruth=pw._pointcoulomb.dsigmaRuth_dOmega_cm(thetas);
        # xs=np.log(pw.dsigma_dOmega_cm(thetas)/;
        xs=pw.dsigma_dOmega_cm(thetas)-ruth;
        
        xsetheta[ie]=xs;#np.log(xs/hsxs);
        #for i in range(0,len(thetas)) :
        #    xsetheta[ie][i]=xs[i];
            #data["data"].append({"E":e.astype(np.float16),"theta":thetas[i].astype(np.float16),"xs":xs[i].astype(np.float16)});

        
    return info,xsetheta;


# # alpha+c12 test

# In[13]:


import pickle

data_sets=[];
info_sets=[];

nsets=[3]; #[10,100,1000,10000,100000];
for nset in nsets :
    for i in range(nset) :
        info,d=create_dataset();
        data_sets.append(d);
        info_sets.append(info);
        if i%1000 == 0 : print(i);

    prefix="data/v5"
        
    file=open(prefix + "_etheta_" + str(nset) + ".pkl", "wb");
    pickle.dump({"E":energies,"theta":thetas},file);
    file.close();
        
    file=open(prefix + "_xsdata_" + str(nset) + ".pkl", "wb");
    pickle.dump(data_sets,file);
    file.close();
    
    file=open(prefix + "_info_" + str(nset) + ".pkl", "wb");
    pickle.dump(info_sets,file);
    file.close();
