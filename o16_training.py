import timeit;
import numpy as np;
import matplotlib.pyplot as plt;
from sympy import S;
from tqdm import tqdm

from twobody.models.rmatrix.rmatrix import *;
from twobody.constants import RADIAN_PER_DEGREE;

from twobody.twobody.models.pointcoulomb import PointCoulomb;

import matplotlib.pyplot as plt;

rm = PhenomenologicalRmatrix.load_complete("data/o16/sam_o16",compressed=True);

header  = { "cn" : { "a":int(rm.cn.a),
                     "z":int(rm.cn.z),
                     "spin":float(rm.cn.spin),
                     "parity":float(rm.cn.parity),
                     "ex":float(rm.cn.ex),
                     "m_gs":float(rm.cn.m_gs) },
            "pp" : [ { "nuc1":{ "a":int(pp.nuc1.a),
                                "z":int(pp.nuc1.z),
                                "spin":float(pp.nuc1.spin),
                                "parity":float(pp.nuc1.parity),
                                "ex":float(pp.nuc1.ex),
                                "m_gs":float(pp.nuc1.m_gs) },
                       "nuc2":{ "a":int(pp.nuc2.a),
                                "z":int(pp.nuc2.z),
                                "spin":float(pp.nuc2.spin),
                                "parity":float(pp.nuc2.parity),
                                "ex":float(pp.nuc2.ex),
                                "m_gs":float(pp.nuc2.m_gs) } } for pp in rm.particle_pairs ],
            "jpi_sets" : [ { "j":float(jps["j"]), "parity":float(jps["parity"]),
                             "channels":[ { "pp_index":rm.particle_pairs.index(c.pp),"s":float(c.s), "l":float(c.l)} for c in jps["channels"]]} for jps in rm.jpi_sets ] };

with open("data/o16/o16_header.json","w") as f :
    json.dump(header,f,indent=2);
f.close();


# Generate training data
cn_ex_min = np.min(rm._observable_sets[0].get_coords("cn_ex")) + 0.5;
cn_ex_max = np.max(rm._observable_sets[0].get_coords("cn_ex"));

# Remove all resonances
training_sets=[];
for nset in tqdm.tqdm(range(0,1000)) :
    
    levels=[];
    level_set_info=[];

    # num_levels in [5, 20]
    num_levels = np.random.randint(15) + 5

    for n in range(0, num_levels) :
    
        jpi_set = np.random.choice(rm.jpi_sets);
        j = jpi_set["j"];
        parity = jpi_set["parity"];
        jpi_index = [ i for i,jpi_set in enumerate(rm.jpi_sets) if jpi_set["j"]==j and jpi_set["parity"]==parity  ][0];

        energy = np.random.random()*(cn_ex_max-cn_ex_min) + cn_ex_min;
        rm.set_bc_to_cn_ex(energy);
        levels.append( rm.add_level(j=j,parity=parity,energy=energy,skip_check=True) );
        level_info = { "energy":energy, "jpi_index":jpi_index, "j":j, "parity":parity, "channels":[], "Gamma":[] };
        
        for ic,channel in enumerate(levels[-1].channels) :
            gamma = (np.random.random()**2 * 0.5 + 0.1) * channel.gamma_wigner;
            levels[-1].set_gamma(channel,gamma);
            temp=rm.get_observed_level(levels[-1]);
            level_info["Gamma"].append(temp["Gamma"][ic]);
            level_info["Gamma_total"] = temp["Gamma_total"];
        level_set_info.append(level_info);
        
    rm.update();

    # truncate data here.
    
    level_dict = { "levels" : [ { "energy" : float(level_info["energy"]),
                                  "jpi_index":int(level_info["jpi_index"]),
                                  #"j":float(level.j),
                                  #"parity":float(level.parity),
                                  "Gamma":[ float(G) for G in level_info["Gamma"] ],
                                  "Gamma_total": float(level_info["Gamma_total"]) } for level,level_info in zip(levels,level_set_info) ] };

    
    data_dict = { "data": [ {"pp_in_index":rm.particle_pairs.index(os.pp_in),
                             "pp_out_index":rm.particle_pairs.index(os.pp_out),
                             "points":[ { "ke_cm_in":float(obs.coords["ke_cm_in"]),
                                          "cn_ex":float(obs.coords["cn_ex"]),
                                          "theta_cm_out":float(obs.coords["theta_cm_out"]),
                                          "dsdO": float(obs.val_th),
                                          "dsdRuth": float(obs.val_th/obs.val_ruth),
                                          "dsdO-dsRuth": float(obs.val_th-obs.val_ruth)} for obs in os.obs  ] } for os in rm._observable_sets  ] };
                   
    training_sets.append({ "levels":level_dict["levels"], "observable_sets":data_dict["data"]});

    for level in levels :
        rm.remove_level(level);
        
with gzip.open("data/o16/multires_training.gz","wb") as f :
    json_str = json.dumps(training_sets);
    json_bytes = json_str.encode("UTF-8");
    f.write(json_bytes);

#with gzip.open("o16_training.gz","rb") as f :
#    json_bytes = f.read();
#    json_str = json_bytes.decode();
#    data = json.loads(json_str);

#with gzip.open(filename+".gz","rt") as f :
#    data = json.load(f);
