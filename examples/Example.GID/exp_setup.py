import os
import sys

import numpy as np
import pickle

from pyxs.Data2D import *
from pyxs.ext.RQconv import *
from pyxs.slnXS import *
import matplotlib.pyplot as plt

es = ExpPara()
es.wavelength = 0.886
es.bm_ctr_x = 619
es.bm_ctr_y = 799
es.ratioDw = 32.6
es.det_orient = 0
es.det_tilt = 0
es.det_phi = 0
es.grazing_incident = False   
es.incident_angle = 0   
es.sample_normal = 0      

# parameters from view.gtk
# D/d = 2.86
# center = 29, 1020
# D/d = 4.60
# center = 36, 1020
# 2.86 * 1024/1042 = 2.81
# 
ew = ExpPara()
ew.wavelength = 0.886
ew.bm_ctr_x = 27 # was 5
ew.bm_ctr_y = 1039 # was 1015
ew.ratioDw = 2.52
ew.det_orient = 45
ew.det_tilt = -17
ew.det_phi = 0
ew.grazing_incident = True  
ew.flip = True
ew.incident_angle = 0.6   
ew.sample_normal = 0.25     

#qsaxs = np.arange(0.005,0.23,0.001)
#msaxs = Mask(1024,1024)
#msaxs.read_file("mask.SAXS")

#qwaxs = np.arange(0.1,2.05,0.005)
#mwaxs = Mask(1042,1042)
#mwaxs.read_file("mask.WAXS")

#dsamp = Data1d(qsaxs,qwaxs)
#dbuf = Data1d(qsaxs,qwaxs)

# this file correspond to the averaged dark current
# change/rebuild this file for new exposure time or qsaxs/qwaxs
#DARK_FILE="dark-90s.pkl"

#if os.path.isfile(DARK_FILE):
#    pkl_file = open(DARK_FILE, 'rb')
#    ddark = pickle.load(pkl_file)
#    pkl_file.close()
#else:
#    ddark = Data1d(qsaxs,qwaxs)
#    # change the list of files below for each unique exposure time 
#    ddark.avg(["dark-july15-1.90s",
#               "dark-july15-2.90s",
#               "dark-july15-3.90s",
#               "dark-july15-4.90s",
#               "dark-july15-5.90s",
#               "dark-july15-6.90s"],
#              es,ew,msaxs,mwaxs,
#              plot_data=False)
#    pkl_file = open(DARK_FILE, 'wb')
#    pickle.dump(ddark, pkl_file)
#    pkl_file.close()


