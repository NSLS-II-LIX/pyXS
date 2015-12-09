import os

try:
    import cPickle as pk
except ImportError:
    import pickle as pk

from pyxs.DetectorConfig import DetectorConfig
from pyxs.ext.RQconv import *
from pyxs.slnXS import *

es = ExpPara()
es.wavelength = 0.874
es.bm_ctr_x = 444
es.bm_ctr_y = 311
es.ratioDw = 40.8
es.det_orient = 0
es.det_tilt = 0
es.det_phi = 0
es.grazing_incident = False
es.flip = False
es.incident_angle = 0.2
es.sample_normal = 0

ew = ExpPara()
ew.wavelength = 0.874
ew.bm_ctr_x = 33
ew.bm_ctr_y = 1041
ew.ratioDw = 4.62
ew.det_orient = 45
ew.det_tilt = -19
ew.det_phi = 0
ew.beam_tX = 2
ew.grazing_incident = False
ew.flip = True
ew.incident_angle = 0.2
ew.sample_normal = 0

# qgrid = np.arange(0.01,2.05,0.01)

# qgrid must be sections of nodes with constant spacing
qgrid = mod_qgrid(np.hstack((np.arange(0.005, 0.0499, 0.001),
                             np.arange(0.05, 0.0999, 0.002),
                             np.arange(0.1, 0.4999, 0.005),
                             np.arange(0.5, 0.9999, 0.01),
                             np.arange(1.0, 2.05, 0.03))))

# this file correspond to the averaged dark current
# change/rebuild this file for new exposure time or qsaxs/qwaxs
DARK_FILE_S = "dark-s.pkl"
DARK_FILE_W = "dark-w.pkl"

# flat field correction includes the incident angle correction
FLAT_FILE_S = "flat-s.pkl"
FLAT_FILE_W = "flat-w.pkl"

if os.path.isfile(DARK_FILE_S) and os.path.isfile(DARK_FILE_W):
    pkl_file = open(DARK_FILE_S, 'rb')
    sdark = pk.load(pkl_file)
    pkl_file.close()
    pkl_file = open(DARK_FILE_W, 'rb')
    wdark = pk.load(pkl_file)
    pkl_file.close()
else:
    sdark = Data1d()
    wdark = Data1d()
    sdark.load_dark_from_2D(["dark1a.90s_SAXS",
                             "dark1b.90s_SAXS",
                             "dark1c.90s_SAXS",
                             "dark1d.90s_SAXS",
                             "dark1e.90s_SAXS"],
                            es, "mask.SAXS", qgrid, plot_data=False)
    wdark.load_dark_from_2D(["dark1a.90s_WAXS",
                             "dark1b.90s_WAXS",
                             "dark1c.90s_WAXS",
                             "dark1d.90s_WAXS",
                             "dark1e.90s_WAXS"],
                            ew, "mask.WAXS", qgrid, plot_data=False)
    # pickle doesn't like PIL objects
    sdark.exp_para = None
    wdark.exp_para = None
    pkl_file = open(DARK_FILE_S, 'wb')
    pk.dump(sdark, pkl_file)
    pkl_file.close()
    pkl_file = open(DARK_FILE_W, 'wb')
    pk.dump(wdark, pkl_file)
    pkl_file.close()

sdark.exp_para = es
wdark.exp_para = ew

if os.path.isfile(FLAT_FILE_W):
    pkl_file = open(FLAT_FILE_W, 'rb')
    wflat = pk.load(pkl_file)
    pkl_file.close()
else:
    fdark = Data1d()
    wflat = Data1d()
    fdark.load_dark_from_2D(["Feb09-dark-00.300s_WAXS",
                             "Feb09-dark-01.300s_WAXS",
                             "Feb09-dark-02.300s_WAXS",
                             "Feb09-dark-03.300s_WAXS",
                             "Feb09-dark-04.300s_WAXS",
                             "Feb09-dark-05.300s_WAXS"],
                            ew, "mask.WAXS", wdark.qgrid, plot_data=False)
    wflat.load_dark_from_2D(["Feb09-bright-00.300s_WAXS",
                             "Feb09-bright-01.300s_WAXS"],
                            ew, "mask.WAXS", wdark.qgrid, plot_data=False)
    wflat.data -= fdark.data
    wflat.d2data -= fdark.d2data
    wflat.err += fdark.err
    wflat.save("wflat.dat")
    wflat.exp_para = None
    del wflat.mask
    pkl_file = open(FLAT_FILE_W, 'wb')
    pk.dump(wflat, pkl_file)
    pkl_file.close()

detectors = []

detectorSAXS = DetectorConfig(extension = "_SAXS", dark = sdark, flat = None, dezinger = False)
detectorWAXS = DetectorConfig(extension = "_WAXS", dark = wdark, flat = wflat, dezinger = True)

detectors.append(detectorSAXS)
detectors.append(detectorWAXS)
