from exp_setup import *
import sys
import pyxs.slnXS as slnXS
import time

t0 = time.time()
slnXS.trans_mode = slnXS.TRANS_FROM_WAXS
slnXS.WAXS_THRESH = 100

# set plot_data=True to see curves from the individual files
# a label can be given to the averaged curve
d1 = slnXS.process(["lysb_5mg-1.90s", "lysb_5mg-2.90s", "lysb_5mg-3.90s"],
                   ["lysb_buf-1.90s", "lysb_buf-2.90s", "lysb_buf-3.90s"],
                   detectors,
                   qmax=0.19, qmin=0.11, reft=-1,
                   conc=0,
                   save1d=True,
                   plot_data=True,
                   fix_scale=51.9,
                   filter_datasets=True,
                   similarity_threshold=0.5
                )

t1 = time.time()
d1.save("tt")
t2 = time.time()
analyze(d1, qstart=0.02, qend=0.09, fix_qe=True, qcutoff=0.9, dmax=100)
plt.show()

print("Process Time: ", t1-t0)
print("Save Time: ", t2-t1)
print("Total Time: ", t2-t0)
