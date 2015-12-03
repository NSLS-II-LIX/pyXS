from exp_setup import *

import sys
import pyxs.slnXS as slnXS
slnXS.trans_mode = slnXS.TRANS_FROM_BEAM_CENTER

if len(sys.argv)<3:
    print("Usage: proc.py sample buf protein_conc out_put_file")
    print("multiple files can be given for sample and buf")
    print("e.g. ./proc.py \"lys20 file2\" lysbuf4 3.7 lyso20.dat")
    exit()
else:
    # set plot_data=True to see curves from the individual files
    # a label can be given to the averaged curve
    d1 = proc_SAXS(sys.argv[1].split(),
                   sys.argv[2].split(),
                   sdark,reft=-1,
                   conc=float(sys.argv[3]))
    d1.save(sys.argv[4])
    
    analyze(d1,qstart=0.02,qend=0.08,fix_qe=False,qcutoff=0.9,dmax=100)

    plt.show()


