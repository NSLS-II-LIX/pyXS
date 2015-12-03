import os
import sys
import matplotlib as mpl
from exp_setup import *

if (len(sys.argv) < 2):
    print("Usage: disp.py file_name_root")
    exit()
else:

    saxs_current_file = sys.argv[1] + '_SAXS'
    waxs_current_file = sys.argv[1] + '_WAXS'

    if os.path.isfile(saxs_current_file) and os.path.isfile(waxs_current_file):
        plt.figure(figsize=(10, 6))
    else:
        plt.figure(figsize=(6, 6))

    if os.path.isfile(saxs_current_file):
        if os.path.isfile(waxs_current_file):
            plt.subplot(121)
        ax1 = plt.gca()
        dsaxs = Data2d(saxs_current_file)
        dsaxs.set_exp_para(es)
        pax1 = Axes2dplot(ax1, dsaxs)
        pax1.plot(mask=sdark.mask)
        # pax1.set_color_scale(mpl.cm.get_cmap('jet'),3.5)
        # pax1.img.set_clim(0,200)

        if (len(sys.argv) > 2):
            pax1.add_dec("Q 0.1076 72 r-")
            pax1.add_dec("Q 0.2152 72 r-")
            pax1.add_dec("Q 0.3228 72 r-")
            pax1.add_dec("Q 0.4304 72 r-")

    if os.path.isfile(waxs_current_file):
        if os.path.isfile(saxs_current_file):
            plt.subplot(122)
        ax2 = plt.gca()
        dwaxs = Data2d(waxs_current_file)
        dwaxs.set_exp_para(ew)
        pax2 = Axes2dplot(ax2, dwaxs)
        pax2.plot(mask=wdark.mask)
        # pax2.set_color_scale(mpl.cm.get_cmap('jet'),3.5)
        # pax2.img.set_clim(1000,2000)

        if (len(sys.argv) > 2):
            pax2.add_dec("Q 0.1076 72 r-")
            pax2.add_dec("Q 0.2152 72 r-")
            pax2.add_dec("Q 0.3228 72 r-")
            pax2.add_dec("Q 0.4304 72 r-")
            pax2.add_dec("Q 0.5380 72 r-")
            pax2.add_dec("Q 0.6456 72 r-")
            pax2.add_dec("Q 0.7532 72 r-")
            pax2.add_dec("Q 0.8608 72 r-")
            pax2.add_dec("Q 0.9684 72 r-")
            pax2.add_dec("Q 1.0760 72 r-")
            pax2.add_dec("Q 1.37 72 r-")

    plt.show()
