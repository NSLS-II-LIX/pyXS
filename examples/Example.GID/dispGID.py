from exp_setup import *
import matplotlib as mpl
import matplotlib.gridspec as gridspec
from pyxs.plot.Axes2DPlot import Axes2dplot

mpl.rc('lines', linewidth=1)

gid_file = 'no36_500A_PTCDI_Td_RT-Ta_140C-ODTS-sth0.60.60s_WAXS'
xr_file = 'no36_500A_PTCDI_Td_RT-Ta_140C-ODTS-sth_-0.5_10-attn0.5_42s_WAXS'
bkg_file = 'no83_bare_Si-sth0.60.60s_WAXS'

dbkg = Data2d(bkg_file)
dbkg.set_exp_para(ew)
dbkg.data = (dbkg.data - 1000) * 15 / 10

dwaxs = Data2d(gid_file)
dwaxs.set_exp_para(ew)
dwaxs.data = np.abs(dwaxs.data - dbkg.data - 900)

dwaxs.conv_to_Iqrqz()
dwaxs.qdata += 50
dwaxs.data += 50

dxr = Data2d(xr_file)
dxr.set_exp_para(ew)
dxr.data -= 600

fig = plt.figure(figsize=(10, 10))
# fig = plt.figure()

gs = gridspec.GridSpec(2, 2,
                       width_ratios=[1, 10], height_ratios=[1, 1],
                       wspace=0.1, hspace=0.1
                       )

plt.subplot(gs[0])
ax = plt.gca()
paxr = Axes2dplot(ax, dxr)
paxr.plot(log=True)
paxr.set_color_scale(mpl.cm.gist_yarg)
paxr.img.set_clim(410, 2400)
ax.set_xlim(ew.bm_ctr_x - 40, ew.bm_ctr_x + 60)
ax.set_xticklabels([])
ax.set_yticklabels([])
ax.set_xticks([])
ax.set_yticks([])

# paxr.add_dec("A b:")
paxr.add_dec("P 0. 0. b+")
paxr.add_dec("L 0.0 -0.1 0.0 3.0 32 b:")

plt.subplot(gs[1])
ax = plt.gca()
pax = Axes2dplot(ax, dwaxs)
pax.plot(log=True)
pax.set_color_scale(mpl.cm.gist_yarg)
pax.img.set_clim(70, 8400)
ax.set_xticklabels([])
ax.set_yticklabels([])
ax.set_xticks([])
ax.set_yticks([])

# pax.add_dec("A b:")
pax.add_dec("Y 0.031524 r--")  # second parameter is qc

pax.add_dec("L 0.0 -0.1 0.0 3.0 32 b:")
pax.add_dec("L 1.0 -0.1 1.0 3.0 32 b--")
pax.add_dec("L 2.0 -0.1 2.0 3.0 32 b--")
pax.add_dec("L 3.0 -0.1 3.0 3.0 32 b--")
pax.add_dec("L -0.7 0.0 3.0 0.0 32 b--")
pax.add_dec("L -0.7 1.0 3.0 1.0 32 b--")
pax.add_dec("L -0.7 2.0 3.0 2.0 32 b--")
pax.add_dec("L -0.7 3.0 3.0 3.0 32 b--")

plt.subplot(gs[3])
ax = plt.gca()
paxq = Axes2dplot(ax, dwaxs, show_q_data=True)
paxq.plot(log=True)
paxq.set_color_scale(mpl.cm.gist_yarg)
mpl.cm.gist_yarg.set_under(color='y')
paxq.img.set_clim(70, 8400)
ax.set_xticklabels([])
ax.set_yticklabels([])
ax.set_xticks([])
ax.set_yticks([])

# paxq.add_dec("A b-")
paxq.add_dec("Y 0.031524 r--")

paxq.add_dec("L 0.0 -0.1 0.0 3.0 32 b--")
paxq.add_dec("L 1.0 -0.1 1.0 3.0 32 b--")
paxq.add_dec("L 2.0 -0.1 2.0 3.0 32 b--")
paxq.add_dec("L 3.0 -0.1 3.0 3.0 32 b--")
paxq.add_dec("L -0.7 0.0 3.0 0.0 32 b--")
paxq.add_dec("L -0.7 1.0 3.0 1.0 32 b--")
paxq.add_dec("L -0.7 2.0 3.0 2.0 32 b--")
paxq.add_dec("L -0.7 3.0 3.0 3.0 32 b--")

# gs.tight_layout(fig, rect=[None, None, 0.45, None])
# gs.tight_layout(fig,pad=1)

plt.savefig("image1.pdf", dpi=300)

# line cut
plt.figure()

# size of the "pixel"
ldx = 4
ldy = 2

# case #1, the image is recorded as an oscillation image (rocking curve) 
# the trajectory is given by (q, phi)

d1_q = np.arange(0.01, 2, 0.002)
xyphi = []
for q in d1_q:
    (px, py) = dxr.qphi2xy(q, np.pi / 2)
    xyphi.append((px, py, 0.))
d1_Ixy = dxr.profile_xyphi(xyphi, 3, 1, bkg=1)
plt.plot(d1_q, d1_Ixy + 100, label="$q_r=0$")
d1_Ixy1 = dxr.profile_xyphi2(xyphi, 3, 1, sub_bkg=True)
plt.plot(d1_q, d1_Ixy1 + 100, label="$q_r=0$, new roi")

# case #2, GID with fixed incident angle specified in exp_para
# the trajectory is given by (qr, qz)

d2_qz = np.arange(-0.05, 2, 0.002)

xyphi = []
for qz in d2_qz:
    (px, py) = dxr.qrqz2xy(0.741, qz)
    xyphi.append((px, py, 0.))

# this version use roi_cnt, which utilizes the 2D image, not affected by image mathmatics
d2_Ixy = dwaxs.profile_xyphi(xyphi, 3, 1, bkg=1)
plt.plot(d2_qz, d2_Ixy + 10, label="$q_r=0.741\AA^{-1}$, original roi")

# this version use roi_cnt, which utilizes the 2D array, affected by image mathmatics
d2_Ixy1 = dwaxs.profile_xyphi2(xyphi, 3, 1, sub_bkg=True, use_qdata=False)
plt.plot(d2_qz, d2_Ixy1 + 10, label="$q_r=0.741\AA^{-1}$, new roi, cut in D, w=3")

# the proper method is to do the line cut on qr-qz plane
xyphi = []
for qz in d2_qz:
    xyphi.append((0.741, qz, 0.))
d2_Ixy2 = dwaxs.profile_xyphi2(xyphi, 3, 1, sub_bkg=True, use_qdata=True)
plt.plot(d2_qz, d2_Ixy2, label="$q_r=0.741\AA^{-1}$, new roi, cut in q, w=3")

ax = plt.gca()
ax.set_xlabel("$q (\AA^{-1})$", fontsize='x-large')
ax.set_ylabel("$I$", fontsize='x-large')
ax.set_yscale('log')
ax.set_xlim(-0.05, 2)
leg = ax.legend(loc='upper right', frameon=False)
for t in leg.get_texts():
    t.set_fontsize('small')

plt.show()
