import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from ..utils import cmap_map


class Axes2DPlot:
    """
    define another class to handle how the data is displayed
    zoom/crop, colormap, decorations
    pixel, position display
    most functionalities already exist in matplotlib
    """

    def __init__(self, ax, data2d, show_q_data=False):
        """
        """
        self.ax = ax
        self.cmap = plt.get_cmap('spectral')
        self.scale = 'linear'
        # self.cid = ax.figure.canvas.mpl_connect('motion_notify_event', self.move_event)
        self.showing_q_data = show_q_data
        if show_q_data and not data2d.q_data_available:
            data2d.conv_to_Iqrqz()
        self.plot(data2d)
        self.ptns = []
        self.capture_mouse()

    def capture_mouse(self):
        self.ax.figure.canvas.mpl_connect('button_press_event', self.move_event)
        # self.ax.figure.canvas.mpl_connect('motion_notify_event', self.move_event)

    def right_click(self, event):
        """ display menu to change image color scale etc.
        """
        pass

    def move_event(self, event):
        if event.inaxes != self.ax:
            return True
        toolbar = plt.get_current_fig_manager().toolbar
        x = int(event.xdata + 0.5)
        y = int(event.ydata + 0.5)
        if x < 0 or y < 0 or x >= self.d2.width or y >= self.d2.height:
            return True
        if self.showing_q_data:
            v = self.d2.qdata[y][x]
            qr = x * self.d2.exp.dq + self.d2.exp.qr0
            qz = (self.d2.exp.nz - y) * self.d2.exp.dq + self.d2.exp.qz0
            s = "%d, q = (%7.4f, %7.4f)" % (v, qr, qz)
        else:
            v = self.d2.data[y][x]
            s = "(%4d, %4d) = %d, " % (x, y, v)
            if self.d2.exp.grazing_incident:
                (qr, qz) = self.d2.xy2qrqz(event.xdata, event.ydata)
                q = np.sqrt(qr * qr + qz * qz)
                s += "q = %7.4f (%7.4f, %7.4f)" % (q, qr, qz)
            else:
                s += "q = %7.4f" % self.d2.xy2q(event.xdata, event.ydata)
        toolbar.set_message(s)
        return True

    def plot(self, data=None, mask=None, log=False):
        if data is not None:
            self.d2 = data
        if self.d2 is None:
            return  # this should never happen

        if self.showing_q_data:
            dd = self.d2.qdata
        else:
            if mask is not None:
                dd = (1 - mask.map) * (self.d2.data + 1) - 1
                # print "showing mask"
            else:
                dd = self.d2.data

        immax = np.average(dd) + 5 * np.std(dd)
        immin = np.average(dd) - 5 * np.std(dd)
        if immin < 0:
            immin = 0

        if log:
            self.img = self.ax.imshow(dd,
                                      cmap=self.cmap, interpolation='nearest', norm=LogNorm())
        else:
            self.img = self.ax.imshow(dd, vmax=immax, vmin=immin,
                                      cmap=self.cmap, interpolation='nearest')
        if self.showing_q_data:
            pass
            # xformatter = FuncFormatter(lambda x,pos: "%.3f" % self.d2.exp.qr0+self.d2.exp.dq*x)
            # self.ax.yaxis.set_major_formatter(xformatter)
            # yformatter = FuncFormatter(millions)
            # self.ax.yaxis.set_major_formatter(yformatter)

    def set_color_scale(self, cmap, gamma=1):
        """ linear, log/gamma
        """
        if not gamma == 1:
            cmap = cmap_map(lambda x: np.exp(gamma * np.log(x)), cmap)
        self.cmap = cmap
        self.img.set_cmap(cmap)

    def add_dec(self, ptn):
        self.ptns.append(ptn)
        self.draw_dec(self.ptns)

    def draw_dec(self, ptns):
        for ptn in ptns:
            items = ptn.strip().split()
            if items[0] == 'q' or items[0] == 'Q':
                # ring at the specified q
                # q      q0      N     line_type
                q0, n = [float(t) for t in items[1:3]]
                ang = np.append(np.arange(0, 360., 360. / n, float), [360.])
                ang *= np.pi / 180.
                p = np.array([self.d2.qphi2xy(q0, t) for t in ang])
                self.ax.plot(p[:, 0], p[:, 1], items[3], scalex=False, scaley=False)
            elif items[0] == 'l' or items[0] == 'L':
                # line connect (qr1,qz1) to (qr2,qz2)
                # L    qr1    qz1    qr2    qz2    N   line_type
                if not self.d2.exp.grazing_incident:
                    return
                qr1, qz1, qr2, qz2, n = [float(t) for t in items[1:6]]
                dist = np.append(np.arange(0, 1., 1. / n), [1.])
                if self.showing_q_data:
                    p = np.array([(qr1 * t + qr2 * (1. - t), qz1 * t + qz2 * (1. - t)) for t in dist])
                    px = (np.array(p[:, 0]) - self.d2.exp.qr0) / self.d2.exp.dq
                    py = self.d2.exp.nz - (np.array(p[:, 1]) - self.d2.exp.qz0) / self.d2.exp.dq
                else:
                    p = np.array([self.d2.qrqz2xy(qr1 * t + qr2 * (1. - t), qz1 * t + qz2 * (1. - t)) for t in dist])
                    px = np.array(p[:, 0])
                    py = np.array(p[:, 1])
                # print px,py
                self.ax.plot(px, py, items[6], scalex=False, scaley=False)
            elif items[0] == 'p' or items[0] == 'P':
                # a point
                # P    qr   qz    marker_type
                qr, qz = [float(t) for t in items[1:3]]
                x, y = self.d2.qrqz2xy(qr, qz)
                self.ax.plot(x, y, items[3], scalex=False, scaley=False)
            elif items[0] == 'a' or items[0] == 'A':
                # plot the qr, qz axes for GID
                if self.showing_q_data:
                    # position of the origin
                    x0 = -self.d2.exp.qr0 / self.d2.exp.dq
                    y0 = self.d2.exp.nz + self.d2.exp.qz0 / self.d2.exp.dq
                    self.ax.plot([0, self.d2.exp.nr], [y0, y0], items[1], scalex=False, scaley=False)
                    self.ax.plot([x0, x0], [0, self.d2.exp.nz], items[1], scalex=False, scaley=False)
                else:
                    # qr-axis
                    dist = np.append(np.arange(0, 1., 1. / 32),
                                     [1.]) * self.d2.exp.nr * self.d2.exp.dq + self.d2.exp.qr0
                    p = np.array([self.d2.qrqz2xy(t, 0) for t in dist])
                    # px = np.array(p[:, 0])
                    # py = np.array(p[:, 1])
                    self.ax.plot(p[:, 0], p[:, 1], items[1], scalex=False, scaley=False)
                    # qz-axis
                    dist = np.append(np.arange(0, 1., 1. / 32),
                                     [1.]) * self.d2.exp.nz * self.d2.exp.dq + self.d2.exp.qz0
                    p = np.array([self.d2.qrqz2xy(0, t) for t in dist])
                    # px = np.array(p[:, 0])
                    # py = np.array(p[:, 1])
                    self.ax.plot(p[:, 0], p[:, 1], items[1], scalex=False, scaley=False)
            elif items[0] == 'y' or items[0] == 'Y':
                # "A qc fmt", plot the Yoneda peak/line for GID
                # Yoneda wing at theta_o = critical angle
                # K_out contribution to q is qc, K_in contribution to q is K sin(alpha_i)
                qc = np.float(items[1])
                q_yoneda = qc + 2.0 * np.pi / self.d2.exp.wavelength * np.sin(self.d2.exp.incident_angle / 180. * np.pi)
                if self.showing_q_data:
                    y0 = self.d2.exp.nz - (q_yoneda - self.d2.exp.qz0) / self.d2.exp.dq
                    self.ax.plot([0, self.d2.exp.nr], [y0, y0], items[2], scalex=False, scaley=False)
                else:
                    dist = np.append(np.arange(0, 1., 1. / 32),
                                     [1.]) * self.d2.exp.nr * self.d2.exp.dq + self.d2.exp.qr0
                    p = np.array([self.d2.qrqz2xy(t, q_yoneda) for t in dist])
                    # px = np.array(p[:, 0])
                    # py = np.array(p[:, 1])
                    self.ax.plot(p[:, 0], p[:, 1], items[2], scalex=False, scaley=False)
            else:
                print("invalid pattern: %s" % ptn)

# Back compatibility issues
Axes2dplot = Axes2DPlot
