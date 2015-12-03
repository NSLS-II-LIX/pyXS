import pyxs.ext.RQconv as RQconv
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.colors as mc
from matplotlib.colors import LogNorm
from scipy import interpolate
from functools import reduce

""" this module handle data I/O and display
    the heavy-duty calculations are done by a C module, RQconv:
    dezinger
    qphi2xy
    qrqz2xy
    xy2q
    xy2qrqz
    conv_to_Iq
    conv_to_Iqrqz
"""


class Data2d:
    """ 2D scattering data class
    stores the scattering pattern itself, 
    as well as parameters of the experiment
    NOTE: PIL must be TiffImagePlugin.py updated to correctly read WAXS tiff (Big Endian)
    see http://mail.python.org/pipermail/image-sig/2006-November/004195.html
    """
    data = np.array([])

    def __init__(self, filename, flip=0):
        """ read 2D scattering pattern
        will rely on PIL to recognize the file format 
        flip=1 for PSI WAXS data 
        """
        # most detector images have mode "I;16"
        # have to be converted to mode "I" for transpose and conversion to array to work
        # conversion to mode "I" apparent also works for tiff32 (PILATUS)
        # NOTE: the index of the 2d array is data[row,col]
        self.exp = None
        self.im = Image.open(filename).convert("I")
        if flip:
            self.im = self.im.transpose(Image.ROTATE_90).transpose(Image.FLIP_LEFT_RIGHT)
        # convert into an array
        # copy() seems to be necessary for later alteration of the 2D data
        self.data = np.asarray(self.im).copy()
        (self.height, self.width) = np.shape(self.data)
        self.q_data_available = False
        self.qdata = None
        # self.exp = RQconv.ExpPara()

    def set_exp_para(self, exp):
        if exp.flip:
            self.im = self.im.transpose(Image.ROTATE_90).transpose(Image.FLIP_LEFT_RIGHT)
            self.data = np.asarray(self.im).copy()
            (self.height, self.width) = np.shape(self.data)
        self.exp = exp
        RQconv.calc_rot(self.exp)

    def val(self, fx, fy):
        """ intensity at the pixel position (fx, fy), interpolatd from the neighbors 
        """
        """
        ix = int(fx)
        iy = int(fy)
        if ix<0 or iy<0 or ix>=self.width or iy>=self.height : return(0)
        t = (1.-(fx-ix))*(1.-(fy-iy))*self.data[iy,ix]
        t += (fx-ix)*(1.-(fy-iy))*self.data[iy+1,ix]
        t += (1.-(fx-ix))*(fy-iy)*self.data[iy,ix+1]
        t += (fx-ix)*(fy-iy)*self.data[iy+1,ix+1]
        return t
        """
        return RQconv.get_value(self.data, fx, fy)

    def roi_stat(self, cx, cy, w, h):
        """ calculate the average and standard deviation within the ROI
        """
        dd = self.data[cy - h + 1:cy + h, cx - w + 1:cx + w]
        print(cx, cy, w, h)
        # print "\n".join(str(t) for t in dd.flatten())
        # need to removed the zeros
        print(np.average(dd), np.std(dd))
        return np.average(dd), np.std(dd)

    def roi_COM(self, cx, cy, w, h):  # what if the ROI is out-of-bounds
        """ return the center of mass (intensity) within the ROI
            ROI pixels from [cx-(w-1),cx+(w-1)],[cy-(h-1),cy+(h-1)]
        """
        dd = self.data[cy - h + 1:cy + h, cx - w + 1:cx + w]
        iy, ix = np.indices((2 * h - 1, 2 * w - 1))
        ix += cx - w + 1
        iy += cy - h + 1
        mx = float((ix * dd).flatten().sum()) / dd.flatten().sum()
        my = float((iy * dd).flatten().sum()) / dd.flatten().sum()
        return mx, my

    def roi(self, cx, cy, w, h, phi, use_qdata=False):
        # should check how data are stored, data[irow,icol] or data[icol,irow]
        # elements in a 2D array can be referred to as data[irow,icol], or data[irow][icol]
        # or data[irow*Ncol+icol], (Nrow,NCol)=data.shape
        if use_qdata:  # convert qr,qz into index
            cx = (cx - self.exp.qr0) / self.exp.dq
            cy = self.exp.nz - 1 - (cy - self.exp.qz0) / self.exp.dq  # the highest index is nz-1

        dm = np.sqrt(w ** 2 + h ** 2)
        ix1 = np.int(cx - dm)
        ix2 = np.int(cx + dm) + 1
        iy1 = np.int(cy - dm)
        iy2 = np.int(cy + dm) + 1

        if ix1 < 0:
            ix1 = 0
        if ix2 < 0:
            ix2 < 0
        if use_qdata:
            if ix1 >= self.exp.nr:
                ix1 = self.exp.nr - 1
            if ix2 >= self.exp.nr:
                ix2 = self.exp.nr - 1
        else:
            if ix1 >= self.width:
                ix1 = self.width - 1
            if ix2 >= self.width:
                ix2 = self.width - 1
        if iy1 < 0:
            iy1 = 0
        if iy2 < 0:
            iy2 < 0
        if use_qdata:
            if iy1 >= self.exp.nz:
                iy1 = self.exp.nz - 1
            if iy2 >= self.exp.nz:
                iy2 = self.exp.nz - 1
        else:
            if iy1 >= self.height:
                iy1 = self.height - 1
            if iy2 >= self.height:
                iy2 = self.height - 1

        if ix1 == ix2 or iy1 == iy2:
            return 0

        if use_qdata:
            d2s = self.qdata[iy1:iy2 + 1, ix1:ix2 + 1]
        else:
            d2s = self.data[iy1:iy2 + 1, ix1:ix2 + 1]
        yy, xx = np.mgrid[iy1:iy2 + 1, ix1:ix2 + 1]
        # tck = interpolate.bisplrep(xx,yy,d2s,s=0)
        points = np.vstack((xx.flatten(), yy.flatten())).T
        values = d2s.flatten()

        box_x0, box_y0 = np.meshgrid(np.arange(2 * w - 1) - (w - 1), np.arange(2 * h - 1) - (h - 1))
        phi *= np.pi / 180
        box_x = box_x0 * np.cos(phi) - box_y0 * np.sin(phi) + cx
        box_y = box_x0 * np.sin(phi) + box_y0 * np.cos(phi) + cy
        # ii = interpolate.bisplev(box_x[:,0],box_y[0,:],tck).sum()
        ii = interpolate.griddata(points, values, (box_x, box_y), method='cubic').sum()
        return ii

    def roi_cnt(self, cx, cy, w, h, phi, show=False):
        """ return the total counts in the ROI
        cx,cy is the center and w,h specify the size (2w-1)x(2h-1)
        phi is the orientation of the width of the box from x-axis, CCW
        useful calculating for line profile on a curve
        NOTE: PIL image rotate about the center of the image
        NOTE: potential problem: crop wraps around the image if cropping near the edge
        """
        # get a larger box
        t = int(np.sqrt(w * w + h * h) + 1)
        imroi = self.im.crop((np.int(cx - t + 1), np.int(cy - t + 1), np.int(cx + t), np.int(cy + t)))
        # rotate 
        imroi = imroi.rotate(-phi, Image.BILINEAR)
        # get the roi
        imroi = imroi.crop((np.int(t - w + 1), np.int(t - h + 1), np.int(t + w), np.int(t + h)))
        dd = np.asarray(imroi.convert("I"))
        if show:
            plt.figure()
            ax = plt.gca()
            ax.imshow(dd, interpolation='nearest')
        return dd.sum()

    def profile_xyphi(self, xy_grid, w, h, bkg=0):
        """ the shape of xy_grid should be (3,N)
        if bkg=1: subtract the counts in the box next to the width as bkg
        if bkg=-1: subtract the counts in the box next to the height as bkg        """

        ixy = []
        for [x, y, phi] in xy_grid:
            ii = self.roi_cnt(x, y, w, h, phi)
            if bkg == 1:
                iib = (self.roi_cnt(x, y, w * 2, h, phi) - ii) / (w * 2) * (w * 2 - 1)
                ii -= iib
            elif bkg == -1:
                iib = (self.roi_cnt(x, y, w, h * 2, phi) - ii) / (h * 2) * (h * 2 - 1)
                ii -= iib
            ixy.append(ii)
        # print ixy
        return np.asarray(ixy)

    def profile_qphi(self, q_grid, w, h, phi):
        """ q_grid is 1D, phi specify the direction of the cut
        calls profile_xyphi(self,xy_grid,w,h)
        """
        d_xyphi = []
        for qq in q_grid:
            (tx, ty) = self.qphi2xy(qq, phi)
            d_xyphi.append([tx, ty, phi])

        return self.profile_xyphi(np.asarray(d_xyphi), w, h)

    def profile_xyphi2(self, xy_grid, w, h, sub_bkg=True, use_qdata=False):
        """ the shape of xy_grid should be (3,N)
        if bkg=1: subtract the counts in the box next to the width as bkg
        if use_qdata=1, xy_grid specifies the trajectory in qr-qz
        """

        if use_qdata and not self.q_data_available:
            print("reciprocal space data not available.")
            return

        ixy = []
        for [x, y, phi] in xy_grid:
            ii = self.roi(x, y, w, h, phi, use_qdata)
            if sub_bkg:
                iib = (self.roi(x, y, w * 2, h, phi, use_qdata) - ii) / (w * 2) * (w * 2 - 1)
                ii -= iib
            ixy.append(ii)
        # print ixy
        return np.asarray(ixy)

    def qrqz2xy(self, qr, qz):
        """calls the C-code in RQconv
        need to deal with the special situation when the (qr, qz) is not visible on the detector
        use the smallest allowable qr at the same qz instead
        this is done in RQconv, with the last argument in RQconv.qrqz2xy set to 1
        """

        ret = RQconv.qrqz2xy(self.data.astype(np.int32), self.exp, qr, qz, 1)
        return ret.x, ret.y

    def qphi2xy(self, q, phi):
        """calls the C-code in RQconv
        """
        # print q,phi
        ret = RQconv.qphi2xy(self.data, self.exp, q, phi)
        return ret.x, ret.y

    def xy2qrqz(self, x, y):
        """calls the C-code in RQconv
        """
        ret = RQconv.xy2qrqz(self.data, self.exp, x, y)
        return ret.x, ret.y

    def xy2q(self, x, y):
        """calls the C-code in RQconv
        """
        return RQconv.xy2q(self.data, self.exp, x, y)

    def zinger(self, x, y, w=3, tol=3):
        avg, std = self.roi_stat(x, y, w, w)
        if (self.data[y, x] - avg) ** 2 > (tol * std) ** 2:
            return avg
        return 0

    def flat_cor(self, dflat, mask=None):
        """ dflat should be a 2D array (float) with the same shape as mask and self.data 
            assume that dflat has been normalized already: values near 1
            also assume that the values behind the mask are 1
        """
        if not (self.data.shape == dflat.shape and self.data.shape == mask.map.shape):
            print("cannot perform flat field correction, shape mismatch:", self.shape, dflat.shape, mask.map.shape)
        dm = (1 - mask.map) * dflat
        # do not touch masked parts
        index = (dm > 0)
        self.data[index] *= np.average(dm[index]) / dm[index]

    def cor_IAdep_2D(self, mask=None, corCode=3, invert=False):
        """ if invert==True, the data is mulitplied by the correction factor, instead of being divided by
            this is useful for obtaining the correction factor itself for each pixel
        """
        dm = np.ones((self.height, self.width), np.int32)
        if mask is not None:
            dm *= (1 - mask.map) * self.data
        else:
            dm *= self.data
        RQconv.cor_IAdep_2D(dm, self.exp, corCode, invert)
        self.data = dm

    def conv_to_Iq(self, qidi, mask, dz=True, w=3, tol=3, cor=0):
        """ convert solution scattering/powder diffraction data into 1D scattering curve
        the q axis is given by grid (1d array)
        calls the C-code in RQconv
        the cor parameter can take positive or negative values
        if cor>0, the 1D data will be corrected for (divided by) the factor due to polarization
        and the non-normal incident X-rays onto the detector
        if cor<0, the 1D data will be multipled by this factor instead. this is useful to
        build this correction into the flat field correction
        """
        # apply the mask before passing the data to RQconv
        # RQconv should discard all data with zero intensity
        dm = np.zeros((self.height, self.width), np.int32) + 1
        # dm = 1-np.asarray(mask.map,np.int32)
        if dz:
            print("dezinger ...")
            RQconv.dezinger(self.data.astype(np.int32), dm, w, tol)
        # NOTE: use self.data+1, instead of self.data, to avoid confusion betwee
        # zero-count pixels and masked pixels. The added 1 count will be subtracted in RQconv
        # print (dm*2-1==0).any()
        # dm = (dm*2-1)*self.data
        dm = np.multiply(dm, (1 - mask.map) * (self.data + 1)).astype(np.int32)
        # plt.figure()
        # plt.imshow(dm)
        RQconv.conv_to_Iq(dm, self.exp, qidi, cor)

    def conv_to_Iqrqz(self):
        # calls the C-code in RQconv
        self.q_data_available = True
        if self.qdata is not None:
            del self.qdata
        RQconv.pre_conv_Iqrqz(self.data.astype(np.int32), self.exp)
        self.qdata = np.ones((self.exp.nz, self.exp.nr), dtype=np.int32)
        RQconv.conv_to_Iqrqz(self.data.astype(np.int32), self.qdata, self.exp)

    def scale(self, sf):
        print(self.data)
        print("sf: ", sf)
        self.data = np.multiply(self.data, sf)

    def subtract(self, darray):
        if not (self.data.shape == darray.shape):
            print("cannot subtract 2D data of different shapes:", self.shape, darray.shape)
        self.data = np.subtract(self.data, darray)

    def add(self, darray):
        """ self = self + dset  
        """
        if not (self.data.shape == darray.shape):
            print("cannot add 2D data of different shapes:", self.shape, darray.shape)
            return False
        else:
            self.data += darray
            return True

    def merge2(self, dset):
        """ intended to merge dset with self, doesn't work, see notes in RQconv.c
        """
        if not self.data.shape == dset.data.shape:
            print("merging 2D data sets have different shapes:")
            print(self.shape, " and ", dset.data.shape, "\n")
            exit()
        RQconv.merge(self.data, dset.data)


def avg_images(image_list):
    if len(image_list) < 1:
        print("List of image is empty.\n")
        return None
    d1 = Data2d(image_list[0])
    ct = 1
    for img in image_list:
        d2 = Data2d(img)
        if d1.add(d2.data):
            ct += 1
        del d2
    d1.scale(1. / ct)
    return d1


def avg_d2sets(d2sets):
    davg = d2sets[0].copy()
    ct = 0
    davg.data *= 0
    for dset in d2sets:
        davg.add(dset)
        ct += 1
    davg.scale(1. / ct)
    return davg


def reduced_cmap(cmap, step):
    return np.array(cmap(step)[0:3])


def cmap_map(function, cmap):
    """ from scipy cookbook 
    Applies function (which should operate on vectors of shape 3: [r, g, b], on colormap cmap. 
    This routine will break any discontinuous points in a colormap.
    """
    cdict = cmap._segmentdata
    step_dict = {}
    # Firt get the list of points where the segments start or end
    for key in ('red', 'green', 'blue'):
        step_dict[key] = [x[0] for x in cdict[key]]
    step_list = reduce(lambda x, y: x + y, list(step_dict.values()))
    step_list = np.array(list(set(step_list)))
    # Then compute the LUT, and apply the function to the LUT
    # reduced_cmap = lambda step: np.array(cmap(step)[0:3])
    # old_lut = np.array(list(map(reduced_cmap, step_list)))
    # new_lut = np.array(list(map(function, old_lut)))

    old_lut = np.array([reduced_cmap(cmap, s) for s in step_list])
    new_lut = np.array([function(s) for s in step_list])

    # Now try to make a minimal segment definition of the new LUT
    cdict = {}
    for i, key in enumerate(('red', 'green', 'blue')):
        this_cdict = {}
        for j, step in enumerate(step_list):
            if step in step_dict[key]:
                this_cdict[step] = new_lut[j, i]
            elif new_lut[j, i] != old_lut[j, i]:
                this_cdict[step] = new_lut[j, i]
        colorvector = [x + (x[1],) for x in list(this_cdict.items())]
        colorvector.sort()
        cdict[key] = colorvector

    return mc.LinearSegmentedColormap('colormap', cdict, 1024)


class Axes2dplot:
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
