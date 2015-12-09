import pyxs.ext.RQconv as RQconv
import scipy.misc
import numpy as np
from scipy import interpolate
import matplotlib.pyplot as plt

import fabio

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
        self.im = fabio.open(filename).data
        if flip:
            self.im = np.fliplr(np.rot90(self.im))
        # convert into an array
        # copy() seems to be necessary for later alteration of the 2D data
        self.data = np.asarray(self.im).copy()
        self.height, self.width = self.data.shape
        self.q_data_available = False
        self.qdata = None
        # self.exp = RQconv.ExpPara()

    def set_exp_para(self, exp):
        if exp.flip:
            self.im = np.fliplr(np.rot90(self.im))
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
        #imroi = self.im.crop((np.int(cx - t + 1), np.int(cy - t + 1), np.int(cx + t), np.int(cy + t)))

        imroi = self.im[np.int(cx - t + 1):np.int(cy - t + 1), np.int(cx + t):np.int(cy + t)]
        # rotate 
        imroi = scipy.misc.imrotate(imroi, -phi, 'bilinear')
        # get the roi
        #imroi = imroi.crop((np.int(t - w + 1), np.int(t - h + 1), np.int(t + w), np.int(t + h)))
        imroi = imroi[np.int(t - w + 1):np.int(t - h + 1), np.int(t + w):np.int(t + h)]


        if show:
            plt.figure()
            ax = plt.gca()
            ax.imshow(imroi, interpolation='nearest')
        return imroi.sum()

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
