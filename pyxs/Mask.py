import cv2
import numpy as np


class Mask:
    """ a bit map to determine whehter a pixel should be included in
    azimuthal average of a 2D scattering pattern
    """

    def __init__(self, width, height):
        self.width = width
        self.height = height
        self.maskfile = ""
        self.map = None

    def reload(self):
        self.read_file(self.maskfile)

    def read_file(self, filename):
        self.maskfile = filename
        mask_map = np.zeros((self.height, self.width), dtype=np.bool)
        for line in open(filename):
            fields = line.strip().split()
            if len(fields) < 2:
                continue
            stype = fields[0]
            if stype in ['h', 'c', 'r', 'f', 'p']:
                para = [float(t) for t in fields[1:]]
                mask_map = self.add_item(mask_map, stype, para)
        self.map = mask_map.astype(np.bool)

    def invert(self):
        """ invert the mask
        """
        self.map = 1 - self.map

    @staticmethod
    def pairwise(iterable):
        "s -> (s0,s1), (s2,s3), (s4, s5), ..."
        a = iter(iterable)
        return zip(a, a)

    @staticmethod
    def add_item(mask_map, stype, para):
        print("Add Item: ", stype, para)
        tmap = np.zeros(mask_map.shape, dtype=np.uint8)

        if stype == 'c':
            # filled circle
            # c  x  y  r
            (x, y, r) = para
            cv2.circle(tmap, (int(x), int(y)), int(r), color=1, thickness=-1)
        elif stype == 'h':
            # inverse of filled circle
            # h  x  y  r
            (x, y, r) = para
            cv2.rectangle(tmap, (0, 0), tmap.shape, 1, -1)
            cv2.circle(tmap, (int(x), int(y)), int(r), color=0, thickness=-1)
        elif stype == 'r':
            # rectangle
            # r  x  y  w  h  rotation
            (x, y, w, h, rot) = para

            margin = (np.asarray(mask_map.shape) / 2).astype(np.int)
            tmap = np.zeros(mask_map.shape + 2*margin, dtype=np.uint8)
            pt1 = (int(tmap.shape[1]/2 - w/2), int(tmap.shape[0]/2 - h/2))
            pt2 = (int(tmap.shape[1]/2 + w/2), int(tmap.shape[0]/2 + h/2))
            cv2.rectangle(tmap, pt1, pt2, color=1, thickness=-1)

            import scipy.ndimage.interpolation as interp
            tmap = interp.rotate(tmap, rot, reshape=False)

            x_offset = int(x + margin[1] - tmap.shape[1] / 2 + 0.5)
            y_offset = int(y + margin[0] - tmap.shape[0] / 2 + 0.5)

            tmap = np.roll(tmap, x_offset, axis=1)
            tmap = np.roll(tmap, y_offset, axis=0)

            crop_params = (margin[0], margin[1], mask_map.shape[0] + margin[0], mask_map.shape[1] + margin[1])
            tmap = tmap[crop_params[0]:crop_params[2], crop_params[1]:crop_params[3]]

        elif stype == 'f':
            pass
            # fan
            # f  x  y  start  end  r1  r2  (r1<r2)
            (x, y, a_st, a_nd, r1, r2) = para
            cv2.ellipse(tmap, (x, y), (r1, r2), a_st, a_nd, 1, -1)
        elif stype == 'p':
            # polygon
            # p  x1  y1  x2  y1  x3  y3  ...
            points = np.array([[x,y] for x,y in Mask.pairwise(para)], dtype=np.int32)
            points = points.reshape((-1, 1, 2))
            cv2.fillConvexPoly(tmap, points, color=1)

        mask_map = np.logical_or(mask_map, tmap)
        del tmap
        return mask_map

    def clear(self):
        self.map = np.zeros(self.map.shape, dtype=np.bool)

    def val(self, x, y):
        return self.map[y][x]
