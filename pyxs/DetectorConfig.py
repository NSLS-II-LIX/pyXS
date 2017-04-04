

class DetectorConfig():
    def __init__(self, extension = "", dark = None, flat = None, dezinger = False, exp_para = None, qgrid = None, mask = "",
                 fix_scale=None):
        self.dark = dark
        self.flat = flat
        self.dezinger = dezinger
        self.extension = extension
        self.exp_para = exp_para
        self.qgrid = qgrid
        self.mask = mask
        self.fix_scale = fix_scale

