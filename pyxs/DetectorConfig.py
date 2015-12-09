

class DetectorConfig():
    def __init__(self, extension = "", dark = None, flat = None, dezinger = False):
        self.dark = dark
        self.flat = flat
        self.dezinger = dezinger
        self.extension = extension

