class ImgException(Exception):
    def __init__(self, msg):
        self.msg = msg

    def __str__(self):
        return self.msg


class RoiException(Exception):
    def __init__(self, msg):
        self.msg = msg

    def __str__(self):
        return self.msg
