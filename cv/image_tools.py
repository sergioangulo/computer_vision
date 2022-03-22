from PIL import Image


class ImageTools():
    def __init__(self):
        self.data = None
    
    def open(self, image_path):
        self.data = Image.open(image_path)
        