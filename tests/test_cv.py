
from cv.image_tools import ImageTools


def test_load_image():
    analyzer = ImageTools()
    analyzer.open("./images/empire.jpg")
    assert(analyzer.data is not None)
