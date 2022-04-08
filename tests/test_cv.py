
from cv.image_tools import ImageTools


def test_load_image():
    analyzer = ImageTools()
    analyzer.set_in_path("./image")
    analyzer.open("empire.jpg")
    analyzer.display()
    analyzer.convert_L()
    assert(analyzer.data is not None)
