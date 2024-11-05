import pydrake
import numpy as np
import pydot
import matplotlib.pyplot as plt
from PIL import Image
from io import BytesIO


def save_diagram(diagram, file: str = 'output/system_diagram.svg'):
    pngfile = pydot.graph_from_dot_data(
        diagram.GetGraphvizString(max_depth=2)
        )[0].create_svg()
    
    with open(file,'wb') as png_file:
        png_file.write(pngfile)


def display_diagram(diagram, max_depth=2):
    dot_data = diagram.GetGraphvizString(max_depth=max_depth)
    
    png_data = pydot.graph_from_dot_data(dot_data)[0].create_png()
    
    image_stream = BytesIO(png_data)
    img = Image.open(image_stream)
    
    plt.figure(figsize=(10, 10))
    plt.imshow(img)
    plt.axis('off')
    plt.show()