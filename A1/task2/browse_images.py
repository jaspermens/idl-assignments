import tkinter as tk
from PIL import Image, ImageTk
import numpy as np

images = np.load('images.npy')

import tkinter as tk
from PIL import Image, ImageTk
import numpy as np

def show_image(idx, zoom_factor=7):
    original_img = Image.fromarray(images[idx], 'L')  # 'L' specifies grayscale
    width, height = original_img.width, original_img.height
    zoomed_img = original_img.resize((width * zoom_factor, height * zoom_factor))
    img = ImageTk.PhotoImage(zoomed_img)
    canvas.config(width=width*zoom_factor, height=height*zoom_factor)
    canvas.create_image(0, 0, anchor=tk.NW, image=img)
    canvas.image = img

def next_image():
    global current_image
    current_image = (current_image + 1) % len(images)
    show_image(current_image)

def prev_image():
    global current_image
    current_image = (current_image - 1) % len(images)
    show_image(current_image)

root = tk.Tk()
root.title("Image Viewer")

canvas = tk.Canvas(root)
canvas.pack()

current_image = 0
show_image(current_image)

next_button = tk.Button(root, text="Next", command=next_image)
next_button.pack()

prev_button = tk.Button(root, text="Previous", command=prev_image)
prev_button.pack()

root.mainloop()


