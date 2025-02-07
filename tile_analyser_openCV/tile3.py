import cv2
import numpy as np
import tkinter as tk
from tkinter import filedialog, messagebox
from tkinter import ttk
from PIL import Image, ImageTk
import os


class CeramicTileAnalyzer:
    def __init__(self, root):
        self.root = root
        self.root.title("Ceramic Tile Analyzer")

        # Initialize variables
        self.image_paths = []
        self.selected_image = None
        self.first_color = None
        self.second_color = None
        self.processed_image = None

        # GUI Layout
        self.create_widgets()

    def create_widgets(self):
        # Frame for image selection
        self.image_frame = ttk.Frame(self.root)
        self.image_frame.pack(side=tk.LEFT, padx=10, pady=10)

        # Button to open directory and load images
        self.load_button = ttk.Button(self.image_frame, text="Load Images", command=self.load_images)
        self.load_button.pack()

        # Listbox to show images in directory
        self.image_listbox = tk.Listbox(self.image_frame, width=30, height=10)
        self.image_listbox.pack()
        self.image_listbox.bind('<<ListboxSelect>>', self.on_image_select)

        # Frame for original image display
        self.original_frame = ttk.Frame(self.root)
        self.original_frame.pack(side=tk.LEFT, padx=10, pady=10)

        # Canvas to display original image
        self.original_canvas = tk.Canvas(self.original_frame, width=800, height=600, bg="white")
        self.original_canvas.pack()

        # Frame for processed image display
        self.processed_frame = ttk.Frame(self.root)
        self.processed_frame.pack(side=tk.LEFT, padx=10, pady=10)

        # Canvas to display processed image
        self.processed_canvas = tk.Canvas(self.processed_frame, width=800, height=600, bg="white")
        self.processed_canvas.pack()

        # Button to export processed image
        self.export_button = ttk.Button(self.processed_frame, text="Export Image", command=self.export_image)
        self.export_button.pack(pady=10)

    def load_images(self):
        """ Opens file dialog and loads images into the listbox """
        directory = filedialog.askdirectory()
        if directory:
            self.image_paths = [f"{directory}/{f}" for f in os.listdir(directory) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
            self.image_listbox.delete(0, tk.END)
            for path in self.image_paths:
                self.image_listbox.insert(tk.END, path)

    def on_image_select(self, event):
        """ Called when user selects an image from the listbox """
        selected_index = self.image_listbox.curselection()
        if selected_index:
            selected_image_path = self.image_paths[selected_index[0]]
            self.selected_image = cv2.imread(selected_image_path)
            self.show_selected_image()

    def show_selected_image(self):
        """ Displays the selected image on the original image canvas """
        if self.selected_image is not None:
            # Convert to RGB for displaying with PIL
            image_rgb = cv2.cvtColor(self.selected_image, cv2.COLOR_BGR2RGB)
            image_pil = Image.fromarray(image_rgb)
            image_tk = ImageTk.PhotoImage(image_pil)
            
            # Display the original image in the original_canvas
            self.original_canvas.create_image(0, 0, anchor=tk.NW, image=image_tk)
            self.original_canvas.image = image_tk

            # Create an empty white processed image (same size as the original)
            self.create_processed_image()

    def create_processed_image(self):
        """ Initialize an empty processed image with a white background """
        if self.selected_image is not None:
            self.processed_image = np.ones_like(self.selected_image) * 255

    def highlight_color(self, color, highlight_value):
        """ Highlight the given color in processed image with the specified value (black or grey) """
        if self.processed_image is not None:
            # Define the tolerance for color matching
            lower_bound = np.array([max(0, c - 20) for c in color])  # Lower bound for color range
            upper_bound = np.array([min(255, c + 20) for c in color])  # Upper bound for color range
            
            # Create a mask for pixels within the color range
            mask = cv2.inRange(self.selected_image, lower_bound, upper_bound)
            
            # Apply the mask to highlight the selected color in the processed image
            if highlight_value == 0:  # Black for first color
                self.processed_image[mask == 255] = (0, 0, 0)  # Set pixels to black where the mask is 255
            elif highlight_value == 128:  # Grey for second color (lighter grey)
                self.processed_image[mask == 255] = (169, 169, 169)  # Set pixels to grey (RGB: 169,169,169)
            
            # Refresh the display of the processed image
            self.show_processed_image()

    def show_processed_image(self):
        """ Displays the processed image with highlights on the canvas """
        if self.processed_image is not None:
            processed_image_rgb = cv2.cvtColor(self.processed_image, cv2.COLOR_BGR2RGB)
            processed_image_pil = Image.fromarray(processed_image_rgb)
            processed_image_tk = ImageTk.PhotoImage(processed_image_pil)

            # Display the processed image in the processed_canvas
            self.processed_canvas.create_image(0, 0, anchor=tk.NW, image=processed_image_tk)
            self.processed_canvas.image = processed_image_tk

    def export_image(self):
        """ Export the current processed image as PNG """
        if self.processed_image is not None:
            save_path = filedialog.asksaveasfilename(defaultextension=".png", filetypes=[("PNG files", "*.png")])
            if save_path:
                cv2.imwrite(save_path, self.processed_image)
                messagebox.showinfo("Export", "Image successfully exported!")


if __name__ == "__main__":
    root = tk.Tk()
    app = CeramicTileAnalyzer(root)
    root.mainloop()

