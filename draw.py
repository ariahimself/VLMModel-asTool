import torch
import tkinter as tk
from tkinter import messagebox, filedialog
import numpy as np
from PIL import Image, ImageDraw
import os

# Import the model classes from main2.py
from main2 import VisionTransformer, predict_and_show

class DigitDrawer:
    def __init__(self, model_path='vit_mnist_model.pth'):
        self.model_path = model_path
        self.model = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.load_model()

        # Drawing parameters
        self.canvas_size = 280  # 10x scale for easier drawing
        self.image_size = 28
        self.brush_size = 8  # Larger brush for easier drawing
        self.last_x = None
        self.last_y = None

        self.setup_gui()

    def load_model(self):
        if os.path.exists(self.model_path):
            self.model = VisionTransformer().to(self.device)
            self.model.load_state_dict(torch.load(self.model_path, map_location=self.device))
            self.model.eval()
            print("Model loaded successfully!")
        else:
            print(f"Model file {self.model_path} not found. Please train the model first.")

    def setup_gui(self):
        self.root = tk.Tk()
        self.root.title("MNIST Digit Drawer - Draw and Predict!")
        self.root.geometry("600x700")

        # Create main frame
        main_frame = tk.Frame(self.root)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # Title
        title_label = tk.Label(main_frame, text="Draw a Digit (0-9)", font=("Arial", 16, "bold"))
        title_label.pack(pady=(0, 10))

        # Canvas frame
        canvas_frame = tk.Frame(main_frame, relief=tk.SUNKEN, borderwidth=2)
        canvas_frame.pack(pady=(0, 10))

        # Drawing canvas
        self.canvas = tk.Canvas(canvas_frame, width=self.canvas_size, height=self.canvas_size,
                               bg='white', cursor='crosshair')
        self.canvas.pack()

        # Create PIL image for drawing
        self.image = Image.new('L', (self.canvas_size, self.canvas_size), 255)  # White background
        self.draw = ImageDraw.Draw(self.image)

        # Bind mouse events
        self.canvas.bind('<B1-Motion>', self.draw_line)
        self.canvas.bind('<ButtonPress-1>', self.start_drawing)
        self.canvas.bind('<ButtonRelease-1>', self.stop_drawing)

        # Button frame
        button_frame = tk.Frame(main_frame)
        button_frame.pack(fill=tk.X, pady=(0, 10))

        # Buttons
        self.predict_button = tk.Button(button_frame, text="Predict Digit", command=self.predict_digit,
                                       font=("Arial", 12), bg='#4CAF50', fg='white', height=2)
        self.predict_button.pack(side=tk.LEFT, padx=(0, 10), expand=True, fill=tk.X)

        self.clear_button = tk.Button(button_frame, text="Clear Canvas", command=self.clear_canvas,
                                     font=("Arial", 12), bg='#f44336', fg='white', height=2)
        self.clear_button.pack(side=tk.LEFT, padx=(0, 10), expand=True, fill=tk.X)

        self.save_button = tk.Button(button_frame, text="Save Image", command=self.save_image,
                                    font=("Arial", 12), bg='#2196F3', fg='white', height=2)
        self.save_button.pack(side=tk.LEFT, expand=True, fill=tk.X)

        # Result frame
        result_frame = tk.Frame(main_frame)
        result_frame.pack(fill=tk.X)

        self.result_label = tk.Label(result_frame, text="Draw a digit and click 'Predict Digit'",
                                    font=("Arial", 14), fg='#666666')
        self.result_label.pack(pady=(10, 0))

        # Instructions
        instructions = """
Instructions:
• Click and drag to draw smoothly
• Use 'Predict Digit' to see what the model thinks it is
• Use 'Clear Canvas' to start over
• Use 'Save Image' to save your drawing

The drawing area is scaled up for easier drawing!
        """
        instr_label = tk.Label(main_frame, text=instructions, font=("Arial", 10),
                              justify=tk.LEFT, fg='#666666')
        instr_label.pack(pady=(20, 0))

    def start_drawing(self, event):
        self.last_x = event.x
        self.last_y = event.y

    def draw_line(self, event):
        if self.last_x and self.last_y:
            # Draw on canvas
            self.canvas.create_line(self.last_x, self.last_y, event.x, event.y,
                                   width=self.brush_size, fill='black', capstyle=tk.ROUND, smooth=True)

            # Draw on PIL image
            self.draw.line([self.last_x, self.last_y, event.x, event.y],
                          fill=0, width=self.brush_size)  # 0 is black

        self.last_x = event.x
        self.last_y = event.y

    def stop_drawing(self, event):
        self.last_x = None
        self.last_y = None

    def clear_canvas(self):
        self.canvas.delete('all')
        self.image = Image.new('L', (self.canvas_size, self.canvas_size), 255)
        self.draw = ImageDraw.Draw(self.image)
        self.result_label.config(text="Draw a digit and click 'Predict Digit'", fg='#666666')

    def save_image(self):
        if self.model is None:
            messagebox.showerror("Error", "Model not loaded!")
            return

        file_path = filedialog.asksaveasfilename(
            defaultextension=".png",
            filetypes=[("PNG files", "*.png"), ("All files", "*.*")]
        )
        if file_path:
            self.image.save(file_path)
            messagebox.showinfo("Success", f"Image saved to {file_path}")

    def predict_digit(self):
        if self.model is None:
            messagebox.showerror("Error", "Model not loaded! Please train the model first.")
            return

        try:
            # Resize image to 28x28
            resized_image = self.image.resize((self.image_size, self.image_size), Image.LANCZOS)

            # Convert to numpy array and normalize
            image_array = np.array(resized_image) / 255.0  # Scale to 0-1

            # Invert colors (white background to black, black drawing to white)
            image_array = 1.0 - image_array

            # Convert to tensor and apply MNIST normalization
            image_tensor = torch.from_numpy(image_array).float().unsqueeze(0)  # Add channel dimension
            image_tensor = (image_tensor - 0.1307) / 0.3081  # MNIST normalization

            # Predict
            prediction = predict_and_show(self.model, image_tensor, self.device)

            # Update result label
            self.result_label.config(text=f"Predicted Digit: {prediction}", fg='#4CAF50', font=("Arial", 16, "bold"))

        except Exception as e:
            messagebox.showerror("Error", f"Prediction failed: {str(e)}")

    def run(self):
        self.root.mainloop()

def main():
    drawer = DigitDrawer()
    drawer.run()

if __name__ == '__main__':
    main()