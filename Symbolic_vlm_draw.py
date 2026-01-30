import torch
import tkinter as tk
from tkinter import messagebox, filedialog
import numpy as np
from PIL import Image, ImageDraw
import os
import re

# Reuse your ViT model definition (trained in main2.py)
from main2 import VisionTransformer


def _softmax_np(x: np.ndarray) -> np.ndarray:
    x = x - np.max(x)
    ex = np.exp(x)
    return ex / (np.sum(ex) + 1e-12)


def predict_probs(model: torch.nn.Module, image_tensor_1x28x28: torch.Tensor, device: torch.device):
    """
    Returns:
      pred (int): argmax class 0-9
      probs (np.ndarray): shape (10,) probabilities
      logits (np.ndarray): shape (10,) raw logits
    image_tensor_1x28x28: torch.FloatTensor shaped (1, 28, 28), already MNIST-normalized.
    """
    model.eval()
    with torch.no_grad():
        x = image_tensor_1x28x28.unsqueeze(0).to(device)  # -> (1, 1, 28, 28)
        logits_t = model(x).squeeze(0)  # -> (10,)
        logits = logits_t.detach().cpu().numpy().astype(np.float64)
        probs = _softmax_np(logits)
        pred = int(np.argmax(probs))
        return pred, probs, logits


class VLMDigitDrawer:
    """
    "VLM-like" structure (lightweight, no extra training):
      - Vision encoder/classifier (your ViT) predicts digit + probabilities.
      - Language side is a small rule-based prompt router that turns (prompt + prediction)
        into a natural-language answer.

    This gives you: image + text -> text response (VLM-style interface),
    while still using your existing MNIST ViT weights.
    """

    def __init__(self, model_path='vit_mnist_model.pth'):
        self.model_path = model_path
        self.model = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.load_model()

        # Drawing parameters (copied from your draw.py pattern)
        self.canvas_size = 280  # 10x scale
        self.image_size = 28
        self.brush_size = 8
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
        self.root.title("VLM Digit Drawer - Draw + Ask")
        self.root.geometry("720x820")

        main_frame = tk.Frame(self.root)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        title_label = tk.Label(main_frame, text="Draw a Digit (0-9) and Ask a Question",
                               font=("Arial", 16, "bold"))
        title_label.pack(pady=(0, 10))

        canvas_frame = tk.Frame(main_frame, relief=tk.SUNKEN, borderwidth=2)
        canvas_frame.pack(pady=(0, 10))

        self.canvas = tk.Canvas(canvas_frame, width=self.canvas_size, height=self.canvas_size,
                                bg='white', cursor='crosshair')
        self.canvas.pack()

        # PIL image to mirror the canvas
        self.image = Image.new('L', (self.canvas_size, self.canvas_size), 255)
        self.draw = ImageDraw.Draw(self.image)

        self.canvas.bind('<B1-Motion>', self.draw_line)
        self.canvas.bind('<ButtonPress-1>', self.start_drawing)
        self.canvas.bind('<ButtonRelease-1>', self.stop_drawing)

        # Controls: predict / clear / save
        button_frame = tk.Frame(main_frame)
        button_frame.pack(fill=tk.X, pady=(0, 10))

        self.predict_button = tk.Button(
            button_frame, text="Predict Digit", command=self.predict_digit,
            font=("Arial", 12), bg='#4CAF50', fg='white', height=2
        )
        self.predict_button.pack(side=tk.LEFT, padx=(0, 10), expand=True, fill=tk.X)

        self.clear_button = tk.Button(
            button_frame, text="Clear Canvas", command=self.clear_canvas,
            font=("Arial", 12), bg='#f44336', fg='white', height=2
        )
        self.clear_button.pack(side=tk.LEFT, padx=(0, 10), expand=True, fill=tk.X)

        self.save_button = tk.Button(
            button_frame, text="Save Image", command=self.save_image,
            font=("Arial", 12), bg='#2196F3', fg='white', height=2
        )
        self.save_button.pack(side=tk.LEFT, expand=True, fill=tk.X)

        # Prompt area
        prompt_frame = tk.Frame(main_frame)
        prompt_frame.pack(fill=tk.X, pady=(10, 5))

        prompt_label = tk.Label(prompt_frame, text="Ask:", font=("Arial", 12))
        prompt_label.pack(side=tk.LEFT)

        self.prompt_entry = tk.Entry(prompt_frame, font=("Arial", 12))
        self.prompt_entry.pack(side=tk.LEFT, padx=(8, 8), expand=True, fill=tk.X)
        self.prompt_entry.insert(0, "Is this even or odd?")

        self.ask_button = tk.Button(
            prompt_frame, text="Ask (VLM)", command=self.ask_question,
            font=("Arial", 12), bg='#673AB7', fg='white', height=1
        )
        self.ask_button.pack(side=tk.LEFT)

        # Result frames
        result_frame = tk.Frame(main_frame)
        result_frame.pack(fill=tk.X)

        self.pred_label = tk.Label(
            result_frame, text="Draw a digit and click 'Predict Digit' or ask a question.",
            font=("Arial", 14), fg='#666666'
        )
        self.pred_label.pack(pady=(10, 0))

        self.answer_label = tk.Label(
            result_frame, text="Answer will appear here.",
            font=("Arial", 12), fg='#333333', wraplength=680, justify=tk.LEFT
        )
        self.answer_label.pack(pady=(8, 0))

        examples = (
            "Examples you can ask:\n"
            "• What digit is this?\n"
            "• Is this even or odd?\n"
            "• Spell it out\n"
            "• What's the confidence?\n"
            "• Is it greater than 5?\n"
        )
        ex_label = tk.Label(main_frame, text=examples, font=("Arial", 10),
                            justify=tk.LEFT, fg='#666666')
        ex_label.pack(pady=(18, 0))

    # --- drawing handlers ---
    def start_drawing(self, event):
        self.last_x = event.x
        self.last_y = event.y

    def draw_line(self, event):
        if self.last_x is not None and self.last_y is not None:
            self.canvas.create_line(
                self.last_x, self.last_y, event.x, event.y,
                width=self.brush_size, fill='black',
                capstyle=tk.ROUND, smooth=True
            )
            self.draw.line([self.last_x, self.last_y, event.x, event.y], fill=0, width=self.brush_size)

        self.last_x = event.x
        self.last_y = event.y

    def stop_drawing(self, event):
        self.last_x = None
        self.last_y = None

    # --- utility ---
    def clear_canvas(self):
        self.canvas.delete('all')
        self.image = Image.new('L', (self.canvas_size, self.canvas_size), 255)
        self.draw = ImageDraw.Draw(self.image)
        self.pred_label.config(text="Cleared. Draw again.", fg='#666666', font=("Arial", 14))
        self.answer_label.config(text="Answer will appear here.", fg='#333333')

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

    def _preprocess_to_mnist_tensor(self) -> torch.Tensor:
        """
        Matches your existing draw.py preprocessing:
          - resize to 28x28
          - scale to [0,1]
          - invert colors
          - MNIST normalize (mean=0.1307, std=0.3081)
        Returns tensor shaped (1, 28, 28).
        """
        resized = self.image.resize((self.image_size, self.image_size), Image.LANCZOS)
        arr = np.array(resized) / 255.0
        arr = 1.0 - arr
        t = torch.from_numpy(arr).float().unsqueeze(0)          # (1, 28, 28)
        t = (t - 0.1307) / 0.3081
        return t

    def _get_prediction(self):
        if self.model is None:
            raise RuntimeError("Model not loaded!")
        image_tensor = self._preprocess_to_mnist_tensor()
        pred, probs, _ = predict_probs(self.model, image_tensor, self.device)
        return pred, probs

    # --- basic prediction button ---
    def predict_digit(self):
        try:
            pred, probs = self._get_prediction()
            conf = float(probs[pred]) * 100.0
            self.pred_label.config(
                text=f"Predicted Digit: {pred}  (confidence: {conf:.1f}%)",
                fg='#4CAF50',
                font=("Arial", 16, "bold")
            )
            self.answer_label.config(text="(Tip) Type a question above and click Ask.", fg='#333333')
        except Exception as e:
            messagebox.showerror("Error", f"Prediction failed: {str(e)}")

    # --- VLM-ish: prompt -> answer ---
    def ask_question(self):
        if self.model is None:
            messagebox.showerror("Error", "Model not loaded! Please train the model first.")
            return

        prompt = self.prompt_entry.get().strip()
        if not prompt:
            prompt = "What digit is this?"

        try:
            pred, probs = self._get_prediction()
            answer = self._answer_from_prompt(prompt, pred, probs)

            conf = float(probs[pred]) * 100.0
            self.pred_label.config(
                text=f"Vision says: {pred}  (confidence: {conf:.1f}%)",
                fg='#4CAF50',
                font=("Arial", 14, "bold")
            )
            self.answer_label.config(text=answer, fg='#111111')

        except Exception as e:
            messagebox.showerror("Error", f"Ask failed: {str(e)}")

    def _answer_from_prompt(self, prompt: str, digit: int, probs: np.ndarray) -> str:
        """
        A tiny prompt-router: detects intent (odd/even, confidence, spell-out, comparisons, etc.)
        and returns a natural-language answer using the vision prediction.
        """
        p = prompt.lower().strip()

        words = {
            0: "zero", 1: "one", 2: "two", 3: "three", 4: "four",
            5: "five", 6: "six", 7: "seven", 8: "eight", 9: "nine"
        }

        # Helper: top-k
        topk = np.argsort(-probs)[:3]
        topk_str = ", ".join([f"{int(k)} ({probs[k]*100:.1f}%)" for k in topk])

        # 1) parity questions
        if ("even" in p) or ("odd" in p) or ("parity" in p):
            return f"It looks like **{digit}**, which is **{'even' if digit % 2 == 0 else 'odd'}**."

        # 2) "what digit / what number"
        if ("what digit" in p) or ("which digit" in p) or ("what number" in p) or (p in ["digit", "number", "what is this"]):
            return f"I think the digit is **{digit}**."

        # 3) spell-out
        if ("spell" in p) or ("in words" in p) or ("word" in p):
            return f"The digit is **{digit}**, spelled **{words.get(digit, str(digit))}**."

        # 4) confidence / probabilities / top-k
        if ("confidence" in p) or ("sure" in p) or ("prob" in p) or ("top" in p) or ("likelihood" in p):
            return f"Top guesses: {topk_str}. My best guess is **{digit}**."

        # 5) comparisons: greater than / less than / == etc.
        # Extract a number if present
        nums = re.findall(r"-?\d+", p)
        if nums and (("greater" in p) or (">" in p) or ("less" in p) or ("<" in p) or ("equal" in p) or ("==" in p)):
            n = int(nums[0])
            if ("greater" in p) or (">" in p):
                return f"I predict **{digit}**. Is it greater than {n}? **{'Yes' if digit > n else 'No'}**."
            if ("less" in p) or ("<" in p):
                return f"I predict **{digit}**. Is it less than {n}? **{'Yes' if digit < n else 'No'}**."
            if ("equal" in p) or ("==" in p) or ("equals" in p) or ("is it" in p):
                return f"I predict **{digit}**. Is it equal to {n}? **{'Yes' if digit == n else 'No'}**."

        # 6) fallback
        return f"I’m not sure what you meant, but the drawing looks like **{digit}** (top guesses: {topk_str})."


def main():
    app = VLMDigitDrawer()
    app.root.mainloop()


if __name__ == '__main__':
    main()
