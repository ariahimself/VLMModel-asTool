"""
vlm_draw_infer.py

Tkinter drawing app that uses the *trained* MNISTVLM:
- You draw a digit
- You type a question (prompt)
- The model generates a text answer (learned, not rule-based)

Run after training:
  python vlm_draw_infer.py
"""
from __future__ import annotations

import os
import tkinter as tk
from tkinter import messagebox, filedialog
import numpy as np
from PIL import Image, ImageDraw

import torch

from vlm_tokenizer import CharTokenizer
from vlm_model import MNISTVLM, VLMConfig


class VLMDrawerApp:
    def __init__(self, ckpt_path: str = "vlm_mnist_vqa.pth"):
        self.ckpt_path = ckpt_path
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.tokenizer, self.model = self.load_model()

        # Drawing params
        self.canvas_size = 280
        self.image_size = 28
        self.brush_size = 8
        self.last_x = None
        self.last_y = None

        self.setup_gui()

    def load_model(self):
        if not os.path.exists(self.ckpt_path):
            raise FileNotFoundError(
                f"Checkpoint not found: {self.ckpt_path}\n"
                f"Train it first: python vlm_train.py --freeze_vision"
            )

        ckpt = torch.load(self.ckpt_path, map_location="cpu")
        tokenizer = CharTokenizer.from_state_dict(ckpt["tokenizer"])
        cfg = VLMConfig(**ckpt["cfg"])

        model = MNISTVLM(vocab_size=tokenizer.vocab_size, cfg=cfg).to(self.device)
        model.load_state_dict(ckpt["model_state"])
        model.eval()
        print("Loaded VLM checkpoint.")

        return tokenizer, model

    def setup_gui(self):
        self.root = tk.Tk()
        self.root.title("MNIST VLM - Draw + Ask (Learned)")
        self.root.geometry("740x860")

        main = tk.Frame(self.root)
        main.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        title = tk.Label(main, text="Draw a digit and ask a question (learned VLM)", font=("Arial", 16, "bold"))
        title.pack(pady=(0, 10))

        canvas_frame = tk.Frame(main, relief=tk.SUNKEN, borderwidth=2)
        canvas_frame.pack(pady=(0, 10))
        self.canvas = tk.Canvas(canvas_frame, width=self.canvas_size, height=self.canvas_size,
                                bg="white", cursor="crosshair")
        self.canvas.pack()

        self.image = Image.new("L", (self.canvas_size, self.canvas_size), 255)
        self.draw = ImageDraw.Draw(self.image)

        self.canvas.bind("<ButtonPress-1>", self.start_drawing)
        self.canvas.bind("<B1-Motion>", self.draw_line)
        self.canvas.bind("<ButtonRelease-1>", self.stop_drawing)

        # Buttons
        btns = tk.Frame(main)
        btns.pack(fill=tk.X, pady=(0, 10))

        tk.Button(btns, text="Clear", command=self.clear_canvas,
                  font=("Arial", 12), bg="#f44336", fg="white", height=2).pack(side=tk.LEFT, padx=(0, 10), expand=True, fill=tk.X)

        tk.Button(btns, text="Save Image", command=self.save_image,
                  font=("Arial", 12), bg="#2196F3", fg="white", height=2).pack(side=tk.LEFT, padx=(0, 10), expand=True, fill=tk.X)

        tk.Button(btns, text="Ask", command=self.ask,
                  font=("Arial", 12), bg="#673AB7", fg="white", height=2).pack(side=tk.LEFT, expand=True, fill=tk.X)

        # Prompt
        pframe = tk.Frame(main)
        pframe.pack(fill=tk.X, pady=(10, 5))
        tk.Label(pframe, text="Prompt:", font=("Arial", 12)).pack(side=tk.LEFT)

        self.prompt_entry = tk.Entry(pframe, font=("Arial", 12))
        self.prompt_entry.pack(side=tk.LEFT, padx=(8, 8), expand=True, fill=tk.X)
        self.prompt_entry.insert(0, "Is this even or odd?")

        # Output
        self.answer_label = tk.Label(main, text="Answer will appear here.", font=("Arial", 13),
                                     wraplength=700, justify=tk.LEFT, fg="#111111")
        self.answer_label.pack(pady=(12, 0))

        examples = (
            "Examples:\n"
            "• What digit is this?\n"
            "• Is this even or odd?\n"
            "• Spell it out.\n"
            "• Is it greater than 5?\n"
            "• Is it less than 5?\n"
        )
        tk.Label(main, text=examples, font=("Arial", 10), justify=tk.LEFT, fg="#666666").pack(pady=(18, 0))

    # Drawing handlers
    def start_drawing(self, event):
        self.last_x = event.x
        self.last_y = event.y

    def draw_line(self, event):
        if self.last_x is not None and self.last_y is not None:
            self.canvas.create_line(self.last_x, self.last_y, event.x, event.y,
                                    width=self.brush_size, fill="black",
                                    capstyle=tk.ROUND, smooth=True)
            self.draw.line([self.last_x, self.last_y, event.x, event.y], fill=0, width=self.brush_size)
        self.last_x = event.x
        self.last_y = event.y

    def stop_drawing(self, event):
        self.last_x = None
        self.last_y = None

    def clear_canvas(self):
        self.canvas.delete("all")
        self.image = Image.new("L", (self.canvas_size, self.canvas_size), 255)
        self.draw = ImageDraw.Draw(self.image)
        self.answer_label.config(text="Cleared. Draw again.")

    def save_image(self):
        file_path = filedialog.asksaveasfilename(defaultextension=".png",
                                                 filetypes=[("PNG files", "*.png"), ("All files", "*.*")])
        if file_path:
            self.image.save(file_path)
            messagebox.showinfo("Saved", f"Saved to {file_path}")

    def _preprocess_drawn(self) -> torch.Tensor:
        """
        Same idea as your draw.py:
        resize->invert->normalize as MNIST.
        Returns: (1,1,28,28)
        """
        resized = self.image.resize((self.image_size, self.image_size), Image.LANCZOS)
        arr = np.array(resized, dtype=np.float32) / 255.0
        arr = 1.0 - arr
        t = torch.from_numpy(arr).float().unsqueeze(0).unsqueeze(0)  # (1,1,28,28)
        t = (t - 0.1307) / 0.3081
        return t

    @torch.no_grad()
    def ask(self):
        prompt = self.prompt_entry.get().strip()
        if not prompt:
            prompt = "What digit is this?"

        try:
            img = self._preprocess_drawn().to(self.device)

            input_text = f"question: {prompt.lower().strip()}\nanswer: "
            input_ids = torch.tensor([self.tokenizer.encode(input_text, add_bos=True, add_eos=False)],
                                     dtype=torch.long, device=self.device)

            out_ids = self.model.generate(
                images=img,
                input_ids=input_ids,
                eos_id=self.tokenizer.eos_id,
                max_new_tokens=24,
            )
            # Decode only the generated tail (skip the prompt part)
            decoded_full = self.tokenizer.decode(out_ids[0].tolist(), stop_at_eos=True)
            # decoded_full includes prompt text; we extract after "answer:"
            if "answer:" in decoded_full:
                answer = decoded_full.split("answer:", 1)[1].strip()
            else:
                answer = decoded_full.strip()

            if not answer:
                answer = "(empty)"
            self.answer_label.config(text=f"Model answer: {answer}")

        except Exception as e:
            messagebox.showerror("Error", f"Failed: {e}")


def main():
    app = VLMDrawerApp()
    app.root.mainloop()


if __name__ == "__main__":
    main()
