# MNIST VLM (ViT) + Handwritten Digit Predictor 🖊️🤖

This project trains a small Vision Transformer (ViT) on the MNIST dataset and then lets you **draw a handwritten digit** (or load one) and have the model **predict the number (0–9)**.

✅ Train a ViT on MNIST  
✅ Save/load the trained model (`vit_mnist_model.pth`)  
✅ Tkinter drawing app to sketch digits and run inference  
🚧 Future goal: a **LangChain agent** that uses this VLM as a tool to “look” at a handwritten image and return the answer (in progress)

---

## Project Structure

```text
.
├── main2.py              # ViT model + training + evaluation + predict_and_show()
├── digit_drawer.py       # Tkinter GUI for drawing, saving, and predicting digits
├── vit_mnist_model.pth   # Saved model weights (generated after training)
└── data/                 # MNIST dataset download location
