
import time
import torch
import gradio as gr
from pathlib import Path
from torchvision import transforms
from model_builder import TinyVGG

# --------------------------------------------------
# 1. Configure the application
# --------------------------------------------------

# Hugging Face Spaces can run this model on CPU
device = "cpu"

# CIFAR-10 object categories
class_names = [
    "airplane", "automobile", "bird", "cat",
    "deer", "dog", "frog", "horse",
    "ship", "truck"
]

# --------------------------------------------------
# 2. Load the trained TinyVGG model
# --------------------------------------------------

model = TinyVGG(
    input_shape=3,
    hidden_units=10,
    output_shape=len(class_names)
)

# Find model weights in the same folder as app.py
model_path = (
    Path(__file__).resolve().parent
    / "smartvision_best_model.pth"
)

# Load the trained model weights
model.load_state_dict(
    torch.load(
        model_path,
        map_location=device,
        weights_only=True
    )
)

# Put the model in evaluation mode
model.eval()

# --------------------------------------------------
# 3. Define image preprocessing
# --------------------------------------------------

# Match the image preprocessing used during training
predict_transform = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.ToTensor()
])

# --------------------------------------------------
# 4. Create the prediction function
# --------------------------------------------------

def predict_image(image):

    # Handle cases where no image is uploaded
    if image is None:
        return {}, 0.0

    # Start measuring prediction time
    start_time = time.perf_counter()

    # Ensure the uploaded image has 3 RGB channels
    image = image.convert("RGB")

    # Convert the image into a model-compatible tensor
    image_tensor = predict_transform(image)
    image_tensor = image_tensor.unsqueeze(0).to(device)

    # Generate predictions without calculating gradients
    with torch.inference_mode():
        prediction_logits = model(image_tensor)

        # Convert class scores into probabilities
        prediction_probs = torch.softmax(
            prediction_logits,
            dim=1
        )

    # Build a dictionary of class probabilities
    prediction_dict = {
        class_names[i]: float(prediction_probs[0][i])
        for i in range(len(class_names))
    }

    # Calculate elapsed prediction time
    prediction_time = time.perf_counter() - start_time

    return prediction_dict, prediction_time

# --------------------------------------------------
# 5. Build the Gradio web interface
# --------------------------------------------------

demo = gr.Interface(
    fn=predict_image,

    inputs=gr.Image(
        type="pil",
        label="Upload an Image"
    ),

    outputs=[
        gr.Label(
            num_top_classes=3,
            label="Predicted Object Classes"
        ),
        gr.Number(
            label="Prediction Time (seconds)"
        )
    ],

    title="SmartVision — CIFAR-10 Image Classifier",

    description=(
        "Upload an image to classify it into one of "
        "10 CIFAR-10 categories using a PyTorch "
        "TinyVGG convolutional neural network."
    )
)

# --------------------------------------------------
# 6. Launch the application
# --------------------------------------------------

if __name__ == "__main__":
    demo.launch()
