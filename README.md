# SmartVision — CIFAR-10 Image Classifier

An end-to-end **image classification** project built with **PyTorch**, **Torchvision**, and **Gradio**. SmartVision trains a compact TinyVGG-style convolutional neural network (CNN) to recognize ten object categories in CIFAR-10 images and provides an interactive web interface for single-image predictions.

> **Project status:** Model training, evaluation, and local Gradio testing completed. Hugging Face Spaces deployment is being prepared.

## Project overview

SmartVision is a **multiclass classifier**, not a regression model. Given an RGB image, it predicts a probability for each of the ten CIFAR-10 classes and displays the three highest-scoring predictions alongside the time required to process the image.

**Input:** RGB image, resized to **32 × 32 pixels** during preprocessing.  
**Output:** Class probabilities for airplane, automobile, bird, cat, deer, dog, frog, horse, ship, and truck.

The project demonstrates the full machine-learning workflow: dataset preparation, model design, training, validation, testing, inference, and web application development.

## Dataset and data preparation

The project uses the official **CIFAR-10** dataset, loaded with `torchvision.datasets.CIFAR10`.

| Split | Images | Purpose |
| --- | ---: | --- |
| Training | 45,000 | Fit the network parameters |
| Validation | 5,000 | Monitor generalization and select the best checkpoint |
| Test | 10,000 | Final evaluation on held-out images |
| **Total** | **60,000** | |

The original 50,000-image CIFAR-10 training portion was split into training and validation sets with a fixed random seed (`42`). The official 10,000-image test set was kept separate from training and validation.

- **Preprocessing during training:** `transforms.ToTensor()`.
- **Training batch size:** 32; training DataLoader uses shuffling.
- **Validation and test loaders:** Batch size 32, without shuffling.
- **Uploaded images:** Converted to RGB, resized to 32 × 32, and converted to tensors before inference.

The dataset is **not committed to this repository**. The notebook downloads it automatically through Torchvision.

## Model architecture

SmartVision uses a compact **TinyVGG** CNN with two convolutional blocks and a linear classification head:

```text
Input RGB image (3 × 32 × 32)
  │
  ├─ Conv2d(3 → 10, kernel_size=3) → ReLU
  ├─ Conv2d(10 → 10, kernel_size=3) → ReLU
  ├─ MaxPool2d(kernel_size=2)
  │
  ├─ Conv2d(10 → 10, kernel_size=3) → ReLU
  ├─ Conv2d(10 → 10, kernel_size=3) → ReLU
  ├─ MaxPool2d(kernel_size=2)
  │
  ├─ Flatten (10 × 5 × 5 = 250 features)
  └─ Linear(250 → 10 class scores)
```

The convolution layers use stride 1 and padding 0. The final model outputs **10 logits**, which are converted to class probabilities with softmax during inference.

## Training configuration

| Setting | Value |
| --- | --- |
| Framework | PyTorch |
| Model | TinyVGG |
| Loss function | CrossEntropyLoss |
| Optimizer | Adam |
| Learning rate | 0.001 |
| Maximum epochs | 10 |
| Early-stopping patience | 3 epochs |
| Minimum improvement (`min_delta`) | 0.001 |
| Checkpoint selection | Lowest validation loss |
| Training hardware | Google Colab T4 GPU |

Training completed all **10 epochs** because validation loss continued to improve. The best-performing checkpoint was saved as `smartvision_best_model.pth`.

At epoch 10, the recorded results were:

| Metric | Value |
| --- | ---: |
| Training loss | 1.2204 |
| Training accuracy | 56.85% |
| Validation loss | 1.2277 |
| Validation accuracy | 56.11% |

The training notebook includes the training and validation loss/accuracy curves.

## Final test-set evaluation

The saved checkpoint was evaluated on the untouched **10,000-image CIFAR-10 test set**.

| Metric | Result |
| --- | ---: |
| **Accuracy** | **57.44%** |
| Weighted precision | 57.40% |
| Weighted recall | 57.44% |
| Weighted F1-score | 57.05% |

The notebook also contains a per-class classification report and a confusion matrix. Performance varies across categories: the model recognizes some classes more reliably than others, and visually similar animals are frequently confused.

**Interpretation:** SmartVision demonstrates a complete, functioning image-classification pipeline, but its predictive accuracy is moderate. It is an educational baseline rather than a production-grade image-recognition system.

## Gradio web application

The Gradio interface lets users:

1. Upload a JPG or PNG image.
2. Run the saved PyTorch model on the image.
3. See the top three predicted classes and their probabilities.
4. View prediction time in seconds.

For example, one test in the Colab-hosted Gradio interface predicted **automobile** with approximately **90%** probability. The displayed processing time for that run was approximately **0.00193 seconds**. Timing varies by hardware and runtime conditions; this is an example, not a guaranteed benchmark.

The standalone deployment application is configured to run on **CPU**, so it does not require a GPU on Hugging Face Spaces.

## Repository structure

```text
SmartVision-CIFAR10/
├── notebooks/
│   └── SmartVision_CIFAR10.ipynb     # Training, evaluation, and experiments
├── SmartVision_Deployment/
│   ├── app.py                       # Standalone Gradio application
│   ├── model_builder.py             # TinyVGG network definition
│   ├── smartvision_best_model.pth   # Saved trained model weights
│   ├── requirements.txt             # Runtime dependencies
│   └── README.md                    # Hugging Face Spaces configuration
├── .gitignore
└── README.md                        # This project overview
```

> The notebook filename shown above is illustrative; use the actual `.ipynb` filename in the `notebooks` folder if it differs.

## Run the project locally

**Requirements:** Python with the packages listed in `SmartVision_Deployment/requirements.txt`.

```bash
# Clone the repository
git clone https://github.com/Arthur5412/SmartVision-CIFAR10.git
cd SmartVision-CIFAR10

# Install the application's dependencies
pip install -r SmartVision_Deployment/requirements.txt

# Start the web application
cd SmartVision_Deployment
python app.py
```

Open the local URL printed by Gradio in your terminal (typically `http://127.0.0.1:7860`). The deployment directory must contain `app.py`, `model_builder.py`, and `smartvision_best_model.pth` together.

To reproduce training and evaluation, open the notebook in Google Colab and execute its cells in order. Google Drive paths used to save checkpoints may need to be adjusted for another user's environment.

## Deployment

The standalone Gradio application is prepared for **Hugging Face Spaces**. The Space repository should contain the **contents** of `SmartVision_Deployment/` at its root, including its own `README.md` with the Spaces metadata.

**Live demo:** Link will be added after deployment.

**Source repository:** https://github.com/Arthur5412/SmartVision-CIFAR10

## Limitations and future improvements

- CIFAR-10 images are only **32 × 32 pixels**, so resizing larger photographs can discard important visual detail.
- The compact TinyVGG model has limited capacity and achieves **57.44%** test accuracy.
- Predictions on photographs outside the CIFAR-10 distribution may be unreliable.
- Future work could evaluate data augmentation, transfer learning, deeper architectures, and improved hyperparameter tuning using the validation set.

## Technologies

**Python · PyTorch · Torchvision · Google Colab · Gradio · Hugging Face Spaces · Git · GitHub**

## Acknowledgment

This project was developed as an educational end-to-end PyTorch image-classification exercise, adapting the TinyVGG-style architecture and training workflow used in the course materials.
