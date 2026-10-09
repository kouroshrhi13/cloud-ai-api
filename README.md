# Cloud AI API

A cloud-hosted image classification application built with **FastAPI** and **PyTorch**, powered by the **MobileNetV3-Small** deep learning model.

The application allows users to upload an image through a web interface and receive a predicted ImageNet class label along with its confidence score.

## Live Demo

**Try the application:** https://cloud-ai.fastapicloud.dev

**Source Code:** https://github.com/kouroshrhi13/cloud-ai-api

## Features

* **Image Classification:** Predicts image categories using a pretrained MobileNetV3-Small model.
* **REST API:** Provides an image prediction endpoint built with FastAPI.
* **Web Interface:** Upload images and view prediction results through a simple browser-based interface.
* **Confidence Score:** Displays the model's confidence for the predicted class.
* **Cloud Deployment:** Deployed online so users can access the application from different devices, including smartphones.
* **Efficient Inference:** Uses CPU-based inference with PyTorch.

## Tech Stack

* **Language:** Python
* **Backend:** FastAPI
* **Deep Learning:** PyTorch, Torchvision
* **Image Processing:** Pillow
* **Model:** MobileNetV3-Small
* **Deployment:** FastAPI Cloud

## How It Works

1. The user uploads an image through the web interface.
2. The backend receives the image through the FastAPI application.
3. The image is preprocessed using the model's standard transformations.
4. MobileNetV3-Small performs image classification.
5. The application returns the predicted class label and confidence score.

## API Endpoint

### `POST /predict`

Accepts an image upload and returns the predicted class and confidence score.

Example request using `curl`:

```bash
curl -X POST "https://cloud-ai.fastapicloud.dev/predict" \
  -F "file=@image.jpg"
```

Replace `image.jpg` with the path to your own image file.

> Note: The prediction is based on ImageNet classes. The confidence score represents the model's predicted probability for the selected class, not a guarantee that the prediction is correct.

## Run Locally

### 1. Clone the repository

```bash
git clone https://github.com/kouroshrhi13/cloud-ai-api.git
cd cloud-ai-api
```

### 2. Create and activate a virtual environment

**Windows:**

```bash
python -m venv venv
venv\Scripts\activate
```

**Linux / macOS:**

```bash
python -m venv venv
source venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Start the application

```bash
fastapi dev main.py
```

Open the local URL shown in the terminal to access the application.

If the local development server starts on its default address, the API documentation is available at:

`http://127.0.0.1:8000/docs`

## Project Structure

```text
cloud-ai-api/
├── main.py
├── model.py
├── requirements.txt
├── models/
│   └── mobilenet_v3_small.pth
└── static/
    └── index.html
```

## Future Improvements

* Automatic image resizing and compression for large uploads.
* Improved error handling and input validation.
* API performance monitoring and optimization.
* Additional testing and documentation.

## Author

**Kourosh Ruhi**

Computer Engineering Student | Python & AI Developer

GitHub: https://github.com/kouroshrhi13

---

*Built as a practical cloud AI demonstration project.*
