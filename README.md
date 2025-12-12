# Skin Type Detector (Flask + EfficientNet)

A web application that predicts skin types using an uploaded facial image.  
This project uses a PyTorch EfficientNet-B0 model served with a Flask backend,  
and includes LocalStorage-based history tracking.

---

## Model Classes
The trained model classifies facial skin images into:

- **Dry**
- **Normal**
- **Oily**

---

## Features
- Image upload for skin type detection.
- EfficientNet-B0 model for classification.
- Flask backend with a lightweight web interface.
- Frontend history tracking using LocalStorage.

---

## Installation

Install the required packages:

```bash
pip install flask torch torchvision pillow
```

---

## Running the Application

Create the uploads directory
```bash
mkdir -p static/uploads
```

Start the Flask server
```bash
python app.py
```

Open your browser and go to
```bash
http://127.0.0.1:5000
```
---

## Project Structure

```text
skin-type-detector/
│
├── app.py  # Main Flask application
│
├── models/
│   └── classification_model_best.pth  # Trained EfficientNet model
│
├── static/
│   ├── uploads/  # Uploaded images are stored here
│   ├── style.css  # Styling
│
├── templates/
│   └── index.html  # Main web page
│
└── README.md  # Project documentation
```
