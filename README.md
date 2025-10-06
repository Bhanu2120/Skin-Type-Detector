Skin Type Detector ✨
A full-stack web application that uses a deep learning model to classify your skin type as oily, dry, or normal from a photo. This project features a responsive frontend, a Python/Flask backend, user authentication, and a complete ML pipeline for inference.

Features:

(1) 👤 User Authentication: Secure sign-up and login functionality.

(2) 📸 Image Upload: Users can upload a photo from their gallery or take one directly using their device's camera or webcam.

(3) 🧠 AI-Powered Analysis: A two-step process first detects a face in the image and then classifies the skin type.

(4) 📊 User Profiles & History: Registered users can view their profile and a complete history of their past scans.

(5) 📱 Responsive Design: A clean and modern UI that works seamlessly on both desktop and mobile devices.

Tech Stack:

-> Frontend: HTML5, CSS3, Vanilla JavaScript
-> Backend: Python, Flask, SQLAlchemy
-> Machine Learning: PyTorch, OpenCV
-> Database: PostgreSQL (Production), SQLite (Development)
-> Deployment: Render, Gunicorn
-> Version Control: Git & Git LFS for handling the large model file.

How It Works:

The prediction pipeline is a two-stage process designed for accuracy:

. Face Detection: When an image is uploaded, a pre-trained Caffe-based Single Shot-Multibox Detector (SSD) model running on OpenCV's DNN module first detects and crops the user's face from the image.
. Skin Classification: The cropped face image is then pre-processed and passed to the primary skin type classification model, which is a fine-tuned ResNet model built with PyTorch. The model outputs the final prediction (oily, dry, or normal) along with a confidence score.
