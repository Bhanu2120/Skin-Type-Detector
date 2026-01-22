Skin Type Detector ✨

A full-stack web application that uses a deep learning model to classify your skin type as oily, dry, or normal from a photo. This project features a responsive frontend, a Python/Flask backend, user authentication, and a complete ML pipeline for inference.

Features:

(1) 👤 User Authentication: Secure sign-up and login functionality.

(2) 📸 Image Upload: Users can upload a photo from their gallery or take one directly using their device's camera or webcam.

(3) 🧠 AI-Powered Analysis: A two-step process first detects a face in the image and then classifies the skin type.

(4) 📊 User Profiles & History: Registered users can view their profile and a complete history of their past scans.

(5) 📱 Responsive Design: A clean and modern UI that works seamlessly on both desktop and mobile devices.

Tech Stack:

-> Frontend: HTML5, CSS3, JavaScript

-> Backend: Python, Flask, SQLAlchemy

-> Machine Learning: PyTorch, OpenCV

-> Database: PostgreSQL (Production), SQLite (Development)

-> Deployment: Render, Gunicorn

-> Version Control: Git & Git LFS for handling the large model file.

How It Works:

The prediction pipeline is a two-stage process designed for accuracy:

. Face Detection: When an image is uploaded, a pre-trained Caffe-based Single Shot-Multibox Detector (SSD) model running on OpenCV's DNN module first detects and crops the user's face from the image.

. Skin Classification: The cropped face image is then pre-processed and passed to the primary skin type classification model, which is a fine-tuned ResNet model built with PyTorch. The model outputs the final prediction (oily, dry, or normal) along with a confidence score.


App Screenshots:

<img width="1853" height="1037" alt="Screenshot 2026-01-22 100607" src="https://github.com/user-attachments/assets/ab462821-dfe4-4581-b593-a73dbb771c3c" />

<img width="1860" height="1042" alt="Screenshot 2026-01-22 100642" src="https://github.com/user-attachments/assets/0658b9b5-6758-4c06-aa96-9ae507b953ad" />

<img width="1859" height="1039" alt="Screenshot 2026-01-22 100904" src="https://github.com/user-attachments/assets/0cd149c9-673c-4c5f-ad77-18a579b14128" />

<img width="1858" height="1030" alt="Screenshot 2026-01-22 101149" src="https://github.com/user-attachments/assets/847b63b6-56e0-4a8d-83e4-9933236a6fd0" />

<img width="1859" height="1027" alt="Screenshot 2026-01-22 101223" src="https://github.com/user-attachments/assets/5f66177d-a802-4b26-808a-14671192aec6" />

<img width="1860" height="1048" alt="Screenshot 2026-01-22 101312" src="https://github.com/user-attachments/assets/87366625-0ad8-4d54-95c8-52301f957540" />

<img width="1855" height="1032" alt="Screenshot 2026-01-22 101349" src="https://github.com/user-attachments/assets/64871d5a-5bb0-4e61-96f3-34a933a3f207" />

<img width="1871" height="1069" alt="Screenshot 2026-01-22 103727" src="https://github.com/user-attachments/assets/83ced454-c166-4155-a579-0cb923659083" />

<img width="1848" height="1029" alt="Screenshot 2026-01-22 103758" src="https://github.com/user-attachments/assets/a66ad919-5d51-4bbb-8fd0-7cc39d616ab3" />

<img width="1848" height="1030" alt="Screenshot 2026-01-22 103821" src="https://github.com/user-attachments/assets/433906d9-7da6-406b-b94b-a988f2657cec" />










