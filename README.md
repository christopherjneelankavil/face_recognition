# Facial Recognition in Google Colab

This project demonstrates how to implement facial recognition using the `face_recognition` library in Google Colab. The goal is to detect faces in uploaded images and compare a test image with a reference image to see if they match. The matching is done using cosine similarity and face distance.

## Requirements

- Google Colab (cloud-based notebook platform)
- Python 3.x
- `face_recognition` library
- `numpy` library
- `opencv-python` library

## Setup

1. Open Google Colab: [Google Colab](https://colab.research.google.com/)
2. Create a new Python notebook.
3. Install the required libraries by running the following code in a code cell:

```python
!pip install face_recognition numpy opencv-python
