# Install dependencies
!pip install face_recognition numpy opencv-python

import face_recognition
import numpy as np
import cv2
from google.colab import files
from PIL import Image

# Upload images
print("Upload the reference image:")
uploaded_ref = files.upload()
ref_image_path = list(uploaded_ref.keys())[0]

print("Upload the test image:")
uploaded_test = files.upload()
test_image_path = list(uploaded_test.keys())[0]

# Load images
ref_image = face_recognition.load_image_file(ref_image_path)
test_image = face_recognition.load_image_file(test_image_path)

# Encode faces
ref_encoding = face_recognition.face_encodings(ref_image)
test_encoding = face_recognition.face_encodings(test_image)

if len(ref_encoding) == 0 or len(test_encoding) == 0:
    print("No face detected in one of the images.")
else:
    ref_encoding = ref_encoding[0]
    test_encoding = test_encoding[0]

    # Compute cosine similarity
    cosine_similarity = np.dot(ref_encoding, test_encoding) / (np.linalg.norm(ref_encoding) * np.linalg.norm(test_encoding))

    # Compare faces using `face_recognition`
    result = face_recognition.compare_faces([ref_encoding], test_encoding)
    distance = face_recognition.face_distance([ref_encoding], test_encoding)[0]

    print(f"Cosine Similarity: {cosine_similarity:.4f}")
    print(f"Face Distance: {distance:.4f}")
    print("Match Result:", "Match" if result[0] else "No Match")
