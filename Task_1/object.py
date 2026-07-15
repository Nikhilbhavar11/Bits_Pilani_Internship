import cv2
import matplotlib.pyplot as plt

# Load image
img = cv2.imread("people.jpg")
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Load pre-trained face detector (comes with OpenCV, no download needed)
detector = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

# Detect faces
faces = detector.detectMultiScale(gray, 1.1, 4)
print(f"Faces found: {len(faces)}")

# Draw boxes around faces
for (x, y, w, h) in faces:
    cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)

# Show result
plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
plt.title("Object Detection - Faces")
plt.show()