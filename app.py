import streamlit as st
import cv2
import io
import torch
from torchvision import transforms
from PIL import Image
from facenet_pytorch import MTCNN, InceptionResnetV1
import numpy as np
import tempfile

# Load models
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
mtcnn = MTCNN(device=device)
resnet = InceptionResnetV1(pretrained='vggface2').eval().to(device)

# Function to detect faces and calculate embeddings
def detect_faces_and_calculate_similarity(image_bytes, video_bytes):
    # Convert image bytes to PIL Image
    image = Image.open(io.BytesIO(image_bytes)).convert('RGB')

    # Detect faces in the still image
    boxes, _ = mtcnn.detect(image)

    if boxes is not None and len(boxes) > 0:
        # Take the first detected face
        box = boxes[0]
        face = image.crop((box[0], box[1], box[2], box[3]))
        transform = transforms.Compose([
            transforms.Resize((160, 160)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ])
        face_tensor = transform(face).unsqueeze(0).to(device)
        embedding_image = resnet(face_tensor).detach().cpu().numpy()[0]

        # Process the video frames
        try:
            # Save video bytes to a temporary file
            with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as temp_video:
                temp_video.write(video_bytes)
                temp_video_path = temp_video.name

            cap = cv2.VideoCapture(temp_video_path)

            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                # Detect faces in the current frame
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                boxes, _ = mtcnn.detect(frame)

                if boxes is not None and len(boxes) > 0:
                    # Take the first detected face
                    box = boxes[0]
                    face = frame[int(box[1]):int(box[3]), int(box[0]):int(box[2])]
                    face_pil = Image.fromarray(face)
                    face_tensor = transform(face_pil).unsqueeze(0).to(device)
                    embedding_video = resnet(face_tensor).detach().cpu().numpy()[0]

                    # Calculate the cosine similarity between the embeddings
                    similarity = np.dot(embedding_image, embedding_video) / (np.linalg.norm(embedding_image) * np.linalg.norm(embedding_video))
                    similarity_percentage = round(similarity * 100, 2)

                    # Print output for each frame
                    if similarity_percentage >= 70:
                        st.write(f"Frame: {cap.get(cv2.CAP_PROP_POS_FRAMES)}, Matching: Yes with a similarity score of {similarity_percentage}%")
                        st.write("They are similar")
                    else:
                        st.write(f"Frame: {cap.get(cv2.CAP_PROP_POS_FRAMES)}, Matching: No with a similarity score of {similarity_percentage}%")

            cap.release()
            cv2.destroyAllWindows()

        except Exception as e:
            st.error(f"Error processing video: {e}")

    else:
        st.write("No faces detected in the image.")

# Streamlit UI
st.title("Face Similarity Checker")

# Upload image and video
image_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
video_file = st.file_uploader("Upload a video", type=["mp4"])

if image_file is not None and video_file is not None:
    try:
        # Read the uploaded files
        image_bytes = image_file.read()
        video_bytes = video_file.read()

        # Call function to detect faces and calculate similarity
        detect_faces_and_calculate_similarity(image_bytes, video_bytes)
    except Exception as e:
        st.error(f"Error uploading files: {e}")
