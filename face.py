import streamlit as st
import cv2
import os
import tempfile
import torch
from torchvision import transforms
from facenet_pytorch import MTCNN, InceptionResnetV1
import numpy as np

def main():
    st.title("Face Recognition Demo")

    # Load models
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    mtcnn = MTCNN(device=device)
    resnet = InceptionResnetV1(pretrained='vggface2').eval().to(device)

    uploaded_image = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
    if uploaded_image is not None:
        # Read the uploaded image
        image = cv2.imdecode(np.frombuffer(uploaded_image.read(), np.uint8), 1)

        # Detect faces in the uploaded image
        boxes, _ = mtcnn.detect(image)

        if boxes is not None and len(boxes) > 0:
            box = boxes[0]
            face = image[int(box[1]):int(box[3]), int(box[0]):int(box[2])]
            transform = transforms.Compose([
                transforms.ToPILImage(),
                transforms.Resize((160, 160)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
            ])
            face_tensor = transform(face).unsqueeze(0).to(device)
            embedding_image = resnet(face_tensor).detach().cpu().numpy()[0]

            st.image(image, caption="Uploaded Image", use_column_width=True)

            uploaded_video = st.file_uploader("Upload a video", type=["mp4"])
            if uploaded_video is not None:
                # Save the uploaded video to a temporary file
                with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp_file:
                    tmp_file.write(uploaded_video.read())
                    video_path = tmp_file.name

                # Open the video with cv2.VideoCapture() using its file path
                cap = cv2.VideoCapture(video_path)
                if cap.isOpened():
                    while True:
                        ret, frame = cap.read()
                        if not ret:
                            break

                        # Detect faces in the current frame
                        boxes, _ = mtcnn.detect(frame)

                        if boxes is not None and len(boxes) > 0:
                            box = boxes[0]
                            face = frame[int(box[1]):int(box[3]), int(box[0]):int(box[2])]
                            face_tensor = transform(face).unsqueeze(0).to(device)
                            embedding_video = resnet(face_tensor).detach().cpu().numpy()[0]

                            # Calculate the cosine similarity between the embeddings
                            similarity = np.dot(embedding_image, embedding_video) / (
                                        np.linalg.norm(embedding_image) * np.linalg.norm(embedding_video))
                            similarity_percentage = round(similarity * 100, 2)

                            # Display result
                            if similarity_percentage >= 70:
                                st.write(f"Matching: Yes with {similarity_percentage}% similarity")
                            else:
                                st.write(f"Matching: No with {similarity_percentage}% similarity")

                    cap.release()

                    # Remove the temporary file
                    os.remove(video_path)
                else:
                    st.write("Error: Unable to open the uploaded video.")
        else:
            st.write("No faces detected in the image.")

if __name__ == "__main__":
    main()
