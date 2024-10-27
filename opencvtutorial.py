from yoloface.face_detector import YoloDetector
from CNN import ConvNeuralNet
from PIL import Image
from torchvision import transforms
import cv2 as cv
import torch
import sys
import torch.nn.functional as F
sys.path.append('yoloface')

num_classes = 7
model = ConvNeuralNet(num_classes)
model.load_state_dict(torch.load('best_model.pth'))
model.eval()
emotion_labels = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']


face_model = YoloDetector(target_size=720, device="cuda:0", min_face=2)  # import pre-trained yolov5 face-detection

transform = transforms.Compose([
    transforms.Resize((48, 48)),
    transforms.Grayscale(num_output_channels=1),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5])
])

cap = cv.VideoCapture(0)


def process_frame(frame):
    pil_img = Image.fromarray(cv.cvtColor(frame, cv.COLOR_BGR2RGB))
    img_tensor = transform(pil_img).unsqueeze(0)

    return img_tensor


def detect_bounding_box(vid, model_emotion):
    bboxes, _ = face_model.predict(vid)
    if bboxes:
        for bbox in bboxes:
            for box in bbox:
                if len(box) == 4:  # we find 2 points use em! else dont do any prediction
                    x, y, w, h = box
                    face_frame = vid[y:h, x:w]  # slice frame of
                    face_tensor = process_frame(face_frame)

                    cv.rectangle(vid, (x, y), (w, h), (0, 255, 0), 8)  # deaw rectanlge around face

                    output = model_emotion(face_tensor)
                    probabilities = F.softmax(output, dim=1)
                    predicted_class = torch.argmax(output, dim=1)
                    predicted_emotion = emotion_labels[predicted_class.item()]

                    cv.putText(vid, predicted_emotion, (x, y - 10), cv.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
                    confidence = probabilities[0, predicted_class.item()].item() * 100  # Convert to percentage

# Display the predicted emotion and confidence on the video frame
                    cv.putText(vid, f'{predicted_emotion}: {confidence:.2f}%', (x, y - 10),
                               cv.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
                else:
                    print("Invalid bounding box format:", box)
    else:
        print("No bounding boxes detected.")

    return vid


if not cap.isOpened():
    print("Cannot open camera")
    exit()
while True:
    # Capture frame-by-frame
    ret, frame = cap.read()
    detect_bounding_box(frame, model)  # this function does everything lol

    # if frame is read correctly ret is True
    if not ret:
        print("Can't receive frame (stream end?). Exiting ...")
        break
    # Our operations on the frame come here
    # Display the resulting frame
    cv.imshow('frame', frame)
    if cv.waitKey(1) == ord('q'):
        break

# When everything done, release the capture
cap.release()
cv.destroyAllWindows()
