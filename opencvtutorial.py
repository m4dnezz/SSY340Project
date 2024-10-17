import cv2 as cv
from CNN import ConvNeuralNet
import torch
from PIL import Image
from torchvision import transforms

num_classes = 7
model = ConvNeuralNet(num_classes)
model.load_state_dict(torch.load('best_model.pth'))
model.eval()
emotion_labels = ['angry' , 'disgust' , 'fear' , 'happy' ,'neutral' ,'sad' ,'surprise']

face_classifier = cv.CascadeClassifier(
    cv.data.haarcascades + "haarcascade_frontalface_default.xml"
)

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

def detect_bounding_box(vid,model):
    gray_image = cv.cvtColor(vid, cv.COLOR_BGR2GRAY)
    faces = face_classifier.detectMultiScale(gray_image, 1.1, 5, minSize=(40, 40))
    for (x, y, w, h) in faces:
        face_frame = frame[y:y+h , x:x+w]
        face_tensor = process_frame(face_frame)
        cv.rectangle(vid, (x, y), (x + w, y + h), (0, 255, 0), 4) # draw the rectangle

        output = model(face_tensor) # make ze prediction
        predicted_class = torch.argmax(output, dim=1)
        predicted_emotion = emotion_labels[predicted_class.item()]

        cv.putText(frame, predicted_emotion, (x, y-10), cv.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)


if not cap.isOpened():
    print("Cannot open camera")
    exit()
while True:
    # Capture frame-by-frame
    ret, frame = cap.read()
    detect_bounding_box(frame,model) # this function does everything lol


    # if frame is read correctly ret is True
    if not ret:
        print("Can't receive frame (stream end?). Exiting ...")
        break
    # Our operations on the frame come here
    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    # Display the resulting frame
    cv.imshow('frame', gray)
    if cv.waitKey(1) == ord('q'):
        break

# When everything done, release the capture
cap.release()
cv.destroyAllWindows()
