from threading import Thread
from cv2 import VideoCapture, rectangle, putText, FONT_HERSHEY_SIMPLEX,imshow, waitKey, destroyAllWindows
from torch.hub import load

def main():
    url = 'http://192.168.0.47:4040/video'
    capture = VideoCapture(url)

    model = load('ultralytics/yolov5', 'yolov5s', pretrained=True)

    # model = load('path/repo', 'custom', path='path/model.pt')

    while True:
        ret, frame = capture.read()
        if ret:
            results = model(frame)
            for detection in results.xyxy[0]:
                if detection[4] > 0.65:

                    # Colocar retangulo
                    rectangle(frame, (int(detection[0]), int(detection[1])), (int(detection[2]), int(detection[3])), (120,60,0), 1)

                    # Pegar o nome e colocar no retangulo
                    class_index = int(detection[5])
                    class_name = results.names[class_index]
                    putText(frame, class_name, (int(detection[0]), int(detection[1])-10), FONT_HERSHEY_SIMPLEX, 0.9, (120, 60, 0), 2)

            imshow('Agro Photo', frame)
            if waitKey(1) & 0xFF == ord('k'):
                break
        else:
            break

    capture.release()
    destroyAllWindows()

thread = Thread(target=main).start()