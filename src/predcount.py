import cv2
from ultralytics import YOLO

def predict(chosen_model, img, classes=[], conf=0.5):
    if classes:
        results = chosen_model.predict(img, classes=classes, conf=conf)
    else:
        results = chosen_model.predict(img, conf=conf)
    return results

def predict_and_detect(chosen_model, img, classes=[], conf=0.4, rectangle_thickness=2, text_thickness=1):
    results = predict(chosen_model, img, classes, conf=conf)
    for result in results:
        for box in result.boxes:
            cv2.rectangle(img, (int(box.xyxy[0][0]), int(box.xyxy[0][1])),
                          (int(box.xyxy[0][2]), int(box.xyxy[0][3])), (255, 0, 0), rectangle_thickness)
            cv2.putText(img, f"{result.names[int(box.cls[0])]}",
                        (int(box.xyxy[0][0]), int(box.xyxy[0][1]) - 10),
                        cv2.FONT_HERSHEY_PLAIN, 1, (255, 0, 0), text_thickness)
    return img, results

def create_video_writer(video_cap, output_filename):
    frame_width = int(video_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(video_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(video_cap.get(cv2.CAP_PROP_FPS))
    fourcc = cv2.VideoWriter_fourcc(*'MP4V')
    writer = cv2.VideoWriter(output_filename, fourcc, fps, (frame_width, frame_height))
    return writer

model = YOLO("/Users/rianrachmanto/miniforge3/yolov11n_custom/train2/weights/best.pt")
output_filename = "/Users/rianrachmanto/miniforge3/project/gamification_wellness/test_vid/result.mp4"
video_path = r"/Users/rianrachmanto/miniforge3/project/gamification_wellness/test_vid/3526401275-preview.mp4"

cap = cv2.VideoCapture(video_path)
writer = create_video_writer(cap, output_filename)

repetition_counter = 0
state = 0  # Start with State 0: No exercise detected

while True:
    success, img = cap.read()
    if not success:
        break
    result_img, results = predict_and_detect(model, img, classes=[], conf=0.5)

    current_detected = None
    if results:
        detected_classes = [result.names[int(box.cls[0])] for result in results for box in result.boxes]
        if 'push-ups' in detected_classes:
            current_detected = 'push-ups'
        elif 'push-downs' in detected_classes:
            current_detected = 'push-downs'

    if state == 0 and current_detected == 'push-ups':
        state = 1
    elif state == 1 and current_detected == 'push-downs':
        state = 2
    elif state == 2 and current_detected == 'push-ups':
        repetition_counter += 1
        state = 1

    cv2.putText(result_img, f"Repetitions: {repetition_counter}", (10, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

    writer.write(result_img)
    cv2.imshow("Image", result_img)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
writer.release()
cv2.destroyAllWindows()
