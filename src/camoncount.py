import cv2
from ultralytics import YOLO

def predict(chosen_model, img, classes=[], conf=0.5):
    if classes:
        results = chosen_model.predict(img, classes=classes, conf=conf)
    else:
        results = chosen_model.predict(img, conf=conf)
    return results

def predict_and_detect(chosen_model, img, classes=[], conf=0.5, rectangle_thickness=2, text_thickness=1):
    results = predict(chosen_model, img, classes, conf=conf)
    for result in results:
        for box in result.boxes:
            cv2.rectangle(img, (int(box.xyxy[0][0]), int(box.xyxy[0][1])),
                          (int(box.xyxy[0][2]), int(box.xyxy[0][3])), (255, 0, 0), rectangle_thickness)
            cv2.putText(img, f"{result.names[int(box.cls[0])]}",
                        (int(box.xyxy[0][0]), int(box.xyxy[0][1]) - 10),
                        cv2.FONT_HERSHEY_PLAIN, 1, (255, 0, 0), text_thickness)
    return img, results

# Load YOLO model
model = YOLO("/Users/rianrachmanto/miniforge3/yolov11n_custom/train2/weights/best.pt")

# Initialize the video capture from the built-in camera (index 0 for default camera)
cap = cv2.VideoCapture(1)
if not cap.isOpened():
    print("Error: Could not open video capture.")
    exit()

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

    # Update the state based on the detected exercise
    if state == 0 and current_detected == 'push-ups':
        state = 1
    elif state == 1 and current_detected == 'push-downs':
        state = 2
    elif state == 2 and current_detected == 'push-ups':
        repetition_counter += 1  # Increase count on the transition from push-downs to push-ups
        state = 1

    # Display the repetition count on the image
    cv2.putText(result_img, f"Repetitions: {repetition_counter}", (10, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

    cv2.imshow("Result", result_img)
    
    # Press 'q' to quit the loop
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
