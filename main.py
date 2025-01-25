from ultralytics import YOLO
import cv2
import cvzone
import math
from sort import *

cap = cv2.VideoCapture("chapter-5-running-yolo/videos/people.mp4") # for videos


model = YOLO("..yolo-weights/yolov8n.pt")


classNames = model.names 


# Tracking
tracker = Sort(max_age = 20, min_hits = 3, iou_threshold = 0.3)

limits_up = [0, 100, 640, 100]
limits_down = [0, 300, 640, 300]
limits_left = [150, 0, 150, 360]
limits_right = [450, 0, 450, 360]

totalCount = []
down = []
up = []
left = []
right = []


fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter('Pedestrian_Count.mp4', fourcc, 24, (640,360))

# ----------------------------------------------------- Infinite Loop ------------------------------------------------------ #

while cap.isOpened():
    success, img = cap.read()
    # imgRegion = cv2.bitwise_and(img, car_mask)
    if not success:
        print("Failed to capture image")
        break

    results = model(img, stream = True)

    detections = np.empty((0, 5))
    for r in results:
        boxes = r.boxes
        for box in boxes:
            # Bounding Box
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 =  int(x1), int(y1), int(x2), int(y2)
            # print(x1, y1, x2, y2)

            

            # Confidence & Object Names
            conf = math.ceil((box.conf[0]*100))/100
            cls = int(box.cls[0])
            currentClass = classNames[cls]
            text = f'{classNames[cls]} {conf}'

            (w, h), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)

            # cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 255), 3)
            # cv2.rectangle(img, (x1, y1-20), (x1+w, y1), (255, 0, 255), -1)
            # # Debugging information
            # print(f'cls: {cls}, len(classNames): {len(classNames)}')
            # cv2.putText(img, f'{classNames[cls]} {conf}', (max(0,x1),max(20, y1)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

            if (currentClass == "person") and conf>0.3:
                currentArray = np.array([x1, y1, x2, y2, conf])
                detections = np.vstack((detections, currentArray))
                # cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 255), 3)
                # cv2.rectangle(img, (x1, y1-20), (x1+w, y1), (255, 0, 255), -1)
                # # Debugging information
                # print(f'cls: {cls}, len(classNames): {len(classNames)}')
                # cv2.putText(img, f'{classNames[cls]} {conf}', (max(0,x1),max(20, y1)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)


            #box.xywh
    resultsTracker = tracker.update(detections)

    # ---------------------------------------- Drawing Four Lines for -------------------------------------------- #

    cv2.line(img, (limits_up[0], limits_up[1]), (limits_up[2], limits_up[3]), (0, 0, 255), 2)
    cv2.line(img, (limits_down[0], limits_down[1]), (limits_down[2], limits_down[3]), (255, 0, 0), 2)
    cv2.line(img, (limits_right[0], limits_right[1]), (limits_right[2], limits_right[3]), (0, 0, 255), 2)
    cv2.line(img, (limits_left[0], limits_left[1]), (limits_left[2], limits_left[3]), (255, 0, 0), 2)

    # ---------------------------------------- Drawing Four Lines for -------------------------------------------- #

    for results in resultsTracker:
        x1, y1, x2, y2, Id = results
        x1, y1, x2, y2, Id =  int(x1), int(y1), int(x2), int(y2), int(Id)
        cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 0), 3)
        (w1, h1), _ = cv2.getTextSize(f"{Id}", cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
        # cv2.rectangle(img, (x1+80, y1-20), (x1+80+w1, y1), (255, 0, 0), -1)
        # cv2.putText(img, f'{Id}', (max(0,x1+80),max(20, y1)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 3)

        cx, cy = int(x1+(x2-x1)//2), int(y1+(y2-y1)//2)
        cv2.circle(img, (cx, cy), 5, (255, 0, 255), cv2.FILLED)
        
        # --------------------------------------------- Pedestrian Number Counter --------------------------------------------- #

        # limits_up = [0, 100, 640, 100]
        if limits_up[0]<= cx<= limits_up[2] and limits_up[1] -2 <= cy <= limits_up[1]+2:
            if totalCount.count(Id) ==0 and (currentClass == "person"):
                totalCount.append(Id)
                cv2.line(img, (limits_up[0], limits_up[1]), (limits_up[2], limits_up[3]), (255, 0, 255), 2)
            if up.count(Id) ==0 and currentClass == 'person':
                up.append(Id)

        # limits_down = [0, 300, 640, 300]
        if limits_down[0]<=cx<= limits_down[2] and limits_down[1] -2 <= cy <= limits_down[1]+2:
            if totalCount.count(Id) ==0 and (currentClass == "person"):
                totalCount.append(Id)
                cv2.line(img, (limits_down[0], limits_down[1]), (limits_down[2], limits_down[3]), (255, 0, 255), 2)
            if down.count(Id) ==0 and currentClass == 'person':
                down.append(Id)

        # limits_left = [100, 0, 100, 360]
        if limits_left[1]<=cy<= limits_left[3] and limits_left[0] -2 <= cx <= limits_left[0]+2:
            if totalCount.count(Id) ==0 and (currentClass == "person"):
                totalCount.append(Id)
                cv2.line(img, (limits_left[0], limits_left[1]), (limits_left[2], limits_left[3]), (255, 0, 255), 2)
            if left.count(Id) == 0 and currentClass == 'person':
                left.append(Id)

        # limits_right = [500, 0, 500, 360]
        if limits_right[1]<cy< limits_right[3] and limits_right[0] -2 <= cx <= limits_right[0]+2:
            if totalCount.count(Id) ==0 and (currentClass == "person"):
                totalCount.append(Id)
                cv2.line(img, (limits_right[0], limits_right[1]), (limits_right[2], limits_right[3]), (255, 0, 255), 2)
            if right.count(Id) ==0 and currentClass == 'person':
                right.append(Id)

        # --------------------------------------------- Pedestrian Number Counter --------------------------------------------- #


    # --------------------------------------- Display Multi-directional Count ------------------------------------ #

    # For people going upwards
    cv2.rectangle(img, (180, 280), (200+170, 340), (255, 255, 255), -1)
    cv2.putText(img, f'People Going Down', (200, 300), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
    cv2.putText(img, f'Total Count: {len(totalCount)}', (200, 315), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
    cv2.putText(img, f'People Count: {len(down)}', (200, 330), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)

    # For people going downwards
    cv2.rectangle(img, (180, 5), (200+170, 70), (255, 255, 255), -1)
    cv2.putText(img, f'People Going UP', (200, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
    cv2.putText(img, f'Total Count: {len(totalCount)}', (200, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
    cv2.putText(img, f'People Count: {len(up)}', (200, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)

    # For people going left
    cv2.rectangle(img, (480, 130), (480+170, 200), (255, 255, 255), -1)
    cv2.putText(img, f'People Going Right', (500, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
    cv2.putText(img, f'Total Count: {len(totalCount)}', (500, 170), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
    cv2.putText(img, f'People Count: {len(right)}', (500, 190), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)

    # For people going right
    cv2.rectangle(img, (5, 130), (5+170, 200), (255, 255, 255), -1)
    cv2.putText(img, f'People Going Left', (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
    cv2.putText(img, f'Total Count: {len(totalCount)}', (10, 170), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
    cv2.putText(img, f'People Count: {len(left)}', (10, 190), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)

    # --------------------------------------- Display Multi-directional Count ------------------------------------ #

    out.write(img)
    cv2.imshow("Image", img)
    # cv2.imshow("Image Region", imgRegion)
  # Add this line to break the loop when 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
# ----------------------------------------------------- Infinite Loop -------------------------------------------- #

cap.release()
out.release()
cv2.destroyAllWindows()