
from ultralytics import YOLO
from ultralytics.utils.plotting import Annotator

def load_model(name):
    return YOLO(name)

def YOLOv8(or_frame, yolo_frame, model, _main):
    results = model.predict(yolo_frame, verbose=False)
    vehicles = []
    for r in results:
        annotator = Annotator(yolo_frame)
        boxes = r.boxes

        for box in boxes:
            c = box.cls
            vehicle_type = model.names[int(c)]
            # Ignore other objects
            if vehicle_type not in ["car", "truck", "motorbike", "bus"]:
                continue

            b = box.xyxy[0]  # get box coordinates in (left, top, right, bottom) format
            cordinates = b.tolist()


            px = int((cordinates[0]+cordinates[2])/2)
            py = int((cordinates[1]+cordinates[3])/2)
            annotator.box_label(b, model.names[int(c)])
            #cv2.circle(yolo_frame, (px,py), 5, (0, 0, 255), -1)
            yolo_frame = annotator.result()

            or_px,or_py = fix_res(_main.frame_real_dims, _main.frame_resized_dims, (px,py))
            or_pos_1 = fix_res(_main.frame_real_dims, _main.frame_resized_dims, (cordinates[0],cordinates[1]))
            or_pos_2 = fix_res(_main.frame_real_dims, _main.frame_resized_dims, (cordinates[2],cordinates[3]))

            # Take a image of the vehicle - high res
            img_vehicle = or_frame[ or_pos_1[1]:or_pos_2[1], 
                                    or_pos_1[0]:or_pos_2[0]]

            vehicles.append([vehicle_type, [or_px,or_py], img_vehicle])

    #cv2.imshow('YOLO V8 Detection', img)
    return vehicles

def fix_res(or_dims, new_dims, pos):
    x = int((pos[0]/new_dims[0])*or_dims[0])
    y = int((pos[1]/new_dims[1])*or_dims[1])
    return (x,y)