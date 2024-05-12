import pathlib
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
import torch
from ultralytics import YOLO, RTDETR, NAS
from ultralytics.utils.plotting import Annotator, colors

from file_utils import save_frame

# Detection based on static + lack of overlap with people.


SHOW_DETECTED_OBJECTS = False  # Set to True to display detected objects, else only shows tracking lines
SHOW_ONLY_ABANDONED_TRACKS = True
IMAGE_SIZE = 640  # Adjust size, must be a multiple of 32
MAKE_FRAME_SQUARE = False
NORMALIZE_FRAME = False
CONSOLE_MODE = False  # disables window display
abandoned_frames = []
DEBUG: bool = True


def intersects(bbox1, bbox2) -> bool:
    # bbox1, bbox2 are in the form (x, y, w, h)
    # convert to x1, y1, x2, y2
    x11, y11, x12, y12 = create_xyxy_from_xywh(*bbox1)
    x21, y21, x22, y22 = create_xyxy_from_xywh(*bbox2)

    # Simple intersection check
    x1, y1, w1, h1 = create_xyxy_from_xywh(*bbox1)
    x2, y2, w2, h2 = create_xyxy_from_xywh(*bbox2)
    # return x1 < x2 + w2 and x1 + w1 > x2 and y1 < y2 + h2 and y1 + h1 > y2

    # adapt: (x1min < x2max AND x2min < x1max AND y1min < y2max AND y2min < y1max)
    # from: https://stackoverflow.com/questions/20925818/algorithm-to-check-if-two-boxes-overlap
    # intersection: bool = (x1 < x2 + w2) and (x2 < x1 + w1) and (y1 < y2 + h2) and (y2 < y1 + h1)

    intersection: bool = (x11 < x22) and (x21 < x12) and (y11 < y22) and (y21 < y12)

    return intersection


def track_objects(video_path, model_name='rtdetr-x.pt', start_frame: int = 0):
    global FRAME_COUNT, FRAMES_TO_PROCESS, ABORT_FLAG, abandoned_frames

    video_path = Path(video_path)
    # extract the video name without the extension
    video_name = video_path.stem
    # convert the video name to a string
    video_name = str(video_name)

    # reset video_path to string
    video_path = str(video_path)

    # create directory
    output_dir = None
    if output_dir is None:
        output_dir = f"output_frames/{video_name}"
        # create the output directory if it does not exist
        Path(output_dir).mkdir(parents=True, exist_ok=True)

    baggage_class_id = [26, 28]  # Update based on your model's specific class ID for baggage
    person_class_id = 0  # Update based on your model's specific class ID for person
    baggage_tracks = defaultdict(list)
    people_tracks = defaultdict(list)
    static_frame_count = defaultdict(int)
    static_threshold = 150  # Movement threshold in pixels
    abandonment_frames_threshold = 1  # Frames threshold for stationary alert
    save_every_x_frames = 5  # Save every x frames

    if model_name.startswith('yolov') or model_name.startswith('gelan'):
        model = YOLO(model_name)
    elif model_name.startswith('rtdetr'):
        model = RTDETR(model_name)
    elif model_name.startswith('yolo_nas'):
        model = NAS(model_name)
    else:
        model = RTDETR(model_name)
    torch.cuda.set_device(0)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device=device)
    print(f'Running on device: {model.device}')

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise Exception("Error: Could not open video.")
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    FRAMES_TO_PROCESS = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(f'{output_dir}/processed_{Path(video_path).stem}_{model_name.split(".")[0]}.avi', fourcc, fps,
                          (frame_width, frame_height))

    if start_frame > 0:
        successfully_set = cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        if not successfully_set:
            print(f"Could not set the frame to {start_frame}")
            exit()
        else:
            print(f"Set the frame to {start_frame}")

    FRAME_COUNT = 0
    while cap.isOpened():
        success, frame = cap.read()
        if success:
            FRAME_COUNT += 1
            if NORMALIZE_FRAME:
                frame = cv2.normalize(frame, None, 0, 255, cv2.NORM_MINMAX)
            if MAKE_FRAME_SQUARE:
                square_size = min(frame.shape[0], frame.shape[1])
                frame = cv2.resize(frame, (square_size, square_size))

            results = model.track(frame, persist=True, show=False,
                                  classes=[baggage_class_id[0], baggage_class_id[1], person_class_id],
                                  imgsz=IMAGE_SIZE)

            try:
                track_ids = results[0].boxes.id.int().cpu().tolist()
                clss = results[0].boxes.cls.cpu().tolist()
                # Extract bounding boxes, tracking IDs, classes names
                boxes = results[0].boxes.xywh.cpu()
                names = results[0].names
            except AttributeError:
                track_ids = []
                clss = []
                boxes = []
                names = []

            for box, track_id, cls in zip(boxes, track_ids, clss):
                if cls == baggage_class_id[0] or cls == baggage_class_id[1]:
                    baggage_tracks[track_id].append(box)
                elif cls == person_class_id:
                    people_tracks[track_id].append(box)

            annotated_frame = frame.copy()

            # display detected objects
            if SHOW_DETECTED_OBJECTS:
                for box, track_id, cls in zip(boxes, track_ids, clss):
                    x, y, w, h = box
                    label = str(track_id) + " " + names[int(cls)]
                    xxyy = create_xyxy_from_xywh(x, y, w, h)
                    annotator = Annotator(annotated_frame, line_width=2)
                    annotator.box_label(xxyy, label=f"{names[cls]}: ID {track_id}", color=colors(cls),
                                        txt_color=(255, 255, 255))

            # display frame number
            cv2.putText(annotated_frame, f"Frame: {start_frame + FRAME_COUNT}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)

            for track_id, bboxes in baggage_tracks.items():
                last_bbox = bboxes[-1]

                # here we check the intersection of the last bbox with all the people bboxes
                # is_abandoned = any(
                #     intersects(last_bbox, p_bbox) for p_bboxes in people_tracks.values() for p_bbox in p_bboxes)
                is_abandoned = not any(
                    intersects(last_bbox, p_bboxes[-1]) for p_bboxes in people_tracks.values() if p_bboxes)
                if is_abandoned:
                    print(f"Abandoned: ID {track_id}")
                    # print(f"Abandoned baggage")
                    static_frame_count[track_id] += 1
                else:
                    print(f"Luggage item with ID {track_id} is not abandoned")
                    static_frame_count[track_id] = 0

                if static_frame_count[track_id] > abandonment_frames_threshold:
                    annotator = Annotator(annotated_frame, line_width=2)
                    bounding_box = create_xyxy_from_xywh(*last_bbox)
                    # annotator.box_label(bounding_box, label=f"Abandoned: ID {track_id}", color=colors(0),
                    annotator.box_label(bounding_box, label=f"Abandoned baggage", color=colors(0),
                                        txt_color=(255, 255, 255))
                    abandoned_frames.append(annotated_frame)
                    # print(f"Abandonment Alert: ID {track_id}")
                    # print(f"Abandonment baggage")

                    if static_frame_count[track_id] % save_every_x_frames == 0:
                        save_frame(output_dir=output_dir, file_name=f"{video_name}_{FRAME_COUNT}.jpg",
                                   frame=annotated_frame)

            out.write(annotated_frame)
            if not CONSOLE_MODE:
                cv2.imshow("Tracking", annotated_frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
        else:
            break

    cap.release()
    out.release()
    if not CONSOLE_MODE:
        cv2.destroyAllWindows()
    return abandoned_frames


def create_bbox_from_xxyy(x1, x2, y1, y2):
    xywh = [(x1 - x2 / 2), (y1 - y2 / 2), (x1 + x2 / 2), (y1 + y2 / 2)]
    return xywh


def create_xyxy_from_xywh(x, y, w, h):
    x1, y1 = x - w / 2, y - h / 2
    x2, y2 = x + w / 2, y + h / 2
    return x1, y1, x2, y2

if __name__ == '__main__':
    # video_path = r'C:\Users\onlin\Downloads\TNex\new_dataset\Left_Object\Left_Object_2_Cam1_1.avi' # 2500
    # video_path = r'C:\Users\onlin\Downloads\TNex\new_dataset\Left_Object\Old\Left_Object_2.avi' # 5300
    video_path = r'C:\Users\onlin\Downloads\TNex\new_dataset\Left_Object\Left_Object_1_Cam2_1.avi' # 3000
    track_objects(video_path=video_path,
                  model_name='rtdetr-x.pt',
                  start_frame=3000)
