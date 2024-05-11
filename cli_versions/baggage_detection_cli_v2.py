import pathlib
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
import torch
from ultralytics import YOLO, RTDETR, NAS
from ultralytics.utils.plotting import Annotator, colors

from file_utils import save_frame

SHOW_DETECTED_OBJECTS = False  # Set to True to display detected objects, else only shows tracking lines
SHOW_ONLY_ABANDONED_TRACKS = True
IMAGE_SIZE = 1024  # Adjust size, must be a multiple of 32
MAKE_FRAME_SQUARE = True
NORMALIZE_FRAME = False
CONSOLE_MODE = False  # disables window display
abandoned_frames = []


def intersects(bbox1, bbox2):
    # Simple intersection check
    x1, y1, w1, h1 = bbox1
    x2, y2, w2, h2 = bbox2
    return (x1 < x2 + w2 and x1 + w1 > x2 and y1 < y2 + h2 and y1 + h1 > y2)


def track_objects(video_path, model_name='rtdetr-x.pt'):
    global FRAME_COUNT, FRAMES_TO_PROCESS, ABORT_FLAG, abandoned_frames

    output_dir = None
    video_path = Path(video_path)
    # extract the video name without the extension
    video_name = video_path.stem
    # convert the video name to a string
    video_name = str(video_name)

    # reset video_path to string
    video_path = str(video_path)

    # save the abandoned frame
    if output_dir is None:
        output_dir = f"output_frames/{video_name}"
        # create the output directory if it does not exist
        Path(output_dir).mkdir(parents=True, exist_ok=True)

    baggage_class_id = 26  # Update based on your model's specific class ID for baggage
    person_class_id = 0  # Update based on your model's specific class ID for person
    baggage_tracks = defaultdict(list)
    people_tracks = defaultdict(list)
    static_frame_count = defaultdict(int)
    static_threshold = 150  # Movement threshold in pixels
    abandonment_frames_threshold = 125  # Frames threshold for stationary alert
    save_every_x_frames = 30

    model = RTDETR(model_name)  # Update model selection based on name if needed
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
    out = cv2.VideoWriter(f'processed_{Path(video_path).stem}_{model_name.split(".")[0]}.avi', fourcc, 30,
                          (frame_width, frame_height))

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

            results = model.track(frame, persist=True, show=False, classes=[baggage_class_id, person_class_id],
                                  imgsz=IMAGE_SIZE)
            # boxes, track_ids, clss = results.pandas().xyxy[0][['xmin', 'ymin', 'xmax', 'ymax']].values, \
            #     results.pandas().xyxy[0]['track_id'].values, \
            #     results.pandas().xyxy[0]['class'].values

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
                if cls == baggage_class_id:
                    baggage_tracks[track_id].append(box)
                elif cls == person_class_id:
                    people_tracks[track_id].append(box)

            annotated_frame = frame.copy()
            for track_id, bboxes in baggage_tracks.items():
                last_bbox = bboxes[-1]
                is_abandoned = not any(
                    intersects(last_bbox, p_bbox) for p_bboxes in people_tracks.values() for p_bbox in p_bboxes)
                if is_abandoned:
                    static_frame_count[track_id] += 1
                else:
                    static_frame_count[track_id] = 0

                if static_frame_count[track_id] > abandonment_frames_threshold:
                    annotator = Annotator(annotated_frame, line_width=2)
                    annotator.box_label(last_bbox, label=f"Abandoned: ID {track_id}", color=colors('red'),
                                        txt_color=(255, 255, 255))
                    abandoned_frames.append(annotated_frame)
                    print(f"Abandonment Alert: ID {track_id}")

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


if __name__ == '__main__':
    video_path = r'C:\Users\onlin\Downloads\TNex\new_dataset\Left_Object\Left_Object_2_Cam1_1.avi'
    # video_path = r'C:\Users\onlin\Downloads\TNex\new_dataset\Left_Object\Left_Object_1_Cam2_1.avi'
    track_objects(video_path=video_path,
                  model_name='rtdetr-x.pt')
