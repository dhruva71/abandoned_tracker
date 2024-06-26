from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
import torch
from ultralytics import YOLO, RTDETR, NAS
from ultralytics.utils.plotting import Annotator, colors

from file_utils import save_frame

# Detection based on the lack of movement in baggage.


SHOW_DETECTED_OBJECTS = False  # Set to True to display detected objects, else only shows tracking lines
SHOW_ONLY_ABANDONED_TRACKS = True
IMAGE_SIZE = 1024,  # [640,864,1024] has to be a multiple of 32, YOLO adjusts to 640x640
MAKE_FRAME_SQUARE = True
NORMALIZE_FRAME = False
CONSOLE_MODE = False  # disables window display
abandoned_frames: list = []


def track_objects(video_path, model_name='rtdetr-x.pt') -> list:
    global FRAME_COUNT
    global FRAMES_TO_PROCESS
    global ABORT_FLAG
    global abandoned_frames

    # Store the track history and static frames count
    track_history = defaultdict(list)
    static_frame_count = defaultdict(int)
    static_threshold = 150  # movement threshold in pixels
    abandonment_frames_threshold = 125  # frames threshold for stationary alert
    save_every_x_frames = 30
    use_old_frames_limit = 90  # use old frames to track objects for this number of frames
    using_old_frames = False
    old_frame_counter = 0

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

    # Open the video file
    # video_path = "../videos/new_videos/LeftObject_4.avi"
    print(f"Processing video: {video_path}")
    cap = cv2.VideoCapture(video_path)

    # Check if the video opened successfully
    if not cap.isOpened():
        print("Error: Could not open video.")
        # exit()
        raise Exception("Error: Could not open video.")

    # Get video properties for the output file
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    FRAMES_TO_PROCESS = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Define the codec and create VideoWriter object to write as mp4
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(f'processed_{video_path.split("/")[-1].split(".")[0]}_{model_name.split(".")[0]}.avi', fourcc,
                          30, (frame_width, frame_height))

    old_tracks = []
    old_classes = []
    old_boxes = []
    old_names = []

    start_frame = 0
    successfully_set = cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    if not successfully_set:
        print(f"Could not set the frame to {start_frame}")
        exit()
    else:
        print(f"Set the frame to {start_frame}")

    FRAME_COUNT = start_frame
    # Loop through the video frames
    while cap.isOpened():

        # Read a frame from the video
        success, frame = cap.read()

        if success:
            FRAME_COUNT += 1
            print(f"Processing frame {FRAME_COUNT}")

            FRAMES_TO_PROCESS -= 1

            # perform frame normalization
            if NORMALIZE_FRAME:
                frame = cv2.normalize(frame, None, 0, 255, cv2.NORM_MINMAX)

            if MAKE_FRAME_SQUARE:
                # make the frame square
                square_size = min(frame.shape[0], frame.shape[1])
                # square_image = np.zeros((square_size, square_size, 3), np.uint8)
                square_image = cv2.resize(frame, (square_size, square_size))
                frame = square_image
                print(f"Resized frame to {square_size}x{square_size}")

            # Run tracking on the frame
            results = model.track(frame,
                                  persist=True,
                                  show=False, classes=[26, 28],
                                  tracker='bytetrack.yaml',
                                  # tracker='botsort.yaml',
                                  vid_stride=1,
                                  visualize=False,
                                  line_width=1,
                                  show_labels=False,
                                  iou=0.1,
                                  conf=0.05,
                                  imgsz=IMAGE_SIZE,
                                  )

            try:
                track_ids = results[0].boxes.id.int().cpu().tolist()
                clss = results[0].boxes.cls.cpu().tolist()
                # Extract bounding boxes, tracking IDs, classes names
                boxes = results[0].boxes.xywh.cpu()
                names = results[0].names

                old_tracks = track_ids
                old_classes = clss
                old_boxes = boxes
                old_names = names

                old_frame_counter = 0
            except AttributeError:
                if using_old_frames:
                    track_ids = old_tracks
                    clss = old_classes
                    boxes = old_boxes
                    names = old_names

                    old_frame_counter += 1
                    if old_frame_counter > use_old_frames_limit:
                        print("Exceeded the limit of using old frames to track objects. Using the current frames.")
                        using_old_frames = False
                else:
                    track_ids = []
                    clss = []
                    boxes = []
                    names = []

            # Visualize the results on the frame
            if SHOW_DETECTED_OBJECTS:
                annotated_frame = results[0].plot()
            else:
                annotated_frame = frame.copy()

            # display frame number
            cv2.putText(annotated_frame, f"Frame: {FRAME_COUNT}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)

            # Iterate through the results
            for box, track_id, cls in zip(boxes, track_ids, clss):

                # If the tracking ID is 1
                if track_id >= 1 and not SHOW_ONLY_ABANDONED_TRACKS:
                    x1, y1, x2, y2 = box
                    label = str(track_id) + " " + names[int(cls)]
                    xywh = [(x1 - x2 / 2), (y1 - y2 / 2), (x1 + x2 / 2), (y1 + y2 / 2)]
                    annotator = Annotator(annotated_frame, line_width=2, example=names)
                    annotator.box_label(xywh, label=label, color=colors(int(cls), True), txt_color=(255, 255, 255))

            # Plot the tracks and check for abandonment
            for box, track_id in zip(boxes, track_ids):
                x, y, w, h = box
                track = track_history[track_id]
                track.append((float(x), float(y)))  # x, y center point
                if len(track) > 30:
                    track.pop(0)

                # Check if object is stationary
                if len(track) > 1 and np.linalg.norm(np.array(track[-1]) - np.array(track[-2])) < static_threshold:
                    static_frame_count[track_id] += 1
                else:
                    static_frame_count[track_id] = 0

                # Trigger abandonment alert
                if static_frame_count[track_id] > abandonment_frames_threshold:
                    # draw bounding box
                    x1, y1, x2, y2 = box
                    label = str(track_id) + " " + names[int(cls)]
                    xywh = [(x1 - x2 / 2), (y1 - y2 / 2), (x1 + x2 / 2), (y1 + y2 / 2)]
                    annotator = Annotator(annotated_frame, line_width=2, example=names)
                    annotator.box_label(xywh, label=label, color=colors(int(cls), True), txt_color=(255, 255, 255))

                    cv2.putText(annotated_frame, f"Abandonment Alert: ID {track_id}", (10, 30 * track_id),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
                    print(f"Abandonment Alert: ID {track_id}")

                    # if static_frame_count[track_id] - abandonment_frames_threshold % 1 == 0:
                    if static_frame_count[track_id] % save_every_x_frames == 0:
                        # add the frame to the abandoned frames list
                        abandoned_frames.append(annotated_frame)

                        output_dir = None
                        video_name = video_path.split("/")[-1].split(".")[0]
                        # save the abandoned frame
                        if output_dir is None:
                            output_dir = f"output_frames/{video_name}"
                            # create the output directory if it does not exist
                            Path(output_dir).mkdir(parents=True, exist_ok=True)

                        save_frame(output_dir=output_dir, file_name=f"{video_name}_{FRAME_COUNT}.jpg",
                                   frame=annotated_frame)

                # Draw the tracking lines
                points = np.array(track).astype(np.int32).reshape((-1, 1, 2))
                cv2.polylines(annotated_frame, [points], isClosed=False, color=(230, 230, 230), thickness=2)

            # Write the frame to the output file
            out.write(annotated_frame)

            # Display the annotated frame
            if not CONSOLE_MODE:
                cv2.imshow("Tracking", annotated_frame)

                # Break the loop if 'q' is pressed
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

                # save image if 's' is pressed
                if cv2.waitKey(1) & 0xFF == ord("s"):
                    cv2.imwrite(
                        f'abandoned_{video_path.split("/")[-1].split(".")[0]}_{model_name.split(".")[0]}_frame_{FRAME_COUNT}.png',
                        annotated_frame)
        else:
            # End of video
            break

    # Release everything if job is finished
    cap.release()
    out.release()

    # reset the abort flag
    ABORT_FLAG = False

    if not CONSOLE_MODE:
        cv2.destroyAllWindows()

    return abandoned_frames


if __name__ == '__main__':
    track_objects(video_path=r'C:\Users\onlin\Downloads\TNex\new_dataset\Left_Object\Left_Object_2_Cam1_1.avi', model_name='rtdetr-x.pt')
