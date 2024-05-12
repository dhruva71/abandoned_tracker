import pathlib
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
import torch
from ultralytics import YOLO, RTDETR, NAS
from ultralytics.utils.plotting import Annotator, colors

from file_utils import save_frame

# Modified version of baggage_detection_cli_v2.py
# Detection based on static + lack of overlap with people.
# Associates abandoned baggage with people in the scene.

SHOW_DETECTED_OBJECTS = True  # Set to True to display detected objects, else only shows tracking lines
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


def detect_fights(video_path, model_name='../fight_det_model.pt', start_frame: int = 0) -> list:
    global FRAME_COUNT
    global FRAMES_TO_PROCESS
    global ABORT_FLAG
    global abandoned_frames
    first_frame:bool = True

    # model_name = 'fight_det_model.pt'
    # model_name = 'fight_detect_dhruva_yolov8x.pt'
    model = YOLO(model_name)

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

    # Define the codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = None

    if start_frame > 0:
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

            # add blur to the frame
            # frame = cv2.GaussianBlur(frame, (5, 5), 0)

            # perform frame normalization
            frame = cv2.normalize(frame, None, 0, 255, cv2.NORM_MINMAX)

            if MAKE_FRAME_SQUARE:
                # make the frame square
                square_size = min(frame.shape[0], frame.shape[1])
                # square_image = np.zeros((square_size, square_size, 3), np.uint8)
                square_image = cv2.resize(frame, (square_size, square_size))
                frame = square_image
                print(f"Resized frame to {square_size}x{square_size}")

            # Run tracking on the frame
            results = model.predict(frame,
                                    show=False,
                                    classes=[0],  # 0 is the class for fight, 1 is the class for non-fight
                                    vid_stride=1,
                                    visualize=False,
                                    line_width=1,
                                    show_labels=False,
                                    iou=0.7,
                                    conf=0.1,
                                    imgsz=IMAGE_SIZE,
                                    )

            # Get the boxes and track IDs
            boxes = results[0].boxes.xywh.cpu()

            # Visualize the results on the frame
            if SHOW_DETECTED_OBJECTS:
                annotated_frame = results[0].plot()
            else:
                annotated_frame = frame.copy()

            # display frame number
            cv2.putText(annotated_frame, f"Frame: {FRAME_COUNT}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)

            # if we have detected objects, save the frame
            if len(boxes) > 0:
                # save the frame
                # add the frame to the abandoned frames list
                abandoned_frames.append(annotated_frame)

                # save the abandoned frame
                save_frame(output_dir=output_dir, file_name=f"{video_name}_{FRAME_COUNT}.jpg",
                           frame=annotated_frame)

            if first_frame:
                first_frame = False

                # get frame dimensions
                frame_width = annotated_frame.shape[1]
                frame_height = annotated_frame.shape[0]

                output_file_name = f'{output_dir}/processed_{video_name}_{model_name.split(".")[0]}.avi'
                print(f'Initializing output writer with {frame_width}x{frame_height}')
                print(f'Output file: {output_file_name}')

                out = cv2.VideoWriter(
                    output_file_name,
                    fourcc,
                    fps, (frame_width, frame_height))

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


def create_bbox_from_xxyy(x1, x2, y1, y2):
    xywh = [(x1 - x2 / 2), (y1 - y2 / 2), (x1 + x2 / 2), (y1 + y2 / 2)]
    return xywh


def create_xyxy_from_xywh(x, y, w, h):
    # convert to x1, x2, y1, y2
    x1, y1 = x - w / 2, y - h / 2
    x2, y2 = x + w / 2, y + h / 2
    return x1, y1, x2, y2


if __name__ == '__main__':
    models = ['fight_det_v4_dhruva_yolov8x.pt', 'fight_det_model.pt', 'fight_detect_dhruva_yolov8x.pt']
    video_path = r'C:\Users\onlin\Downloads\TNex\new_dataset\Physical_Encounter\Fight_1_Cam1_1.avi'  # 2500

    for i in range(1,8):
        file_path=rf'C:\Users\onlin\Downloads\YT_fights\fight{i}.webm'
        detect_fights(video_path=file_path,
                      # model_name='../fight_det_model.pt',
                      model_name=f'../{models[0]}',
                      start_frame=0)
