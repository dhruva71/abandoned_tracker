from collections import defaultdict

import cv2
import numpy as np
import torch
from ultralytics import YOLO, RTDETR, NAS
from ultralytics.utils.plotting import Annotator, colors

import state.server_state_machine
# from ultralytics_experimental.server.file_utils import save_frame
from datatypes import ProcessingState, TaskEnum
from file_utils import save_frame
from state.global_state import GlobalState

# from ultralytics_experimental.server import output_dir

SHOW_DETECTED_OBJECTS = True  # Set to True to display detected objects, else only shows tracking lines
IMAGE_SIZE = 1024,  # [640,864,1024] has to be a multiple of 32, YOLO adjusts to 640x640
MAKE_FRAME_SQUARE = True
CONSOLE_MODE = True  # disables window display
abandoned_frames: list = []


def detect_fights(video_path, model_name) -> list:
    global FRAME_COUNT
    global FRAMES_TO_PROCESS
    global ABORT_FLAG
    global abandoned_frames

    model_name = 'fight_det_model.pt'
    # model_name = 'fight_detect_dhruva_yolov8x.pt'
    model = YOLO(model_name)

    output_dir = GlobalState.get_output_dir()

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
    GlobalState.set_frames_to_process(FRAMES_TO_PROCESS)

    # Define the codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(f'abandoned_{video_path.split("/")[-1].split(".")[0]}_{model_name.split(".")[0]}.avi', fourcc,
                          fps, (frame_width, frame_height))

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
        if GlobalState.get_state() == ProcessingState.ABORTED:
            print("Aborting processing")
            break

        # Read a frame from the video
        success, frame = cap.read()

        if success:
            FRAME_COUNT += 1
            GlobalState.set_frame_count(FRAME_COUNT)
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
                                    # persist=True,
                                    show=False,
                                    classes=[0],  # 0 is the class for fight, 1 is the class for non-fight
                                    # tracker='bytetrack.yaml',
                                    # tracker='botsort.yaml',
                                    vid_stride=5,
                                    visualize=False,
                                    line_width=1,
                                    show_labels=False,
                                    iou=0.1,
                                    conf=0.3,
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
                save_frame(output_dir=output_dir, file_name=f"{GlobalState.get_video_id()}_{FRAME_COUNT}.jpg",
                           frame=annotated_frame)

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

    # unclean separation, should happen via StateMachine
    # set the processing state to completed
    if GlobalState.get_state() != ProcessingState.ABORTED:
        GlobalState.set_state(ProcessingState.COMPLETED)
        state.server_state_machine.ServerStateMachine.set_complete()

    return abandoned_frames
