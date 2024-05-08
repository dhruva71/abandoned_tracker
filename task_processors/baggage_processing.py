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

SHOW_DETECTED_OBJECTS = False  # Set to True to display detected objects, else only shows tracking lines
IMAGE_SIZE = 1024,  # [640,864,1024] has to be a multiple of 32, YOLO adjusts to 640x640
MAKE_FRAME_SQUARE = True
CONSOLE_MODE = True  # disables window display
abandoned_frames: list = []


def track_objects(video_path, model_name) -> list:
    global FRAME_COUNT
    global FRAMES_TO_PROCESS
    global ABORT_FLAG
    global abandoned_frames

    if model_name.startswith('yolov') or model_name.startswith('gelan'):
        model = YOLO(model_name)
    elif model_name.startswith('rtdetr'):
        model = RTDETR(model_name)
    elif model_name.startswith('yolo_nas'):
        model = NAS(model_name)

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

    # Store the track history and static frames count
    track_history = defaultdict(list)
    static_frame_count = defaultdict(int)
    static_threshold = 50  # movement threshold in pixels
    abandonment_frames_threshold = 100  # frames threshold for stationary alert
    use_old_frames_limit = 90  # use old frames to track objects for this number of frames
    using_old_frames = False
    old_frame_counter = 0

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

    def brightness_stabilization(frame):
        # Convert to YUV color space
        yuv = cv2.cvtColor(frame, cv2.COLOR_BGR2YUV)
        # Equalize the histogram of the Y channel (the luminance)
        yuv[:, :, 0] = cv2.equalizeHist(yuv[:, :, 0])
        # Convert back to BGR color space
        frame_eq = cv2.cvtColor(yuv, cv2.COLOR_YUV2BGR)
        return frame_eq

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
            results = model.track(frame,
                                  persist=True,
                                  show=False, classes=[26, 28],
                                  tracker='bytetrack.yaml',
                                  # tracker='botsort.yaml',
                                  vid_stride=5,
                                  visualize=False,
                                  line_width=1,
                                  show_labels=False,
                                  iou=0.1,
                                  conf=0.05,
                                  imgsz=IMAGE_SIZE,
                                  )

            # Get the boxes and track IDs
            boxes = results[0].boxes.xywh.cpu()
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
                if track_id >= 1:
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
                    cv2.putText(annotated_frame, f"Abandonment Alert: ID {track_id}", (10, 30 * track_id),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
                    print(f"Abandonment Alert: ID {track_id}")

                    # if static_frame_count[track_id] - abandonment_frames_threshold % 1 == 0:
                    if static_frame_count[track_id] % 10 == 0:
                        # add the frame to the abandoned frames list
                        abandoned_frames.append(annotated_frame)

                        # save the abandoned frame
                        save_frame(output_dir=output_dir, file_name=f"{GlobalState.get_video_id()}_{FRAME_COUNT}.jpg",
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

    # unclean separation, should happen via StateMachine
    # set the processing state to completed
    if GlobalState.get_state() != ProcessingState.ABORTED:
        GlobalState.set_state(ProcessingState.COMPLETED)
        state.server_state_machine.ServerStateMachine.set_complete()

    return abandoned_frames
