## File storage
* Random folder with 6 character name is created
* video file is saved with the same name as the folder, with the original extension
* Frames are extracted from the video and saved in the folder

## State Machine
* The system runs a state machine with the following states:
  * EMPTY: Initial state
  * PROCESSING: when a video/task is being processed.
  * COMPLETED: when a video is completely processed.
  * ABORTED: when a video processing is aborted via the API.
