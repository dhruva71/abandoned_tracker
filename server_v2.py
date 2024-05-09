#!/usr/bin/env python
import shutil
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, UploadFile, HTTPException, BackgroundTasks, Depends, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse

import database.database
import database.crud
import datatypes
from state import server_state_machine
import models
from state.global_state import GlobalState
from database.database import engine, get_db

PORT = 9001

database.database.Base.metadata.create_all(bind=engine)


@asynccontextmanager
async def lifespan(app: FastAPI):
    print(f"Starting server on port {PORT}")
    try:
        GlobalState.add_observer(server_state_machine.ServerStateMachine())
        yield
    finally:
        print("Shutting down server")

        GlobalState.clear_observers()
        await app.state.shutdown()


app = FastAPI()

origins = [
    '*',
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

state_machine = server_state_machine.ServerStateMachine()

FRAME_COUNT = 0
FRAMES_TO_PROCESS = 0
ABORT_FLAG: bool = False

# Ensure the output directory exists
output_dir = Path("temp_videos")
output_dir.mkdir(parents=True, exist_ok=True)


# app.mount("/output_frames", StaticFiles(directory=output_dir), name="output_frames")


@app.post("/upload-video/")
async def analyze_video(background_tasks: BackgroundTasks, file: UploadFile, task: str = Form('Baggage'),
                        db=Depends(get_db)):
    global state_machine

    print(f"Received video: {file.filename}, task: {task}")

    save_path = state_machine.get_save_path(file.filename)
    print(f"Saving video to {save_path}, task: {task}")
    try:
        if state_machine.is_processing():
            raise HTTPException(status_code=400, detail="A video is already being processed")
        else:
            with open(save_path, "wb") as buffer:
                shutil.copyfileobj(file.file, buffer)
                print(f"Video saved to {save_path}")

            # clear output directory
            # for file in output_dir.iterdir():
            #     if file.is_file():
            #         file.unlink()

            return state_machine.set_state(new_state=datatypes.ProcessingState.PROCESSING,
                                           background_tasks=background_tasks,
                                           db=db,
                                           save_path=save_path,
                                           model_name=models.DetectionModel.get_models()['models'][0],
                                           task=datatypes.TaskEnum[
                                               task] if task in datatypes.TaskEnum.__members__ else datatypes.TaskEnum.Baggage)

    except HTTPException as e:
        raise HTTPException(status_code=500, detail=str(e))
    except Exception as e:
        GlobalState.reset_state()
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        try:
            # file.file.close()
            file.file.close()
        except Exception as e:
            print(e)


@app.get("/frame/{video_id}/{frame_name}")
async def get_frame(video_id: str, frame_name: str):
    DIR = Path(__file__).resolve().parent
    frame_path = DIR / output_dir / video_id / frame_name

    print(f"Getting frame {str(frame_path)}")

    # if frame_path.exists():
    #     return FileResponse(frame_path)
    # else:
    #     raise HTTPException(status_code=404, detail="Frame not found")

    try:
        return FileResponse(frame_path, media_type="image/jpeg")
    except Exception as e:
        print(e)
        raise HTTPException(status_code=404, detail="Frame not found")


# path to get all the abandoned frames as urls via static files
@app.get("/frames/{video_id}")
async def get_abandoned_frames(video_id: str):
    print(f"Getting abandoned frames for video {video_id}")
    # frames_list = [f"/output_frames/{frame.name}" for frame in output_dir.iterdir() if frame.is_file()]
    output_dir_str = f'temp_videos/{video_id}'
    # convert to Path object
    output_dir = Path(output_dir_str)

    # create a list of all the *.jpg files in output_dir
    frames_list = [f"/{output_dir_str.split('/')[-1]}/{frame.name.strip()}" for frame in output_dir.iterdir() if
                   frame.is_file() and frame.suffix == ".jpg"]

    return {
        "frames": frames_list
    }

# endpoint to get access to video
@app.get("/video/{video_id}")
async def get_video(video_id: str):
    print(f"Getting video {video_id}")
    video_path = Path("temp_videos") / video_id / f"{video_id}.avi"
    if video_path.exists():
        return FileResponse(video_path)
    else:
        raise HTTPException(status_code=404, detail="Video not found")

@app.get("/version")
async def get_version():
    return {"version": "0.2.2"}


@app.get("/status")
async def get_processing_status():
    return state_machine.get_status()


@app.post('/abort')
async def set_abort_flag(abort: bool = True):
    # GlobalState.set_state(datatypes.ProcessingState.ABORTED)
    return state_machine.abort()


@app.get("/models")
async def get_models():
    return {"models": models.DetectionModel.get_models()['models']}


@app.post("/set-model")
async def set_model(model: str, db=Depends(get_db)):
    # check if model is valid
    if model not in models.DetectionModel.get_models()['models']:
        raise HTTPException(status_code=400, detail="Invalid model name")

    # if we are processing a video, we cannot change the model
    try:
        state_machine.set_state(new_state=datatypes.ProcessingState.PROCESSING, db=db)
    except ValueError:
        raise HTTPException(status_code=400, detail="Cannot change the model while processing a video")

    return_value = {}
    try:
        return_value = state_machine.set_model(model)
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid model name")
    finally:
        return return_value


# get details of a video from db using its id
@app.get("/video/{video_id}")
async def get_video(video_id: str, db=Depends(get_db)):
    video = database.crud.get_video_by_video_id(db, video_id=video_id)
    if video is None:
        raise HTTPException(status_code=404, detail="Video not found")
    return video

# endpoint to expose all videos in the database
@app.get("/videos")
async def get_videos(skip: int = 0, limit: int = 100, db=Depends(get_db)):
    videos = database.crud.get_videos(db, skip=skip, limit=limit)
    return videos

# To run the server:
# uvicorn script_name:app --reload

if __name__ == '__main__':
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=PORT)
