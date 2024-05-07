#!/usr/bin/env python
import shutil
import string
from contextlib import asynccontextmanager
from pathlib import Path
from random import random

from fastapi import FastAPI, UploadFile, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

import database.database
import server_state_machine
import models

from database.database import SessionLocal, engine

PORT = 9001

database.database.Base.metadata.create_all(bind=engine)


# Dependency
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


@asynccontextmanager
async def lifespan(app: FastAPI):
    print(f"Starting server on port {PORT}")
    try:
        yield
    finally:
        print("Shutting down server")
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

state_machine = server_state_machine.ServerStateMachine

FRAME_COUNT = 0
FRAMES_TO_PROCESS = 0
ABORT_FLAG: bool = False

# Ensure the output directory exists
output_dir = Path("output_frames")
output_dir.mkdir(parents=True, exist_ok=True)
app.mount("/output_frames", StaticFiles(directory=output_dir), name="output_frames")


@app.post("/upload-video/")
async def analyze_video(background_tasks: BackgroundTasks, file: UploadFile):
    global state_machine

    # create 6 character random folder name
    random_folder_name = ''.join(random.choices(string.ascii_lowercase + string.digits, k=6))

    # create folder
    save_path = Path(f"temp_videos/{random_folder_name}")

    # create the folder if it doesn't exist
    save_path.mkdir(parents=True, exist_ok=True)

    # save the video to the folder, with the same name as the folder
    save_path = save_path / f'{random_folder_name}.{file.filename.split(".")[-1]}'

    try:
        if state_machine.is_processing():
            raise HTTPException(status_code=400, detail="A video is already being processed")
        else:
            with open(save_path, "wb") as buffer:
                shutil.copyfileobj(file.file, buffer)
                print(f"Video saved to {save_path}")

            # clear output directory
            for file in output_dir.iterdir():
                if file.is_file():
                    file.unlink()

            return state_machine.set_state(server_state_machine.ProcessingState.PROCESSING,
                                           background_tasks=background_tasks,
                                           save_path=save_path,
                                           model_name=models.DetectionModel.get_models()['models'][0])

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        file.file.close()


@app.get("/frame/{frame_name}")
async def get_frame(frame_name: str):
    frame_path = output_dir / frame_name
    if frame_path.is_file():
        return FileResponse(str(frame_path))
    else:
        raise HTTPException(status_code=404, detail="Frame not found")


# path to get all the abandoned frames as urls via static files
@app.get("/frames")
async def get_abandoned_frames():
    frames_list = [f"/output_frames/{frame.name}" for frame in output_dir.iterdir() if frame.is_file()]
    return {
        "frames": frames_list
    }


@app.get("/status")
async def get_processing_status():
    return state_machine.get_status()


@app.post('/abort')
async def set_abort_flag(abort: bool = True):
    return state_machine.abort()


@app.get("/models")
async def get_models():
    return {"models": models.DetectionModel.get_models()['models']}


@app.post("/set-model")
async def set_model(model: str):
    # check if model is valid
    if model not in models.DetectionModel.get_models()['models']:
        raise HTTPException(status_code=400, detail="Invalid model name")

    # if we are processing a video, we cannot change the model
    try:
        state_machine.set_state(server_state_machine.ProcessingState.PROCESSING, )
    except ValueError:
        raise HTTPException(status_code=400, detail="Cannot change the model while processing a video")

    return_value = {}
    try:
        return_value = state_machine.set_model(model)
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid model name")
    finally:
        return return_value


# To run the server:
# uvicorn script_name:app --reload

if __name__ == '__main__':
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=PORT)
