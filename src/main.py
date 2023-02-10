import requests
from fastapi import FastAPI, Response
from pydantic import BaseModel
import logging
import json
import pathlib
import cv2
from src import grid

# logging
LOG = "logging_data.log"
logging.basicConfig(filename=LOG, filemode="w", level=logging.DEBUG)

# console handler
console = logging.StreamHandler()
console.setLevel(logging.ERROR)
logging.getLogger("").addHandler(console)

# configuration
saturation = '1.1'
grid_buffer = '0'
human_color = 'green'
robot_color = 'red'
image_resize = '500'
camera_id = 1


class Difficulty(BaseModel):
    difficulty: int


app = FastAPI()


@app.get("/")
async def home():
    return {"Hello World"}


@app.post("/ready")
async def ready(difficulty: Difficulty):
    # returns the found and analyzed grid as json
    try:
        grid_json = json.loads(analyze_grid())
        grid_json.update({'Difficulty': difficulty.difficulty})
    except:
        grid_json = json.dumps({'error': 'Error analyzing grid'})
    #message = json.dumps(grid_json, indent=4)
    #messsage = json.loads(grid_json)
    message = grid_json
    print(str(message))
    r = requests.post("http://172.17.0.1:8093/updateBoard", json=message)
    return Response(r.text)


@app.post("/readyDebug")
async def ready_debug(difficulty: Difficulty):
    # returns the found and analyzed grid as json
    try:
        grid_json = json.loads(analyze_grid(False))
        grid_json.update({'Difficulty': difficulty.difficulty})
    except:
        grid_json = json.dumps({'error': 'Error analyzing grid'})

    message = json.dumps(grid_json, indent=4)
    return Response(message)


@app.post("/readyDebugStatic")
async def ready_debug_static(difficulty: Difficulty):
    add_message = {
        "Column1": {"Row1": "h", "Row2": "0", "Row3": "0", "Row4": "0", "Row5": "0", "Row6": "0"},
        "Column2": {"Row1": "0", "Row2": "0", "Row3": "0", "Row4": "0", "Row5": "0", "Row6": "0"},
        "Column3": {"Row1": "0", "Row2": "0", "Row3": "0", "Row4": "0", "Row5": "0", "Row6": "0"},
        "Column4": {"Row1": "0", "Row2": "0", "Row3": "0", "Row4": "0", "Row5": "0", "Row6": "0"},
        "Column5": {"Row1": "0", "Row2": "0", "Row3": "0", "Row4": "0", "Row5": "0", "Row6": "0"},
        "Column6": {"Row1": "h", "Row2": "h", "Row3": "h", "Row4": "0", "Row5": "0", "Row6": "0"},
        "Column7": {"Row1": "r", "Row2": "r", "Row3": "0", "Row4": "0", "Row5": "0", "Row6": "0"},
        "Difficulty": difficulty.difficulty}

    message = json.dumps(add_message, indent=4)
    return Response(message)


def analyze_grid(take_image: bool = True):
    img_path = str(pathlib.Path(__file__).resolve().parent) + '/assets/imagedemo.jpg'

    global saturation, saturation, grid_buffer, human_color, robot_color, image_resize, camera_id

    # Take image from camera if needed
    if take_image:
        cam = None
        try:
            img_path = '/tmp/camImage.jpg'
            cam = cv2.VideoCapture(camera_id)
            ret, image = cam.read()

            cam_flip = cv2.flip(image, 1)
            zoomed = zoom_at(cam_flip, 1.2)
            cv2.imwrite(img_path, zoomed)

        finally:
            if cam is not None:
                cam.release()

    return grid.json([
        img_path,
        '--hc=' + human_color,
        '--rc=' + robot_color,
        '--bb=' + grid_buffer,
        '--resize=' + image_resize,
        '--saturation=' + saturation,
    ])


def zoom_at(img, zoom=1, angle=0, coord=None):
    cy, cx = [i / 2 for i in img.shape[:-1]] if coord is None else coord[::-1]

    rot_mat = cv2.getRotationMatrix2D((cx, cy), angle, zoom)
    result = cv2.warpAffine(img, rot_mat, img.shape[1::-1], flags=cv2.INTER_LINEAR)

    return result

# PORT Bildverarbeitung: 8090
# PORT Spielalgorithmus: 8093
# PORT Hardwaresteuerung: 8096
# docker login -p PASSWORD -u USER/MAIL github.com
