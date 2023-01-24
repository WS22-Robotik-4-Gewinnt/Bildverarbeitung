import requests
from fastapi import FastAPI, Request
from pydantic import BaseModel
# from function import takePicture  # , function
import logging
import json
from function import takePicture

# logging

LOG = "logging_data.log"
logging.basicConfig(filename=LOG, filemode="w", level=logging.DEBUG)

# console handler
console = logging.StreamHandler()
console.setLevel(logging.ERROR)
logging.getLogger("").addHandler(console)

class Difficulty(BaseModel):
  difficulty: int

app = FastAPI()

#########################
# 6 0  0  0  0  0  0  0 #
# 5 0  0  0  0  0  0  0 #
# 4 0  0  0  0  0  0  0 #
# 3 h  0  0  r  0  0  0 #
# 2 h  r  r  h  r  0  r #
# 1 h  h  h  r  h  r  r #
# # 1  2  3  4  5  6  7 #

# h = Human Player | r = Robot Player | 0 = empty space


# }

@app.get("/")
async def home():
  return {"Hello World"}


@app.post("/ready")
async def ready(difficulty: Difficulty):
    if difficulty != 0:
        Difficulty = difficulty
        takePicture()
        logging.info(str(Difficulty))

        addMessage = '{ "Column1": {"Row1":"h", "Row2":"0", "Row3":"0", "Row4":"0", "Row5":"0", "Row6":"0"}, "Column2": {' \
                     '"Row1":"0", "Row2":"0", "Row3":"0", "Row4":"0", "Row5":"0", "Row6":"0"}, "Column3": {"Row1":"0", ' \
                     '"Row2":"0", "Row3":"0", "Row4":"0", "Row5":"0", "Row6":"0"}, "Column4": {"Row1":"0", "Row2":"0", ' \
                     '"Row3":"0", "Row4":"0", "Row5":"0", "Row6":"0"}, "Column5": {"Row1":"0", "Row2":"0", "Row3":"0", ' \
                     '"Row4":"0", "Row5":"0", "Row6":"0"}, "Column6": {"Row1":"h", "Row2":"h", "Row3":"h", "Row4":"0", ' \
                     '"Row5":"0", "Row6":"0"}, "Column7": {"Row1":"r", "Row2":"r", "Row3":"0", "Row4":"0", "Row5":"0", ' \
                     '"Row6":"0"}, "Difficulty": Difficulty.difficulty}'

        addMessage = json.loads(addMessage)
        r = requests.post(f"http://localhost:8093/updateBoard", json=addMessage)
        # logging.info(f"Status Code: {r.status_code}, Response: {r.text}")
        return {"Response": r.text}  # Post http://localhost:8093/updateBoard
    else:
        return {"no difficulty"}

# PORT Bildverarbeitung: 8090
# PORT Spielalgorithmus: 8093
# PORT Hardwaresteuerung: 8096
# docker login -p PASSWORD -u USER/MAIL github.com
