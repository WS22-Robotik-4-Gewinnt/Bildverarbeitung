import requests
from fastapi import FastAPI
from pydantic import BaseModel
from function import takePicture  # , function

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

# Feld={ "Column1": {"Row1":"h", "Row2":"0", "Row3":"0", "Row4":"0", "Row5":"0", "Row6":"0"},
# "Column2": {"Row1":"0", "Row2":"0", "Row3":"0", "Row4":"0", "Row5":"0", "Row6":"0"},
# "Column3": {"Row1":"0", "Row2":"0", "Row3":"0", "Row4":"0", "Row5":"0", "Row6":"0"},
# "Column4": {"Row1":"0", "Row2":"0", "Row3":"0", "Row4":"0", "Row5":"0", "Row6":"0"},
# "Column5": {"Row1":"0", "Row2":"0", "Row3":"0", "Row4":"0", "Row5":"0", "Row6":"0"},
# "Column6": {"Row1":"h", "Row2":"h", "Row3":"h", "Row4":"0", "Row5":"0", "Row6":"0"},
# "Column7": {"Row1":"r", "Row2":"r", "Row3":"0", "Row4":"0", "Row5":"0", "Row6":"0"}
# }

ROWS = 6
COLUMNS = 7


class Column(BaseModel):
    Row1: str
    Row2: str
    Row3: str
    Row4: str
    Row5: str
    Row6: str


class Board(BaseModel):
    Column1: Column
    Column2: Column
    Column3: Column
    Column4: Column
    Column5: Column
    Column6: Column
    Column7: Column
    Difficulty: int


@app.get("/")
async def home():
    return {"Hello World"}


@app.post("/ready")
async def ready(difficulty: Board.Difficulty):
    if difficulty != 0:
        takePicture

        addmessage = {"Colum1": Board.Column1, "Difficulty": Board.Difficulty}
        r = requests.post('http://localhost:8093/updateBoard', "ok", addmessage)
        return {"message": "Hello world" + addmessage}  # Post http://localhost:8093/updateBoard
    else:
        return {"no difficulty"}

# PORT Bildverarbeitung: 8090
# PORT Spielalgorithmus: 8093
# PORT Hardwaresteuerung: 8096
