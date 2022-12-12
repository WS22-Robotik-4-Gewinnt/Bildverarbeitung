# Bildverarbeitung

## Datenstruktur
### Spielfeld 7x6
```
6 0  0  0  0  0  0  0
5 0  0  0  0  0  0  0
4 0  0  0  0  0  0  0
3 h  0  0  r  0  0  0
2 h  r  r  h  r  0  r
1 h  h  h  r  h  r  r
--1--2--3--4--5--6--7

// h = Human Player | r = Robot Player | 0 = empty space
```

## Schnittstellen / Kommunikation
###  Bildverarbeitung -> Spielalgorithmus
```json
{ "Column1": {"Row1":"h", "Row2":"0", "Row3":"0", "Row4":"0", "Row5":"0", "Row6":"0"},
"Column2": {"Row1":"0", "Row2":"0", "Row3":"0", "Row4":"0", "Row5":"0", "Row6":"0"},
"Column3": {"Row1":"0", "Row2":"0", "Row3":"0", "Row4":"0", "Row5":"0", "Row6":"0"},
"Column4": {"Row1":"0", "Row2":"0", "Row3":"0", "Row4":"0", "Row5":"0", "Row6":"0"},
"Column5": {"Row1":"0", "Row2":"0", "Row3":"0", "Row4":"0", "Row5":"0", "Row6":"0"},
"Column6": {"Row1":"h", "Row2":"h", "Row3":"h", "Row4":"0", "Row5":"0", "Row6":"0"},
"Column7": {"Row1":"r", "Row2":"r", "Row3":"0", "Row4":"0", "Row5":"0", "Row6":"0"}
}
```

### Übertragung Spielalgorithmus -> Hardwaresteuerung
```json
{
 "col": 7,
 "row": 1
}
```

# Goal of this project is to find a grid (chessboard) in the given image.

http://docs.opencv.org/doc/tutorials/calib3d/camera_calibration_square_chess/camera_calibration_square_chess.html

```
$ git clone https://github.com/Itseez/opencv.git
$ cd opencv
$ git checkout 3.0.0-rc1
$ cd cmake
$ apt-get install cmake libgstreamer-plugins-base1.0-dev libgstreamer-plugins-base0.10-dev
$ cmake -D CMAKE_BUILD_TYPE=RELEASE -D CMAKE_INSTALL_PREFIX=/usr/local -D BUILD_opencv_java=OFF -D PYTHON_EXECUTABLE=/usr/bin/python3.4 ..
$ cd ..
$ make -j 4
$ PYTHONPATH=opencv/lib/python3/ ./grid_finder.py
```

We want to warp an image to the "airplane" 2d view.

We do this in two steps:

- Find 4 points around something that should be a rectangle
- Use warpPerspective to warp it

To find those 4 points, several methods:

- Detect lines, detect columns, apply and AND:
    - Work only if lines are already horizontal and columns already vertical,
      that's not our case.
- findChessboardCorners
    - Not bad but need to know the chessboard size, that's not our case.
- Detect lines with houghlines2
    - That's we're doing for now.
- What else ?

