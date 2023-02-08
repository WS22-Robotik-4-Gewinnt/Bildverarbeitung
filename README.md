
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
Die Bildverarbeitung wird durch einen Aufruf von der [Hardwaresteuerung](https://github.com/WS22-Robotik-4-Gewinnt/Hardwaresteuerung) ausgelöst. Nach erfolgreicher Aufnahme und Verarbeitung eines Bildes des aktuellen Spielgeschehens durch die angebrachte Kamera, wird der [Spielalgorithmus](https://github.com/WS22-Robotik-4-Gewinnt/Spielalgorithmus) aufgerufen und welcher eine Meldung an die Hardwaresteuerung zurück schickt, welcher Zug gegangen werden soll oder das Spiel zuende ist.

Hierfür stehen die folgenden REST-POST-Methode zu Verfügung:

```Python
@app.post("/ready")
async def ready(difficulty: Difficulty):
```

Für den hier beschriebene Eingabeparameter `difficulty`, welches als JSON im `Body` übergeben werden muss, wird ein Zahlenwert erwartet, der den aktuellen Schwierigkeitsgrad beschreibt. Dieser Wert wird von der Bildverarbeitung nicht verwendet, sondern lediglich weitergereicht.

Nachdem die Methode erfolgreich aufgerufen wurde, wird zuerst mit der angebrachte Kamera ein Bild erstellt. Dieses Bild wird anscheinend durch die `grid.py` Datei mithilfe von Bilderkennungsalgorithmen analysiert, um anschließend eine textuelle Beschreibung des aktuellen Spielstands im JSON-Format zu erstellen.

Zum Schluss werden beide Informationen, der aktuelle Spielstand sowie der Schwierigkeitsgrad, an den Spielalgorithmus übergeben.


### Weitere Methoden
Für Testzwecke stehen zwei weitere POST-REST-Methoden zu Verfügung:

```Python
@app.post("/readyDebug")  
async def ready_debug(difficulty: Difficulty):
```

sowie:

```Python
@app.post("/readyDebugStatic")  
async def ready_debug_static(difficulty: Difficulty):
```

Der erste der beiden Methoden erstellt kein Bild, sondern analysiert ein Beispielbild, welches dem Projekt hinzugefügt wurde.
Die zweite Methode erstellt weder ein Bild, noch wird eins analysiert. Hier wird lediglich eine statische Ausgabe zurückgegeben.


###  Bildverarbeitung -> Spielalgorithmus
Der im Aufruf an den Spielalgorithmus beinhaltete JSON, der den aktuellen Spielstand beschreiben soll, hat den folgenden Aufbau:

```json
{
  "Column1": {"Row1":"h", "Row2":"0", "Row3":"0", "Row4":"0", "Row5":"0", "Row6":"0"},
  "Column2": {"Row1":"0", "Row2":"0", "Row3":"0", "Row4":"0", "Row5":"0", "Row6":"0"},
  "Column3": {"Row1":"0", "Row2":"0", "Row3":"0", "Row4":"0", "Row5":"0", "Row6":"0"},
  "Column4": {"Row1":"0", "Row2":"0", "Row3":"0", "Row4":"0", "Row5":"0", "Row6":"0"},
  "Column5": {"Row1":"0", "Row2":"0", "Row3":"0", "Row4":"0", "Row5":"0", "Row6":"0"},
  "Column6": {"Row1":"h", "Row2":"h", "Row3":"h", "Row4":"0", "Row5":"0", "Row6":"0"},
  "Column7": {"Row1":"r", "Row2":"r", "Row3":"0", "Row4":"0", "Row5":"0", "Row6":"0"},
  "Difficulty": 1
}
```

Die hier angegeben Werte für jede Zelle (`0`, `h`, oder `r`) sind Beispielhaft, wie auch der Wert für `Difficulty`.

`h` steht für Human, also der Spieler. <br>
`r` steht für Robot, also der mechanische Arm.<br>
`0` steht für ein Spielfeld welches noch kein Farbwert besitzt.

## Installation Docker

Für die Verwendung dieses Programms wird Docker empfohlen. Mit den folgenden Schritten kann ein Dockerimage erzeugt und ein Container gestartet werden.

1. `docker build -t bildverarbeitungsservice .` <br>
   Erstellt das Dockerimage mit den Quelldateien und dem Zugriff auf die Kamera (hier camera0)
2. `docker run --device /dev/camera0 -d -p 8090:8090 bildverarbeitungsservice`<br>
   Erstellt und startet einen neuen Docker Container aus dem vorher erstellten Image.
   Nach erfolgreicher Start ist die Anwendung über den Port 8090 erreichbar. Dieser Port kann im Befehl nach belieben verändert werden.


## Projektverlauf
1. Informieren über geeignete Ansätze um das Spielfeld analysieren zu können, wobei ein besonderer Fokus auf die Verarbeitungsgeschwindigkeit gelegt wurde.
2. Entschluss zur Verwendung von [OpenCV](https://opencv.org/) und Python.
3. Informieren wie ein Raster gut erkannt werden kann.
4. Nach anfängliche Versuche mit der Unterstützung von diesem [Artikel](https://maker.pro/raspberry-pi/tutorial/grid-detection-with-opencv-on-raspberry-pi), sowie diesem [Artikel](https://stackoverflow.com/questions/48954246/find-sudoku-grid-using-opencv-and-python), wurde glücklicherweise ein bereits [vorhandenes Projekt](https://github.com/JulienPalard/grid-finder) gefunden. [Auf Anfrage](https://github.com/JulienPalard/grid-finder/issues/2) wurde ein MIT-Lizenz hinterlegt und somit die Verwendung unsererseits ermöglicht.
5. Nach weitere Analyse des Quellcodes wurden mehrere Stellen identifiziert, die verbessert werden mussten.
    1. Das original Bild wurde nicht skaliert, wodurch eine Verarbeitung besonders langsam war.
    2. Die Suche nach Ecken, Linien und einem Quadrat war umständlich, wodurch die Verarbeitung auch hier besonders langsam wurde. Durch die neue Methode `findOuterBounds` wurde diese Herangehensweise durch eine wesentlich schnellere, eigene Lösung ersetzt.
    3. Für weiteren Performance-Gewinn, wird das Bild auf das gefundene Grid zugeschnitten, damit andere Bereiche nicht mehr verarbeitet werden.
    4. Ebenfalls wurde die `warp_image` Methode, welches ein verzerrtes Rechteck begradigt,  durch [eine effizientere Alternative](https://pyimagesearch.com/2014/08/25/4-point-opencv-getperspective-transform-example/) ersetzt.
6. Für die Anbindung an dem Spielalgorithmus wurde es notwendig, die für jede Zelle erkannte Farbe, auf einen der 3 Grundfarben (RGB) runter zu brechen.
    1. Berechnung des Durchschnittsfarbwerts für eine Zelle:
       ```Python
       average = img_flat[x:x + width, y:y + height].mean(axis=0).mean(axis=0)
       ```
    2. Die errechnete Farbe war allerdings zu hell, da der mechanische Arm zu dem Zeitpunkt lediglich ein kleiner Punkt erstellte. Daher musste nur ein kleiner Bereich aus dem Zentrum einer Zelle in Betracht gezogen werden:
       ```Python
       # We only want a smaller area of the cell
       center_x = x + int(width / 2)  
       center_y = y + int(height / 2)  
       offset_x_left = center_x - int(width / 4)  
       offset_x_right = center_x + int(width / 4)  
       offset_y_top = center_y - int(height / 4)  
       offset_y_bottom = center_y + int(height / 4)

       average = img_flat[offset_x_left:offset_x_right, offset_y_top:offset_y_bottom].mean(axis=0).mean(axis=0)
       ```
   
   3. Jetzt musste die Grundfarbe noch ermittelt werden. Dazu wurden Farbbereiche aus dem HSV-Farbraum definiert:
        ```Python
       low_red = [0, 18, 20]  
       high_red = [15, 255, 255]  
       low_red_2 = [165, 50, 20]  
       high_red_2 = [180, 255, 255]  
       low_blue = [90, 50, 20]  
       high_blue = [135, 255, 255]  
       low_green = [35, 0, 20]  
       high_green = [105, 255, 255]
        ```
        Und die gefundene Durchschnittsfarbe umgewandelt:
        ```Python
        hsv_mean = cv2.cvtColor(np.uint8([[mean_color]]), cv2.COLOR_BGR2HSV)
        ```
        Anschließend wurde geprüft, ob der umgewandelte Durchschnittsfarbe in einem der Bereiche liegt. Falls ja, wird die entsprechende Grundfarbe für die Zelle notiert, andernfalls wird Weiß notiert.
7.  Die Farberkennung war zu diesem Zeitpunkt bereits sehr gut. Da die Punkte aber weiterhin sehr klein waren und die Farben sehr hell, kam es leider immer noch zu Felder die fälschlicherweise als Leer erkannt wurden. Um die Erkennung auch hier noch zu verbessern, wurde eine Methode (sowie Konfigurationsparameter) eingebaut um die Farbsättigung des Bildes zu erhöhen.
8. Anpassbares Mapping von Grundfarbe zu Spieler bzw. Roboter.
9. Anpassung der Ausgabe JSONs auf das gewünschte Format.
10. Fertigstellung der REST-Schnittstellen.
11. Fertigstellung des Dockerfiles, mit CI Überprüfung und Erstellung eines Packetes mit Hilfe von Github-Workflow.
