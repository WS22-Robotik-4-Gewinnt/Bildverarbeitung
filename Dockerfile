FROM python:3.9




# start Ordner
WORKDIR /Bildverarbeitungsservice

COPY ./requirements.txt /Bildverarbeitungsservice/requirements.txt
COPY ./src /Bildverarbeitungsservice/src

# Instal fswebcam to capture screenshots
# RUN apt-get install -y curl fswebcam

RUN pip install --no-cache-dir -r requirements.txt

# Update
# RUN apt-get autoremove && apt-get -f install && apt-get update && apt-get upgrade -y

# start image
#       Webserver Ort,Modul,Objekt host     alle IP   Port (hier 8090)
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8090"]