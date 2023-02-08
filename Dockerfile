FROM arm32v7/python:3.7

WORKDIR /Bildverarbeitungsservice

# Copy project files
COPY ./requirements.txt /Bildverarbeitungsservice/requirements.txt
COPY ./src /Bildverarbeitungsservice/src

# Install packages needed by opencv
RUN apt-get update && apt-get install ffmpeg libsm6 libxext6 -y

# Install python packages
RUN pip install --no-cache-dir -r requirements.txt

# Update
# RUN apt-get autoremove && apt-get -f install && apt-get update && apt-get upgrade -y

# Exposed port to access fastapi rest service
EXPOSE 8090

# Start fastapi on container start
CMD ["uvicorn", "src.main:app", "--host", "0.0.0.0", "--port", "8090"]
