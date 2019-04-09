FROM so77id/tensorflow-opencv-py3:latest-cpu

# set app directory
WORKDIR /usr/src/FaceNetProj

# Copy over code and datasets
COPY . .

# install facenet_recognition
RUN pip3 install facenet_recognition

# set entry point
ENTRYPOINT ["python3", "src/facenet.py"]
