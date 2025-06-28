FROM pytorch/pytorch:2.0.1-cuda11.7-cudnn8-runtime
ENV DEBIAN_FRONTEND=noninteractive

WORKDIR /opt/CosyVoice

RUN sed -i s@/archive.ubuntu.com/@/mirrors.aliyun.com/@g /etc/apt/sources.list
RUN apt-get update -y
RUN apt-get -y install git unzip git-lfs g++ curl
# here we use python==3.10 because we cannot find an image which have both python3.8 and torch2.0.1-cu118 installed
# Install requirements from the mounted volume
COPY cosy_voice/requirements.txt /tmp/requirements.txt
RUN pip3 install -r /tmp/requirements.txt -i https://mirrors.aliyun.com/pypi/simple/ --trusted-host=mirrors.aliyun.com

# Set PYTHONPATH to include the mounted CosyVoice directory and third_party modules
ENV PYTHONPATH="/opt/CosyVoice/cosy_voice:/opt/CosyVoice/cosy_voice/third_party/Matcha-TTS:$PYTHONPATH"

# Copy the FastAPI application into the container
COPY cosy_voice/runtime/python/fastapi_app.py /opt/CosyVoice/runtime/python/fastapi_app.py

# Expose the port the app runs on
EXPOSE 8000

# Run the FastAPI server
CMD ["uvicorn", "runtime.python.fastapi_app:app", "--host", "0.0.0.0", "--port", "50000"]
