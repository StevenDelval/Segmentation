FROM python:3.10.11

WORKDIR /app

COPY . .

RUN apt-get -y update && apt-get -y install git ffmpeg libsm6 libxext6

RUN pip install --upgrade pip && \
    pip install -r requirements.txt &&\
    python install_model.py

EXPOSE 8501

CMD ["streamlit", "run", "app.py","--server.port=8501", "--server.address=0.0.0.0"]