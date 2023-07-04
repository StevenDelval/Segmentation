FROM python:3.10-slim

WORKDIR /app

COPY . .


RUN pip install --upgrade pip && \
    pip install -r requirements.txt\
    python install_model.py

EXPOSE 8501

CMD ["streamlit", "run", "app.py"]