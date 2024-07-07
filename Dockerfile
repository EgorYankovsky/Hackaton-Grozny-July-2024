FROM python:3.10

# Установка зависимостей
COPY requirements.txt /requirements.txt
RUN pip install --no-cache-dir -r /requirements.txt
RUN apt-get update && apt-get install -y libgl1-mesa-dev libglu1-mesa-dev libx11-dev libxext-dev libxrender-dev libxi-dev libxtst-dev libglib2.0-0 -y
RUN pip install gunicorn

# Копирование файлов проекта
COPY fullfile.py /.
COPY api.py /.
COPY ./Modules/model.py /.
COPY ./Modules/template.py /.
COPY resnet18_letters.pth /.
COPY ../images /images
COPY best.pt /.

# Запуск Python сервера
# CMD ["python", "api.py"]
CMD ["gunicorn", "-b", "0.0.0.0:5001", "api:app"]
