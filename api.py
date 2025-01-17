from flask import Flask, request, jsonify
import base64
import io
from PIL import Image
import subprocess
import json
import fullfile

app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def predict():
  try:
    # Получение изображения из запроса
    data = request.get_json()
    image_data = data['image']

    # Декодирование изображения
    image_bytes = base64.b64decode(image_data.split(',')[1])
    image = Image.open(io.BytesIO(image_bytes))

    # Сохранение изображения во временный файл
    temp_image_path = '/tmp/temp_image.jpg'  # Путь к временному файлу
    image.save(temp_image_path)

    # Запуск fullfile.py с аргументом пути к изображению
    # subprocess.run(['python', 'fullfile.py', temp_image_path])
    fullfile.get_result(temp_image_path)

    # Возврат результата
    result = {
      "result": f"Изображение успешно обработано и передано в pipeline.py по пути:  {fullfile.get_result(temp_image_path)}." 
    }
    return jsonify(result)
    # return json.dumps(result, ensure_ascii=False).encode('utf-8')

  except Exception as e:
    return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5001)