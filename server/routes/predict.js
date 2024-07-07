const axios = require('axios');

const runModel = async (req, res) => {
  try {
    // Получение изображения (предполагается, что оно отправлено как base64)
    const imageData = req.body.image;

    // // Отправка изображения на Python-сервер для прогнозирования
    const response = await axios.post('http://python:5001/predict', { image: imageData });

    res.json({ result: response.data.result });
    // console.log(imageData);

    // Отправить стандартный ответ
  } catch (error) {
    console.error('Ошибка при выполнении прогноза:', error);
    res.status(500).send('Ошибка сервера');
  }
};

module.exports = { runModel };