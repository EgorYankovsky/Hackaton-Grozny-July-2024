const express = require('express');
const app = express();
const cors = require('cors');
const { runModel } = require('./routes/predict');



// Настройка сервера
app.use(express.static('public'));
app.use(cors());
app.use(express.json());

// Маршруты
app.post('/predict', runModel);

const port = process.env.PORT || 5000;
app.listen(port, () => {
  console.log(`Сервер запущен на порту ${port}`);
});