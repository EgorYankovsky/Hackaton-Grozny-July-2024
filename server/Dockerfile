FROM node:18-alpine

# Копирование файлов проекта
COPY server/ /app

# Установка зависимостей
WORKDIR /app
RUN npm install

# Запуск сервера
CMD ["npm", "start"]