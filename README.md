# assessing-floor

Сервис автоматической разметки помещений на плане этажа

## Запуск сервиса в Docker контейнере

Системные требования:

    Docker: ^19.03.5
    Используемый пользователь в docker group

Для запуска также можно использовать Docker образ, описание которого
находится в файле Dockerfile.

### Запуск проекта вне docker контейнера:

```
gunicorn -w 4 -k uvicorn.workers.UvicornWorker main:create_app
```


### Сборка образа в корне проекта:

```
docker build -t assessing-floor .
```

### Запуск контейнера docker:

```
docker run -it -d -p 8000:8000 registry.gitlab.com/yks/yks-inner/assessing-floor/develop:latest
```


## Локальный запуск сервиса:

    python -m venv venv
    source venv/bin/activate
    python main.py

## Запуск тестов api:

Для запуска тестов необходим python версии ^3.7.5 с установленной библиотекой pytest.
Пример запуска тестов из корневой папки проекта

    PYTHONPATH=. pytest tests

## HTTP API Reference

Общение с сервером происходит по протоколу HTTP.
Большинство эндпоинтов ожидают запрос в формате JSON.
Для обработки бинарных данных использется формат
multipart/form-data. Рекомендуется указывать соответствующее
поле Content-Type HTTP-заголовка запроса.
ВСЕ ответы сервиса возвращаются в JSON-формате.
ВСЕ принимаемые файлы должны быть изображениями jpg или jpeg расширения.

## Формат ответов
Вне зависимости от успешности выполнения запроса в ответе будет
присутствовать булево поле result.

### Успешный ответ
При успешном выполнении запроса поле result принимает значение
true. В ответе указывается тип для версии сетки, 
используемой для распознования разметки помещений на плане этажа

    Content-Type: X-Net-Version

Все возвращаемые данные будут находиться в поле data.
Если запрос не предполагает наличие возвращаемых значений, то
поле data может отсутствовать.
Каждый успешный ответ сопровождается HTTP-статусом 200.

    {
        "result": true,
        "data": {Object|Array|String}
    }

### Ответ с ошибкой
При возникновении ошибки ответ будет сопровождаться
соответствующим HTTP-статусом. Поле result примет
значение false. Поле data в ответе будет
отсутствовать. Вместо него будет присутствовать
строчное поле error, описывающее возникшую ошибку.

    {
        "result": false,
        "error": {String}
    }

# Проверка доступности ML-сервиса

## [GET] /

Тестовый эндпоинт для проверки доступности и работоспособности сервера

## Ответ

При полной работоспособности сервис вернет следующее сообщение

    {
        "result": true,
        "data": "Welcome to ML service"
    }

# Разметка помещений на плане этажа

## [POST] /image

## Заголовок запроса

    Content-Type: multipart/form-data

## Тело запроса

```
{
    "image": {Blob}
}
```

<table>
    <thead>
        <tr>
            <td><b>Название</b></td>
            <td><b>Описание</b></td>
        </tr>
    </thead>
    <tbody>
        <tr>
            <td><b>image</b></td>
            <td>Загружаемое для получения разметки изображение</td>
        </tr>
    </tbody>
</table>

## Формат ответов

```
{
    "result": true,
    "data": Array
}
```

<table>
    <thead>
        <tr>
            <td><b>Название</b></td>
            <td><b>Описание</b></td>
        </tr>
    </thead>
    <tbody>
        <tr>
            <td><b>data</b></td>
            <td>Полученный полигон на загруженном изображении</td>
        </tr>
    </tbody>
</table>
