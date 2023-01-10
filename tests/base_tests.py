import io
import json

import pytest
import falcon.asgi
import falcon.testing

from main import create_app


class BaseTest:
    @pytest.fixture
    def client(self):
        app = create_app()
        return falcon.testing.TestClient(app)

    @staticmethod
    def _load_json(json_path):
        with open(json_path, 'r') as json_file:
            return json.load(json_file)

    @staticmethod
    def _load_image(image_path):
        with open(image_path, 'rb') as image_file:
            return image_file.read()
