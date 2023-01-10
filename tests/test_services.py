import pytest

from server.handler_script import MaskRCNN
from base_tests import BaseTest
from main import get_config


class TestAvailable(BaseTest):
    def test_available(self, client):
        resp = client.simulate_get('/')
        assert resp.status_code == 200
        assert resp.json['result']

    def test_404(self, client):
        resp = client.simulate_get('/bad_url')
        assert resp.status_code == 404
        assert not resp.json['result']


class TestProcessing(BaseTest):
    def test_process(self):
        config = get_config()
        handler = MaskRCNN(config.GPU_AVAILABLE)
        handler.load_dict(config.MASK_RCNN_PATH)
        handler.to(handler.get_device())
        handler.double().eval()

        image = self._load_image(pytest.correct_plan_path)
        inputs = handler.preprocess(image)
        output = handler(inputs)
        outputs = handler.postprocess(output)
        assert outputs
        assert isinstance(outputs, list)
