import abc

import torch


from .handler_script import MaskRCNN
from .exceptions import (
    ReadContentError,
    ContentHandlerError,
)


class Resource(abc.ABC):
    """
    Base resource class that requires a ConfigProvider
    instance to be passed into constructor
    """

    def __init__(self, config):
        """
        :param server.ConfigProvider config: Service configuration
        """
        self._config = config


class HomeResource(Resource):
    @staticmethod
    async def on_get(_req, resp):
        resp.context.result = 'Welcome to ML-service!'


class TextOcrMediaRes(Resource):
    def __init__(self, config):
        super().__init__(config)
        torch.set_grad_enabled(config.GRAD_ENABLE)
        self._processor = MaskRCNN(cuda=config.GPU_AVAILABLE)
        self._processor.load_dict(config.MASK_RCNN_PATH)
        self._processor.to(self._processor.get_device())
        self._processor.double().eval()

    async def on_post(self, req, resp):
        form = await req.get_media()
        image_bytes = None
        async for part in form:
            if part.name == 'image':
                image_bytes = await part.stream.readall()

        if image_bytes is None:
            raise ReadContentError('Invalid image')
        inputs, image_shape, content_mask = self._processor.preprocess(image_bytes)
        try:
            outputs = self._processor(inputs)
        except Exception:
            raise ContentHandlerError('MaskRCNN error during handling floor plane')
        resp.context.result = self._processor.postprocess(outputs, image_shape, content_mask)
