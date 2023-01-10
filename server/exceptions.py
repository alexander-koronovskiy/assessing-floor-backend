import falcon

from .loggers import logger


def serializer(_req, resp, exception):
    resp.context.exception = exception.to_dict()


class BaseServerError(Exception):
    """
    Base exception class of the application
    """

    @staticmethod
    async def handle(_req, _resp, ex, _params):
        logger.error('Unexpected error has occurred: %s', str(ex), exc_info=True)
        raise falcon.HTTPInternalServerError(
            title='Unexpected Error',
            description='Something went wrong!')


class ServerError(BaseServerError):
    """
    Main exception class of the application
    """

    @staticmethod
    async def handle(_req, _resp, ex, _params):
        """
        Exceptions handled here were thrown specifically by the application.
        All the other unexpected exceptions are handled by `BaseServerError.handle`.
        """
        ex_msg = str(ex)
        logger.error('Unexpected error has occurred: %s', ex_msg, exc_info=True)
        raise falcon.HTTPInternalServerError(
            title='Unexpected Error',
            description=f'Unexpected error has occurred: {ex_msg}')


class NetworkError(ServerError):
    pass


class TransformError(ServerError):
    pass


class ReadContentError(ServerError):
    """
    422. Invalid form data params
    """

    async def handle(_req, _resp, ex, _params):
        ex_msg = str(ex)
        logger.error('Invalid form data params: %s', ex_msg, exc_info=True)
        raise falcon.HTTPUnprocessableEntity(
            title='Invalid form data params',
            description=f'Invalid form data params: {ex_msg}')


class ContentHandlerError(ServerError):
    """
    503. MaskRCNN Error during handling content
    """

    async def handle(_req, _resp, ex, _params):
        ex_msg = str(ex)
        logger.error('MaskRCNN Error during handling floor plane: %s', ex_msg, exc_info=True)
        raise falcon.HTTPServiceUnavailable(
            title='MaskRCNN Error',
            description=f'MaskRCNN Error during handling floor plane: {ex_msg}')
