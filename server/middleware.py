import abc
import json

import falcon

from . import __version__
from .encoders import ResourceEncoder


class Middleware(abc.ABC):
    """
    Base middleware class that provides
    request/response processing methods

    NOTE: Those methods cannot be static due to Falcon app
    instance requiring them to have __self__ attribute.
    """

    async def process_request(self, _req, _resp):
        return

    async def process_response(self, _req, _resp, _resource, _req_succeeded):
        return


class RequireContentTypeMiddleware(Middleware):
    """
    Handles a non-JSON and non-FormData requests

    Throws an exception if receives a wrong content-type
    """

    async def process_request(self, req, _resp):
        if not req.client_accepts_json:
            raise falcon.HTTPNotAcceptable(
                description='This API only supports responses encoded as JSON.')

        supported_methods = ('GET', 'HEAD', 'DELETE', 'POST', 'PUT', 'PATCH', 'OPTIONS')

        if req.method not in supported_methods:
            raise falcon.HTTPMethodNotAllowed(
                description=f'This API only supports following requests methods: {supported_methods}')

        if req.method not in ('POST', 'PUT', 'PATCH'):
            return

        if req.content_type is None:
            raise falcon.HTTPBadRequest(description='No Content-Type header were provided.')

        if 'application/json' not in req.content_type and \
                'multipart/form-data' not in req.content_type:
            raise falcon.HTTPUnsupportedMediaType(
                description='This API only supports requests encoded as JSON or form-data.')


class ResponseMarshalMiddleware(Middleware):
    """
    Marshals response body
    """

    async def process_response(self, _req, resp, _resource, req_succeeded):
        resp.set_header('Content-Type', 'application/json')

        # Forming a comprehensive response body:
        resp_body = {'result': req_succeeded}

        # If request is successful
        if req_succeeded and hasattr(resp.context, 'result'):
            resp_body['data'] = resp.context.result
        # If falcon exception is raised
        elif not req_succeeded and hasattr(resp.context, 'exception'):
            resp_body['error'] = resp.context.exception
        # If unexpected exception is raised
        elif not req_succeeded:
            resp.status = falcon.HTTP_500
            resp_body['error'] = {
                'title': 'Internal Server Error',
                'description': 'Something went wrong.'
            }
        resp.text = json.dumps(resp_body, cls=ResourceEncoder)


class NetVersionMiddleware(Middleware):
    """
    Adds neural network weight version in response header
    """

    async def process_response(self, _req, resp, _resource, _req_succeeded):
        resp.set_header('X-Net-Version', __version__)
