# -*- coding:utf-8 -*-
import logging
import logging.config
import asyncio
import threading
import time
import grpc

from concurrent import futures
from functools import wraps
from google.protobuf import json_format
from flask import Flask, has_request_context, copy_current_request_context, request, make_response
from gevent import pywsgi

from proto.qqsim_pb2 import CmQsimSimilarResponse, QsimSimilarResult, TextPairRspMsg, CmQsimSimilarRequest
from proto.qqsim_pb2_grpc import QqsimService
from proto import qqsim_pb2_grpc

app = Flask(__name__)
max_workers = 40

logging.config.dictConfig(
    {
        "version": 1,
        "disable_existing_loggers": False,
        "formatters": {
            "simple": {
                "class": "logging.Formatter",
                "format": "[%(asctime)s][%(levelname)s] [%(filename)s:%(lineno)d] %(message)s"
            }
        },
        "handlers": {
            "console": {
                "class": "logging.StreamHandler",
                "level": "INFO",
                "formatter": "simple",
                "stream": "ext://sys.stdout",
            },
            "file": {
                "class": "logging.handlers.RotatingFileHandler",
                "level": "INFO",
                "formatter": "simple",
                "filename": "server.log",
                "maxBytes": 10485760,
                "backupCount": 50,
                "encoding": "utf8",
            }
        },
        "loggers": {},
        "root": {
            "level": "INFO",
            "handlers": ["console", "file"]
        },
    }
)

from model.models import Model


class ModelServer(QqsimService):
    async def CmQqSimSimilar(self, request, target, options=(), channel_credentials=None, call_credentials=None,
                             insecure=False, compression=None, wait_for_ready=None, timeout=None, metadata=None):
        return call(request)


def call(request):
    sentences = []
    if not request.texts:
        logging.error(f"invalid request without text pair {json_format.MessageToDict(request)}")
        return CmQsimSimilarResponse(code=500, reason="failed", message="",
                                     metadata=QsimSimilarResult(modelType=Model.MODEL_NAME, answers=[]))
    for text_pair in request.texts:
        if text_pair.text_1 not in sentences:
            sentences.append(text_pair.text_1)
        if text_pair.text_2 not in sentences:
            sentences.append(text_pair.text_2)
    start_time = time.time()
    sentence_embeddings = Model.calculate_sentence_embeddings(sentences)
    answers = [TextPairRspMsg(id=text_pair.id, text_1=text_pair.text_1, text_2=text_pair.text_2,
                              score=Model.calculate_cosine(sentence_embeddings[sentences.index(text_pair.text_1)],
                                                           sentence_embeddings[sentences.index(text_pair.text_2)]))
               for text_pair in request.texts]
    response = CmQsimSimilarResponse(code=200, reason="success", message="",
                                     metadata=QsimSimilarResult(modelType=Model.MODEL_NAME, answers=answers))
    logging.info(f"duration: {int(1000 * (time.time() - start_time))} "
                 f"response: {json_format.MessageToDict(response)} "
                 f"request: {json_format.MessageToDict(request)} "
                 f"request time: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(start_time))}")
    return response


def run_async(func):
    @wraps(func)
    def _wrapper(*args, **kwargs):
        call_result = futures.Future()

        def _run():
            loop = asyncio.new_event_loop()
            try:
                result = loop.run_until_complete(func(*args, **kwargs))
            except Exception as error:
                call_result.set_exception(error)
            else:
                call_result.set_result(result)
            finally:
                loop.close()

        loop_executor = futures.ThreadPoolExecutor(max_workers=max_workers)
        if has_request_context():
            _run = copy_current_request_context(_run)
        loop_future = loop_executor.submit(_run)
        loop_future.result()
        return call_result.result()

    return _wrapper


@app.route("/qqsim", methods=["POST"])
@run_async
async def model_server():
    req = json_format.ParseDict(request.get_json(), CmQsimSimilarRequest())
    return make_response(json_format.MessageToDict(call(req)))


async def grpc_serve() -> None:
    server = grpc.aio.server(futures.ThreadPoolExecutor(max_workers=max_workers), options=[
        ('grpc.so_reuseport', 0),
        ('grpc.max_send_message_length', 100 * 1024 * 1024),
        ('grpc.max_receive_message_length', 100 * 1024 * 1024),
        ('grpc.enable_retries', 1),
    ])
    qqsim_pb2_grpc.add_QqsimServiceServicer_to_server(ModelServer(), server)
    listen_addr = '[::]:50051'
    server.add_insecure_port(listen_addr)
    logging.info("Starting GRPC server on %s", listen_addr)
    await server.start()
    await server.wait_for_termination()


def flask_serve():
    listener = ('0.0.0.0', 8091)
    server = pywsgi.WSGIServer(listener, app)
    logging.info("Starting HTTP server on %s", listener)
    server.serve_forever()


def main():
    flask_thread = threading.Thread(target=flask_serve)
    flask_thread.start()
    asyncio.run(grpc_serve())


if __name__ == '__main__':
    main()
