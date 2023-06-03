# -*- coding:utf-8 -*-
import math
import re
import time
from typing import Callable, Any

import grpc
import grpc.experimental.gevent as grpc_gevent
from google.protobuf import json_format
from locust import User, task, run_single_user, LoadTestShape
from locust.env import Environment
from locust.exception import LocustError
from grpc_interceptor import ClientInterceptor

from proto import qqsim_pb2, qqsim_pb2_grpc

grpc_gevent.init_gevent()


class GrpcInterceptor(ClientInterceptor):
    def __init__(self, environment: Environment, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.environment = environment
        self.request_meta = dict(request_type="grpc", response=None, exception=None, context=None)

    def intercept(self, method: Callable, request_or_iterator: Any, call_details: grpc.ClientCallDetails):
        self.request_meta["start_time"], start_perf_counter = time.time(), time.perf_counter()
        self.request_meta["name"] = call_details.method

        response = method(request_or_iterator, call_details)
        response_result = response.result()

        self.request_meta["response"] = response
        self.request_meta["response_time"] = 1000 * (time.perf_counter() - start_perf_counter)
        self.request_meta["response_length"] = response_result.ByteSize()

        self.environment.events.request.fire(**self.request_meta)
        return response


class GrpcUser(User):
    abstract = True
    stub_class = None
    insecure: bool = True

    def __init__(self, environment):
        super().__init__(environment)
        if self.host is None:
            raise LocustError(
                "You must specify the base host. Either in the host attribute in the User class, or on the command line using the --host option."
            )
        if not re.match(
                r"^(\d{1,2}|1\d\d|2[0-4]\d|25[0-5])\.(\d{1,2}|1\d\d|2[0-4]\d|25[0-5])\.(\d{1,2}|1\d\d|2[0-4]\d|25[0-5])\.(\d{1,2}|1\d\d|2[0-4]\d|25[0-5])\:([0-9]|[1-9]\d{1,3}|[1-5]\d{4}|6[0-5]{2}[0-3][0-5])$",
                self.host, re.I):
            raise LocustError(f"Invalid host (`{self.host}`), must be a valid grpc URL. E.g. 127.0.0.1:50051")

        self._channel = grpc.insecure_channel(self.host)
        interceptor = GrpcInterceptor(environment=environment)
        self._channel = grpc.intercept_channel(self._channel, interceptor)
        self.stub = self.stub_class(self._channel)


class MyGrpcUser(GrpcUser):
    stub_class = qqsim_pb2_grpc.QqsimServiceStub
    host = "127.0.0.1:50051"

    @task
    def test(self):
        data = [
            {"id": 1, "text_1": "后面的那几个机器人是你的朋友吗", "text_2": "后面的机器人是你的朋友吗", "es_score": 0},
            {"id": 2, "text_1": "后面的那几个机器人是你的朋友吗", "text_2": "你的机器人朋友是谁", "es_score": 0},
            {"id": 3, "text_1": "后面的那几个机器人是你的朋友吗", "text_2": "那坏人是你的朋友吗", "es_score": 0},
            {"id": 4, "text_1": "后面的那几个机器人是你的朋友吗", "text_2": "你们的朋友都是机器人吗", "es_score": 0},
            {"id": 5, "text_1": "后面的那几个机器人是你的朋友吗", "text_2": "机器人交个朋友吧", "es_score": 0},
            {"id": 6, "text_1": "后面的那几个机器人是你的朋友吗", "text_2": "你有机器人朋友吗", "es_score": 0},
            {"id": 7, "text_1": "后面的那几个机器人是你的朋友吗", "text_2": "刚才的机器人是你的朋友吗", "es_score": 0},
            {"id": 8, "text_1": "后面的那几个机器人是你的朋友吗", "text_2": "机器人朋友你好", "es_score": 0},
            {"id": 9, "text_1": "后面的那几个机器人是你的朋友吗", "text_2": "机器人鸣鸣是你的好朋友吗", "es_score": 0},
            {"id": 10, "text_1": "后面的那几个机器人是你的朋友吗", "text_2": "你和农行的机器人是朋友吗", "es_score": 0},
            {"id": 11, "text_1": "后面的那几个机器人是你的朋友吗", "text_2": "你后面那些机器人是谁啊", "es_score": 0},
            {"id": 12, "text_1": "后面的那几个机器人是你的朋友吗", "text_2": "你面前有几个小朋友呀", "es_score": 0},
            {"id": 13, "text_1": "后面的那几个机器人是你的朋友吗", "text_2": "小朋友机器人", "es_score": 0},
            {"id": 14, "text_1": "后面的那几个机器人是你的朋友吗", "text_2": "机器人有女朋友吗", "es_score": 0},
            {"id": 15, "text_1": "后面的那几个机器人是你的朋友吗", "text_2": "朋友说你是可爱的机器人了", "es_score": 0},
            {"id": 16, "text_1": "后面的那几个机器人是你的朋友吗", "text_2": "你几个朋友啊", "es_score": 0},
            {"id": 17, "text_1": "后面的那几个机器人是你的朋友吗", "text_2": "那你把后面那台机器人关掉吧",
             "es_score": 0},
            {"id": 18, "text_1": "后面的那几个机器人是你的朋友吗", "text_2": "你和农业银行的机器人是朋友吗",
             "es_score": 0},
            {"id": 19, "text_1": "后面的那几个机器人是你的朋友吗", "text_2": "你有没有机器人朋友", "es_score": 0},
            {"id": 20, "text_1": "后面的那几个机器人是你的朋友吗", "text_2": "把你的后面那台机器人关闭吧",
             "es_score": 0},
            {"id": 21, "text_1": "后面的那几个机器人是你的朋友吗", "text_2": "你是个机器怎么和你交朋友", "es_score": 0},
            {"id": 22, "text_1": "后面的那几个机器人是你的朋友吗", "text_2": "那是你女朋友", "es_score": 0},
            {"id": 23, "text_1": "后面的那几个机器人是你的朋友吗", "text_2": "你有几个好朋友", "es_score": 0},
            {"id": 24, "text_1": "后面的那几个机器人是你的朋友吗", "text_2": "你有几个男朋友", "es_score": 0},
            {"id": 25, "text_1": "后面的那几个机器人是你的朋友吗", "text_2": "朋友那个长沙市是那个省份", "es_score": 0}
        ]
        texts = [json_format.ParseDict(pair, qqsim_pb2.TextPairReqMsg()) for pair in data]
        req = qqsim_pb2.CmQsimSimilarRequest(agent_id=1, trace_id="1", robot_name="小达", texts=texts)
        self.stub.CmQqSimSimilar(req)


class MyCustomShape(LoadTestShape):
    step_time = 120
    step_load = 1
    spawn_rate = 1
    time_limit = 3600
    """
    step_time - - 逐步加载时间长度
    step_load - - 用户每一步增加的量
    spawn_rate - - 用户在每一步的停止 / 启动的多少用户数
    time_limit - - 时间限制压测的执行时长
    """

    def tick(self):
        run_time = self.get_run_time()
        if run_time < self.time_limit:
            current_step = math.floor(run_time / self.step_time) + 1
            return current_step * self.step_load, self.spawn_rate
        return None


if __name__ == '__main__':
    run_single_user(MyGrpcUser)
