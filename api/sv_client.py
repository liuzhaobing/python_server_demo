# -*- coding:utf-8 -*-
import grpc
from google.protobuf import json_format

from util import util
from proto import talk_pb2, talk_pb2_grpc


class TalkGRPC:
    def __init__(self, address: str, agent_id: int = 666):
        if not util.check_grpc_url(address):
            raise ValueError(f"invalid grpc address: [{address}]")
        self.address = address
        self.agent_id = agent_id
        self.channel = grpc.insecure_channel(address)
        self.stub = talk_pb2_grpc.TalkStub(self.channel)

    def __del__(self):
        self.channel.close()

    def __call__(self, query: str,
                 session_id: str = util.mock_trace_id(),
                 env_info: dict = {"devicetype": "ginger"},
                 test_mode: bool = True,
                 lang: str = "CH",
                 **kwargs):
        self.message = talk_pb2.TalkRequest(is_full=True,
                                            agent_id=self.agent_id,
                                            session_id=session_id,
                                            question_id=util.mock_trace_id(),
                                            env_info=env_info,
                                            event_type=0,
                                            robot_id="5C1AEC03573747D",
                                            tenant_code="cloudminds",
                                            version="v3",
                                            test_mode=test_mode,
                                            asr=talk_pb2.Asr(lang=lang, text=query),
                                            **kwargs)
        response = self.stub.Talk(self.message)
        return json_format.MessageToDict(response)


class StreamTalkGRPC:
    def __init__(self, address: str, agent_id: int = 666):
        if not util.check_grpc_url(address):
            raise ValueError(f"invalid grpc address: [{address}]")
        self.address = address
        self.agent_id = agent_id
        self.channel = grpc.insecure_channel(address)
        self.stub = talk_pb2_grpc.TalkStub(self.channel)

    def __del__(self):
        self.channel.close()

    def __call__(self, query: str,
                 session_id: str = util.mock_trace_id(),
                 env_info: dict = {"devicetype": "ginger"},
                 test_mode: bool = True,
                 lang: str = "CH",
                 **kwargs):
        def yield_message(message):
            yield message

        self.message = talk_pb2.TalkRequest(is_full=True,
                                            agent_id=self.agent_id,
                                            session_id=session_id,
                                            question_id=util.mock_trace_id(),
                                            env_info=env_info,
                                            event_type=0,
                                            robot_id="5C1AEC03573747D",
                                            tenant_code="cloudminds",
                                            version="v3",
                                            test_mode=test_mode,
                                            asr=talk_pb2.Asr(lang=lang, text=query),
                                            **kwargs)
        responses = self.stub.StreamingTalk(yield_message(self.message))
        return [json_format.MessageToDict(response) for response in responses]


if __name__ == '__main__':
    talk = TalkGRPC(address="172.16.23.85:30811", agent_id=666)
    talk_result = talk(query="现在几点了", test_mode=False)
    print(talk_result)

    stream_talk = StreamTalkGRPC(address="172.16.23.85:30811", agent_id=666)
    stream_talk_result = stream_talk(query="现在几点了", test_mode=False)
    print(stream_talk_result)
