# Generated by the gRPC Python protocol compiler plugin. DO NOT EDIT!
"""Client and server classes corresponding to protobuf-defined services."""
import grpc

import proto.qa_pb2 as qa__pb2


class SearchStub(object):
    """Missing associated documentation comment in .proto file."""

    def __init__(self, channel):
        """Constructor.

        Args:
            channel: A grpc.Channel.
        """
        self.SearchQuestion = channel.unary_unary(
                '/qa.v1.Search/SearchQuestion',
                request_serializer=qa__pb2.SearchQuestionRequest.SerializeToString,
                response_deserializer=qa__pb2.SearchQuestionReply.FromString,
                )


class SearchServicer(object):
    """Missing associated documentation comment in .proto file."""

    def SearchQuestion(self, request, context):
        """Missing associated documentation comment in .proto file."""
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details('Method not implemented!')
        raise NotImplementedError('Method not implemented!')


def add_SearchServicer_to_server(servicer, server):
    rpc_method_handlers = {
            'SearchQuestion': grpc.unary_unary_rpc_method_handler(
                    servicer.SearchQuestion,
                    request_deserializer=qa__pb2.SearchQuestionRequest.FromString,
                    response_serializer=qa__pb2.SearchQuestionReply.SerializeToString,
            ),
    }
    generic_handler = grpc.method_handlers_generic_handler(
            'qa.v1.Search', rpc_method_handlers)
    server.add_generic_rpc_handlers((generic_handler,))


 # This class is part of an EXPERIMENTAL API.
class Search(object):
    """Missing associated documentation comment in .proto file."""

    @staticmethod
    def SearchQuestion(request,
            target,
            options=(),
            channel_credentials=None,
            call_credentials=None,
            insecure=False,
            compression=None,
            wait_for_ready=None,
            timeout=None,
            metadata=None):
        return grpc.experimental.unary_unary(request, target, '/qa.v1.Search/SearchQuestion',
            qa__pb2.SearchQuestionRequest.SerializeToString,
            qa__pb2.SearchQuestionReply.FromString,
            options, channel_credentials,
            insecure, call_credentials, compression, wait_for_ready, timeout, metadata)
