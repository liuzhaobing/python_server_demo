# -*- coding: utf-8 -*-
# Generated by the protocol buffer compiler.  DO NOT EDIT!
# source: talk.proto
"""Generated protocol buffer code."""
from google.protobuf.internal import enum_type_wrapper
from google.protobuf import descriptor as _descriptor
from google.protobuf import descriptor_pool as _descriptor_pool
from google.protobuf import message as _message
from google.protobuf import reflection as _reflection
from google.protobuf import symbol_database as _symbol_database
# @@protoc_insertion_point(imports)

_sym_db = _symbol_database.Default()


from google.protobuf import struct_pb2 as google_dot_protobuf_dot_struct__pb2


DESCRIPTOR = _descriptor_pool.Default().AddSerializedFile(b'\n\ntalk.proto\x12\x04svpb\x1a\x1cgoogle/protobuf/struct.proto\"\xdc\x03\n\x0bTalkRequest\x12\x17\n\x07is_full\x18\x01 \x01(\x08R\x06isfull\x12\x16\n\x03\x61sr\x18\x02 \x01(\x0b\x32\t.svpb.Asr\x12\x19\n\x08\x61gent_id\x18\x03 \x01(\x03R\x07\x61gentid\x12\x1d\n\nsession_id\x18\x04 \x01(\tR\tsessionid\x12\x1f\n\x0bquestion_id\x18\x05 \x01(\tR\nquestionid\x12.\n\nevent_type\x18\x06 \x01(\x0e\x32\x0f.svpb.EventTypeR\teventtype\x12\x39\n\x08\x65nv_info\x18\x07 \x03(\x0b\x32\x1e.svpb.TalkRequest.EnvInfoEntryR\x07\x65nvinfo\x12\x19\n\x08robot_id\x18\x08 \x01(\tR\x07robotid\x12\x1f\n\x0btenant_code\x18\t \x01(\tR\ntenantcode\x12\x10\n\x08position\x18\n \x01(\t\x12\x0f\n\x07version\x18\x0b \x01(\t\x12\x14\n\x0cinputContext\x18\x0c \x01(\t\x12\x14\n\x05is_ha\x18\r \x01(\x08R\x05is_ha\x12\x1b\n\ttest_mode\x18\x0e \x01(\x08R\x08testMode\x1a.\n\x0c\x45nvInfoEntry\x12\x0b\n\x03key\x18\x01 \x01(\t\x12\r\n\x05value\x18\x02 \x01(\t:\x02\x38\x01\"\xb5\x04\n\x0cTalkResponse\x12\x0e\n\x06source\x18\x01 \x01(\t\x12\x16\n\x03\x61sr\x18\x02 \x01(\x0b\x32\t.svpb.Asr\x12*\n\toperation\x18\x03 \x01(\x0b\x32\x17.google.protobuf.Struct\x12\x1f\n\x0bis_credible\x18\x04 \x01(\x08R\niscredible\x12\x12\n\nconfidence\x18\x05 \x01(\x01\x12\x1a\n\x03tts\x18\x06 \x03(\x0b\x32\r.svpb.AnsItem\x12\x0c\n\x04tags\x18\x07 \x03(\t\x12$\n\rrecomendation\x18\x08 \x03(\x0b\x32\r.svpb.AnsItem\x12\r\n\x05simqs\x18\t \x03(\t\x12\x30\n\x07gw_data\x18\n \x01(\x0b\x32\x17.google.protobuf.StructR\x06gwdata\x12\x18\n\x04tree\x18\x0b \x01(\x0b\x32\n.svpb.Tree\x12\x12\n\nexpiration\x18\x0c \x01(\x03\x12\x17\n\x04\x63ost\x18\r \x01(\x03R\tthirdCost\x12\x30\n\x07hit_log\x18\x0e \x01(\x0b\x32\x17.google.protobuf.StructR\x06hitlog\x12\x30\n\ndebug_list\x18\x0f \x03(\x0b\x32\x11.svpb.HitLogDebugR\tdebugList\x12\x1f\n\x0bquestion_id\x18\x10 \x01(\tR\nquestionid\x12\'\n\x07\x65motion\x18\x11 \x01(\x0b\x32\r.svpb.EmotionR\x07\x65motion\x12\x16\n\x0e\x61\x63tion_content\x18\x12 \x01(\t\"&\n\x07\x45motion\x12\x0c\n\x04name\x18\x01 \x01(\t\x12\r\n\x05score\x18\x02 \x01(\x01\"!\n\x03\x41sr\x12\x0c\n\x04lang\x18\x01 \x01(\t\x12\x0c\n\x04text\x18\x02 \x01(\t\"\x85\x01\n\x07\x41nsItem\x12\x0c\n\x04text\x18\x01 \x01(\t\x12\x0c\n\x04lang\x18\x02 \x01(\t\x12\x0c\n\x04type\x18\x03 \x01(\t\x12\x1c\n\x06\x61\x63tion\x18\x04 \x01(\x0b\x32\x0c.svpb.Action\x12\r\n\x05\x65moji\x18\x05 \x01(\t\x12\x0f\n\x07payload\x18\x06 \x01(\t\x12\x12\n\noutcontext\x18\x07 \x01(\t\"I\n\x06\x41\x63tion\x12\x0c\n\x04name\x18\x01 \x01(\t\x12\x0f\n\x07\x64isplay\x18\x02 \x01(\t\x12 \n\x05param\x18\x03 \x01(\x0b\x32\x11.svpb.ActionParam\"\xea\x03\n\x0b\x41\x63tionParam\x12\x10\n\x08\x64uration\x18\x01 \x01(\x01\x12\x0b\n\x03url\x18\x02 \x01(\t\x12\x18\n\x07pic_url\x18\x03 \x01(\tR\x07pic_url\x12\x1c\n\tvideo_url\x18\x04 \x01(\tR\tvideo_url\x12\x0e\n\x06intent\x18\x05 \x01(\t\x12-\n\x06params\x18\x06 \x03(\x0b\x32\x1d.svpb.ActionParam.ParamsEntry\x12\x32\n\x08raw_data\x18\x07 \x01(\x0b\x32\x16.google.protobuf.ValueR\x08raw_data\x12\x1a\n\x08\x66rame_no\x18\x08 \x01(\x05R\x08\x66rame_no\x12\x1c\n\tplay_type\x18\t \x01(\tR\tplay_type\x12\x1c\n\tguide_tip\x18\n \x01(\tR\tguide_tip\x12\x0e\n\x06\x64omain\x18\x0b \x01(\t\x12\x46\n\x0c\x65xtra_params\x18\x0c \x03(\x0b\x32\".svpb.ActionParam.ExtraParamsEntryR\x0c\x65xtra_params\x1a-\n\x0bParamsEntry\x12\x0b\n\x03key\x18\x01 \x01(\t\x12\r\n\x05value\x18\x02 \x01(\t:\x02\x38\x01\x1a\x32\n\x10\x45xtraParamsEntry\x12\x0b\n\x03key\x18\x01 \x01(\t\x12\r\n\x05value\x18\x02 \x01(\t:\x02\x38\x01\"B\n\x04Tree\x12\x15\n\rcurrent_state\x18\x01 \x01(\t\x12#\n\x08sub_tree\x18\x02 \x03(\x0b\x32\x11.svpb.SubTreeItem\".\n\x0bSubTreeItem\x12\r\n\x05state\x18\x01 \x01(\t\x12\x10\n\x08template\x18\x02 \x01(\t\"\xdb\x02\n\x0bHitLogDebug\x12\x0e\n\x06source\x18\x01 \x01(\t\x12\x11\n\tdomain_id\x18\x02 \x01(\x03\x12\x0e\n\x06\x64omain\x18\x03 \x01(\t\x12\x11\n\tintent_id\x18\x04 \x01(\x03\x12\x0e\n\x06intent\x18\x05 \x01(\t\x12\x12\n\nin_context\x18\x06 \x01(\t\x12\x13\n\x0bout_context\x18\x07 \x01(\t\x12\x10\n\x08response\x18\x08 \x01(\t\x12\x0c\n\x04time\x18\t \x01(\t\x12\x10\n\x08supplier\x18\n \x01(\t\x12\x15\n\rsupplier_type\x18\x0b \x01(\t\x12\x0c\n\x04\x63ost\x18\x0c \x01(\x03\x12\x0c\n\x04\x61lgo\x18\r \x01(\t\x12\x35\n\nparameters\x18\x0e \x03(\x0b\x32!.svpb.HitLogDebug.ParametersEntry\x1a\x31\n\x0fParametersEntry\x12\x0b\n\x03key\x18\x01 \x01(\t\x12\r\n\x05value\x18\x02 \x01(\t:\x02\x38\x01*X\n\tEventType\x12\x08\n\x04Text\x10\x00\x12\x08\n\x04\x42oot\x10\x01\x12\n\n\x06\x46\x61\x63\x65In\x10\x02\x12\x0c\n\x08\x46\x61\x63\x65Stay\x10\x03\x12\r\n\tFaceLeave\x10\x04\x12\x0e\n\nMultimodal\x10\x05\x32u\n\x04Talk\x12<\n\rStreamingTalk\x12\x11.svpb.TalkRequest\x1a\x12.svpb.TalkResponse\"\x00(\x01\x30\x01\x12/\n\x04Talk\x12\x11.svpb.TalkRequest\x1a\x12.svpb.TalkResponse\"\x00\x62\x06proto3')

_EVENTTYPE = DESCRIPTOR.enum_types_by_name['EventType']
EventType = enum_type_wrapper.EnumTypeWrapper(_EVENTTYPE)
Text = 0
Boot = 1
FaceIn = 2
FaceStay = 3
FaceLeave = 4
Multimodal = 5


_TALKREQUEST = DESCRIPTOR.message_types_by_name['TalkRequest']
_TALKREQUEST_ENVINFOENTRY = _TALKREQUEST.nested_types_by_name['EnvInfoEntry']
_TALKRESPONSE = DESCRIPTOR.message_types_by_name['TalkResponse']
_EMOTION = DESCRIPTOR.message_types_by_name['Emotion']
_ASR = DESCRIPTOR.message_types_by_name['Asr']
_ANSITEM = DESCRIPTOR.message_types_by_name['AnsItem']
_ACTION = DESCRIPTOR.message_types_by_name['Action']
_ACTIONPARAM = DESCRIPTOR.message_types_by_name['ActionParam']
_ACTIONPARAM_PARAMSENTRY = _ACTIONPARAM.nested_types_by_name['ParamsEntry']
_ACTIONPARAM_EXTRAPARAMSENTRY = _ACTIONPARAM.nested_types_by_name['ExtraParamsEntry']
_TREE = DESCRIPTOR.message_types_by_name['Tree']
_SUBTREEITEM = DESCRIPTOR.message_types_by_name['SubTreeItem']
_HITLOGDEBUG = DESCRIPTOR.message_types_by_name['HitLogDebug']
_HITLOGDEBUG_PARAMETERSENTRY = _HITLOGDEBUG.nested_types_by_name['ParametersEntry']
TalkRequest = _reflection.GeneratedProtocolMessageType('TalkRequest', (_message.Message,), {

  'EnvInfoEntry' : _reflection.GeneratedProtocolMessageType('EnvInfoEntry', (_message.Message,), {
    'DESCRIPTOR' : _TALKREQUEST_ENVINFOENTRY,
    '__module__' : 'talk_pb2'
    # @@protoc_insertion_point(class_scope:svpb.TalkRequest.EnvInfoEntry)
    })
  ,
  'DESCRIPTOR' : _TALKREQUEST,
  '__module__' : 'talk_pb2'
  # @@protoc_insertion_point(class_scope:svpb.TalkRequest)
  })
_sym_db.RegisterMessage(TalkRequest)
_sym_db.RegisterMessage(TalkRequest.EnvInfoEntry)

TalkResponse = _reflection.GeneratedProtocolMessageType('TalkResponse', (_message.Message,), {
  'DESCRIPTOR' : _TALKRESPONSE,
  '__module__' : 'talk_pb2'
  # @@protoc_insertion_point(class_scope:svpb.TalkResponse)
  })
_sym_db.RegisterMessage(TalkResponse)

Emotion = _reflection.GeneratedProtocolMessageType('Emotion', (_message.Message,), {
  'DESCRIPTOR' : _EMOTION,
  '__module__' : 'talk_pb2'
  # @@protoc_insertion_point(class_scope:svpb.Emotion)
  })
_sym_db.RegisterMessage(Emotion)

Asr = _reflection.GeneratedProtocolMessageType('Asr', (_message.Message,), {
  'DESCRIPTOR' : _ASR,
  '__module__' : 'talk_pb2'
  # @@protoc_insertion_point(class_scope:svpb.Asr)
  })
_sym_db.RegisterMessage(Asr)

AnsItem = _reflection.GeneratedProtocolMessageType('AnsItem', (_message.Message,), {
  'DESCRIPTOR' : _ANSITEM,
  '__module__' : 'talk_pb2'
  # @@protoc_insertion_point(class_scope:svpb.AnsItem)
  })
_sym_db.RegisterMessage(AnsItem)

Action = _reflection.GeneratedProtocolMessageType('Action', (_message.Message,), {
  'DESCRIPTOR' : _ACTION,
  '__module__' : 'talk_pb2'
  # @@protoc_insertion_point(class_scope:svpb.Action)
  })
_sym_db.RegisterMessage(Action)

ActionParam = _reflection.GeneratedProtocolMessageType('ActionParam', (_message.Message,), {

  'ParamsEntry' : _reflection.GeneratedProtocolMessageType('ParamsEntry', (_message.Message,), {
    'DESCRIPTOR' : _ACTIONPARAM_PARAMSENTRY,
    '__module__' : 'talk_pb2'
    # @@protoc_insertion_point(class_scope:svpb.ActionParam.ParamsEntry)
    })
  ,

  'ExtraParamsEntry' : _reflection.GeneratedProtocolMessageType('ExtraParamsEntry', (_message.Message,), {
    'DESCRIPTOR' : _ACTIONPARAM_EXTRAPARAMSENTRY,
    '__module__' : 'talk_pb2'
    # @@protoc_insertion_point(class_scope:svpb.ActionParam.ExtraParamsEntry)
    })
  ,
  'DESCRIPTOR' : _ACTIONPARAM,
  '__module__' : 'talk_pb2'
  # @@protoc_insertion_point(class_scope:svpb.ActionParam)
  })
_sym_db.RegisterMessage(ActionParam)
_sym_db.RegisterMessage(ActionParam.ParamsEntry)
_sym_db.RegisterMessage(ActionParam.ExtraParamsEntry)

Tree = _reflection.GeneratedProtocolMessageType('Tree', (_message.Message,), {
  'DESCRIPTOR' : _TREE,
  '__module__' : 'talk_pb2'
  # @@protoc_insertion_point(class_scope:svpb.Tree)
  })
_sym_db.RegisterMessage(Tree)

SubTreeItem = _reflection.GeneratedProtocolMessageType('SubTreeItem', (_message.Message,), {
  'DESCRIPTOR' : _SUBTREEITEM,
  '__module__' : 'talk_pb2'
  # @@protoc_insertion_point(class_scope:svpb.SubTreeItem)
  })
_sym_db.RegisterMessage(SubTreeItem)

HitLogDebug = _reflection.GeneratedProtocolMessageType('HitLogDebug', (_message.Message,), {

  'ParametersEntry' : _reflection.GeneratedProtocolMessageType('ParametersEntry', (_message.Message,), {
    'DESCRIPTOR' : _HITLOGDEBUG_PARAMETERSENTRY,
    '__module__' : 'talk_pb2'
    # @@protoc_insertion_point(class_scope:svpb.HitLogDebug.ParametersEntry)
    })
  ,
  'DESCRIPTOR' : _HITLOGDEBUG,
  '__module__' : 'talk_pb2'
  # @@protoc_insertion_point(class_scope:svpb.HitLogDebug)
  })
_sym_db.RegisterMessage(HitLogDebug)
_sym_db.RegisterMessage(HitLogDebug.ParametersEntry)

_TALK = DESCRIPTOR.services_by_name['Talk']
if _descriptor._USE_C_DESCRIPTORS == False:

  DESCRIPTOR._options = None
  _TALKREQUEST_ENVINFOENTRY._options = None
  _TALKREQUEST_ENVINFOENTRY._serialized_options = b'8\001'
  _ACTIONPARAM_PARAMSENTRY._options = None
  _ACTIONPARAM_PARAMSENTRY._serialized_options = b'8\001'
  _ACTIONPARAM_EXTRAPARAMSENTRY._options = None
  _ACTIONPARAM_EXTRAPARAMSENTRY._serialized_options = b'8\001'
  _HITLOGDEBUG_PARAMETERSENTRY._options = None
  _HITLOGDEBUG_PARAMETERSENTRY._serialized_options = b'8\001'
  _EVENTTYPE._serialized_start=2342
  _EVENTTYPE._serialized_end=2430
  _TALKREQUEST._serialized_start=51
  _TALKREQUEST._serialized_end=527
  _TALKREQUEST_ENVINFOENTRY._serialized_start=481
  _TALKREQUEST_ENVINFOENTRY._serialized_end=527
  _TALKRESPONSE._serialized_start=530
  _TALKRESPONSE._serialized_end=1095
  _EMOTION._serialized_start=1097
  _EMOTION._serialized_end=1135
  _ASR._serialized_start=1137
  _ASR._serialized_end=1170
  _ANSITEM._serialized_start=1173
  _ANSITEM._serialized_end=1306
  _ACTION._serialized_start=1308
  _ACTION._serialized_end=1381
  _ACTIONPARAM._serialized_start=1384
  _ACTIONPARAM._serialized_end=1874
  _ACTIONPARAM_PARAMSENTRY._serialized_start=1777
  _ACTIONPARAM_PARAMSENTRY._serialized_end=1822
  _ACTIONPARAM_EXTRAPARAMSENTRY._serialized_start=1824
  _ACTIONPARAM_EXTRAPARAMSENTRY._serialized_end=1874
  _TREE._serialized_start=1876
  _TREE._serialized_end=1942
  _SUBTREEITEM._serialized_start=1944
  _SUBTREEITEM._serialized_end=1990
  _HITLOGDEBUG._serialized_start=1993
  _HITLOGDEBUG._serialized_end=2340
  _HITLOGDEBUG_PARAMETERSENTRY._serialized_start=2291
  _HITLOGDEBUG_PARAMETERSENTRY._serialized_end=2340
  _TALK._serialized_start=2432
  _TALK._serialized_end=2549
# @@protoc_insertion_point(module_scope)
