import json


def encode_dict(d):
    """
    把dict给序列化为流，可以直接send
    """
    string_d = json.dumps(d, indent=4)
    return string_d.encode()


def decode_dict(msg):
    """
    接收一个流，解码为dict
    """
    data = json.loads(msg.decode())
    return data