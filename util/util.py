# -*- coding:utf-8 -*-
import re
import uuid


def mock_trace_id():
    return f"{uuid.uuid4()}@cloudminds-test.com.cn"


def check_grpc_url(address: str) -> bool:
    """检查GRPC地址是否合法
    Parameters
    ----------
    address : str
        172.16.23.33:8080

    Returns
    -------
    bool
        True/False
    """
    pattern = re.compile('(\d|[1-9]\d|1\d{2}|2[0-4]\d|25[0-5])\.(\d|[1-9]\d|1\d{2}|2[0-4]\d|25[0-5])\.(\d|[1-9]\d|1\d{2}|2[0-4]\d|25[0-5])\.(\d|[1-9]\d|1\d{2}|2[0-4]\d|25[0-5]):(6[0-5]{2}[0-3][0-5]|[1-5]\d{4}|[1-9]\d{1,3}|[0-9])')
    return True if pattern.match(address) else False


def check_http_url(address: str) -> bool:
    """检查HTTP地址是否合法
        Parameters
        ----------
        address : str
            http://172.16.23.84:32194/nlp-sdk/qqsim

        Returns
        -------
        bool
            True/False
        """
    pattern = re.compile(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+')
    return True if pattern.match(address) else False
