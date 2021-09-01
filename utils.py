import logging
import time
import functools
import json
import yaml

import numpy as np

logger = logging.getLogger(__name__)

def save2txt(data, path):
    data = list(map(lambda x: str(x) + '\n', data))
    with open(path,'w', encoding='utf-8') as f:
        f.writelines(data)

def save2json(data, path):
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(data, f)


def json2dict(json_path):
    with open(json_path, encoding='utf-8') as f:
        d = json.load(f)
    return d

def save2yaml(data, path):
    with open(path, 'w', encoding='utf-8') as f:
        yaml.dump(data, f)


def yaml2dict(path):
    with open(path, encoding='utf-8') as f:
        d = yaml.load(f, Loader=yaml.FullLoader)
    return d

class Timer:
    """
    一个计时器工具
    可以作为装饰器，也可以作为上下文管理器使用
    使用方法
    ```python
    timer = Timer()

    for i in range(N):
        with timer:
            func()
    print(timer.average_time())        
    ```
    """

    def __init__(self):
        self._total_time = []

    def __enter__(self):
        self._begin_time = time.time()

    def __exit__(self, exc_type, exc_val, exc_tb):
        self._end_time = time.time()
        self._total_time.append(self._end_time - self._begin_time)

    def run_num(self):
        return len(self._total_time)

    def this_time(self):
        return self._end_time - self._begin_time

    def average_time(self):
        return sum(self._total_time) / len(self._total_time)

    def total_time(self):
        return sum(self._total_time)

    def all_time(self):
        return self._total_time

    def clear(self):
        self._total_time = []

    @classmethod
    def timer(cls, msg=''):
        """
        一个装饰器计时器
        使用方法如下例所示：
        ```python
        import time

        @Timer.timer("test time is {:.3f}s.")
        def timer_test():
            time.sleep(2)

        timer_test()
        ```
        输出
        ```
        DEBUG:__main__:test time is 2.004170s.
        ```
        """

        msg = 'run time: {:.2f}s.' if msg == '' else msg

        def timer_(func):
            @functools.wraps(func)
            def wrapper(*args, **kargs):
                begin_time = time.time()
                res = func(*args, **kargs)
                end_time = time.time()
                during_time = end_time - begin_time

                logger.debug(msg.format(during_time))

                return res, during_time

            return wrapper
        return timer_
