import os
from datetime import datetime


def generate_save_path():
    abspath_dir_name = os.path.dirname(os.path.abspath(__file__))
    date_time = datetime.now()
    dir_name = date_time.strftime('%d%m%Y%H%M%S')
    path = os.path.abspath(
        os.path.join(abspath_dir_name, os.pardir, os.pardir, "executions",
                     dir_name))
    if not os.path.isdir(path):
        os.makedirs(path)
    return path
