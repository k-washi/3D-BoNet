import configparser
import os


def get_connfigure(path):
    config = configparser.ConfigParser()
    if not os.path.isfile(path):
        raise Exception('設定ファイルのパスが間違っている')

    config.read(path)

    config_dic = {}

    sec1 = 'train'
    ids = config.get(sec1, 'IDs')
    config_dic['train'] = [int(i) for i in ids.split(',')]

    sec2 = 'eval'
    ids = config.get(sec2, 'IDs')
    config_dic['eval'] = [int(i) for i in ids.split(',')]

    return config_dic

if __name__ == '__main__':
    path = './config/config.ini'

    print(get_connfigure(path))