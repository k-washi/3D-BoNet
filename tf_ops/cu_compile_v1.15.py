"""
Operation作成に伴うcuda, cppのコンパイル コマンドの出力
https://www.tensorflow.org/guide/create_op

---example------------

!/usr/local/cuda-10.1/bin/nvcc 3D-BoNet/tf_ops/../tf_ops/grouping/tf_grouping_g.cu -o 3D-BoNet/tf_ops/../tf_ops/grouping/tf_grouping_g.cu.o -c -O2 -DGOOGLE_CUDA=1 -x cu -Xcompiler -fPIC
!/usr/local/cuda-10.1/bin/nvcc 3D-BoNet/tf_ops/../tf_ops/sampling/tf_sampling_g.cu -o 3D-BoNet/tf_ops/../tf_ops/sampling/tf_sampling_g.cu.o -c -O2 -DGOOGLE_CUDA=1 -x cu -Xcompiler -fPIC
!g++ -std=c++11 3D-BoNet/tf_ops/../tf_ops/grouping/tf_grouping.cpp 3D-BoNet/tf_ops/../tf_ops/grouping/tf_grouping_g.cu.o -o 3D-BoNet/tf_ops/../tf_ops/grouping/tf_grouping_so.so -shared -fPIC -I /usr/local/lib/python3.6/dist-packages/tensorflow_core/include -I /usr/local/cuda-10.1/include -I /usr/local/lib/python3.6/dist-packages/tensorflow_core/include/external/nsync/public -L /usr/local/lib/python3.6/dist-packages/tensorflow_core -l:libtensorflow_framework.so.1 -lcudart -lcuda -L /usr/local/cuda-10.1/lib64 -O2 -D_GLIBCXX_USE_CXX11_ABI=0
!g++ -std=c++11 3D-BoNet/tf_ops/../tf_ops/sampling/tf_sampling.cpp 3D-BoNet/tf_ops/../tf_ops/sampling/tf_sampling_g.cu.o -o 3D-BoNet/tf_ops/../tf_ops/sampling/tf_sampling_so.so -shared -fPIC -I /usr/local/lib/python3.6/dist-packages/tensorflow_core/include -I /usr/local/cuda-10.1/include -I /usr/local/lib/python3.6/dist-packages/tensorflow_core/include/external/nsync/public -L /usr/local/lib/python3.6/dist-packages/tensorflow_core -l:libtensorflow_framework.so.1 -lcudart -lcuda -L /usr/local/cuda-10.1/lib64 -O2 -D_GLIBCXX_USE_CXX11_ABI=0
!g++ -std=c++11 3D-BoNet/tf_ops/../tf_ops/interpolation/tf_interpolate.cpp -o 3D-BoNet/tf_ops/../tf_ops/interpolation/tf_interpolate_so.so -shared -fPIC -I /usr/local/lib/python3.6/dist-packages/tensorflow_core/include -I /usr/local/cuda-10.1/include -I /usr/local/lib/python3.6/dist-packages/tensorflow_core/include/external/nsync/public -L /usr/local/lib/python3.6/dist-packages/tensorflow_core -l:libtensorflow_framework.so.1 -lcudart -lcuda -L /usr/local/cuda-10.1/lib64 -O2 -D_GLIBCXX_USE_CXX11_ABI=0

cflag: ref
https://github.com/lsds/KungFu/commit/a7faaf326163122d3890a725427c61179651d6d5
"""
import os
import sys

import tensorflow as tf
import subprocess
import argparse

BASE_DIR = os.path.join(os.path.dirname(__file__), '..')
sys.path.append(BASE_DIR)

# from utils.logger import set_logger
# import utils.config_ini as config

# logger = set_logger(__name__)

parser = argparse.ArgumentParser()
parser.add_argument('--cuda', help='path to cuda directory', default='/usr/local/cuda-10.1')
parser.add_argument('--googleColab', action='store_true')
args = parser.parse_args()

cuda_path = args.cuda
google_colab = args.googleColab


# compileオプション
cflag = tf.sysconfig.get_compile_flags()
TF_CFLAGS = cflag[0]
TF_CFLAGS_D = cflag[1]
TF_NSYNC = TF_CFLAGS + "/external/nsync/public"

# リンクディレクトリ
clink = tf.sysconfig.get_link_flags()
TF_LFLAGS = clink[0] + " " + clink[1]

GROUPING_PATH = BASE_DIR + '/tf_ops/grouping'
INTERPOLATION_PATH = BASE_DIR + '/tf_ops/interpolation'
SAMPLING_PATH = BASE_DIR + '/tf_ops/sampling'

NVCC_CMD = os.path.join(cuda_path, 'bin', 'nvcc')
CUDA_CFLAG = '-I ' + os.path.join(cuda_path, 'include')
CUDA_LFLAG = '-lcudart -L ' + os.path.join(cuda_path, 'lib64')


def confirm_dir(path):
    if os.path.isdir(path):
        return
    raise Exception("{} がありません".format(path))


def confirm_file(path):
    if os.path.isfile(path):
        logger.debug("OK: {}".format(path))
        return
    raise Exception("{} がありません".format(path))


confirm_dir(GROUPING_PATH)
confirm_dir(INTERPOLATION_PATH)
confirm_dir(SAMPLING_PATH)


# confirm_file(config.CUDA_PATH)


def nvcc_grouping():
    return NVCC_CMD + ' ' + os.path.join(GROUPING_PATH, 'tf_grouping_g.cu') + ' -o ' \
           + os.path.join(GROUPING_PATH, 'tf_grouping_g.cu.o') \
           + ' -c -O2 -DGOOGLE_CUDA=1 -x cu -Xcompiler -fPIC'


def nvcc_sampling():
    return NVCC_CMD + ' ' + os.path.join(SAMPLING_PATH, 'tf_sampling_g.cu') + ' -o ' \
           + os.path.join(SAMPLING_PATH, 'tf_sampling_g.cu.o') \
           + ' -c -O2 -DGOOGLE_CUDA=1 -x cu -Xcompiler -fPIC'


def cpp_sampling():
    return 'g++ -std=c++11 ' + os.path.join(SAMPLING_PATH, 'tf_sampling.cpp') \
           + ' ' + os.path.join(SAMPLING_PATH, 'tf_sampling_g.cu.o') + ' -o ' \
           + os.path.join(SAMPLING_PATH, 'tf_sampling_so.so') + ' -shared -fPIC ' \
           + TF_CFLAGS + ' ' + TF_CFLAGS_D + ' ' + CUDA_CFLAG + ' ' + TF_NSYNC + ' ' + TF_LFLAGS + ' ' + CUDA_LFLAG + ' -O2'


def cpp_interpolation():
    return 'g++ -std=c++11 ' + os.path.join(INTERPOLATION_PATH, 'tf_interpolate.cpp') + ' -o ' \
           + os.path.join(INTERPOLATION_PATH, 'tf_interpolate_so.so') + ' -shared -fPIC ' \
           + TF_CFLAGS + ' ' + TF_CFLAGS_D + ' ' + CUDA_CFLAG + ' ' + TF_NSYNC + ' ' + TF_LFLAGS + ' ' + CUDA_LFLAG + ' -O2'


def cpp_grouping():
    return 'g++ -std=c++11 ' + os.path.join(GROUPING_PATH, 'tf_grouping.cpp') \
           + ' ' + os.path.join(GROUPING_PATH, 'tf_grouping_g.cu.o') + ' -o ' \
           + os.path.join(GROUPING_PATH, 'tf_grouping_so.so') + ' -shared -fPIC ' \
           + TF_CFLAGS + ' ' + TF_CFLAGS_D + ' ' + CUDA_CFLAG + ' ' + TF_NSYNC + ' ' + TF_LFLAGS + ' ' + CUDA_LFLAG + ' -O2'


def nvcc_proc_call(gcolab=False):
    gr = nvcc_grouping()
    sm = nvcc_sampling()
    if gcolab is False:
        print(gr)
        print(sm)
    else:
        print("!" + gr)
        print("!" + sm)


def cpp_proc_call(gcolab=False):
    gr = cpp_grouping()
    sm = cpp_sampling()
    inter = cpp_interpolation()
    if gcolab is False:
        print(gr)
        print(sm)
        print(inter)
    else:
        print("!" + gr)
        print("!" + sm)
        print("!" + inter)


if __name__ == "__main__":
    print("-"*30)
    print(" ")
    nvcc_proc_call(gcolab=google_colab)
    cpp_proc_call(gcolab=google_colab)
