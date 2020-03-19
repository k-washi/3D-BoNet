import boto3

import argparse

parser = argparse.ArgumentParser()
parser.add_argument('-profile', default='qbs', help = 'pfoile name')
parser.add_argument('-region', default='ap-northeast-1', help = 'region name')
parser.add_argument('-i', default='bonet-train' ,help='backet name')
parser.add_argument('-o', help='path to output directory')

parser.add_argument('-f', default='')


s3_client = boto3.Session(profile_name='qbs', region_name='ap-northeast-1').client('s3')
response = s3_client.get_object(Bucket='bonet-train', Key='area_1.h5')

