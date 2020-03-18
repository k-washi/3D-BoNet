import boto3

s3_client = boto3.Session(profile_name='qbs', region_name='ap-northeast-1').client('s3')
response = s3_client.get_object(Bucket='bonet-train', Key='area_0.h5')

