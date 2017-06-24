import boto3
import config
import json
from collections import namedtuple


def send_report_completed_message(message_body):
    sqs = boto3.resource('sqs',
                         aws_access_key_id=config.AWS_ACCESS_KEY_ID,
                         aws_secret_access_key=config.AWS_SECRET_ACCESS_KEY,
                         region_name=config.AWS_REGION)

    # Get the queue
    queue = sqs.get_queue_by_name(QueueName=config.EMAIL_QUEUE_NAME)

    response = queue.send_messages(
        Entries=[
            {
                'Id': '1',
                'MessageBody': message_body,
                'MessageGroupId': 'ReportCompleted'
            },
        ]
    )

    return response


def read_report_queue():
    sqs = boto3.resource('sqs',
                         aws_access_key_id=config.AWS_ACCESS_KEY_ID,
                         aws_secret_access_key=config.AWS_SECRET_ACCESS_KEY,
                         region_name=config.AWS_REGION)

    queue = sqs.get_queue_by_name(QueueName=config.REPORT_QUEUE_NAME)

    message_list = queue.receive_messages(MessageAttributeNames=['MessageGroupId'],
                                          MaxNumberOfMessages=1,
                                          WaitTimeSeconds=3)
    if len(message_list) > 0:
        return message_list[0]
    else:
        return None


def deserialize_message(message):
    deserialized_message = json.loads(message.body, object_hook=lambda d: namedtuple('X', d.keys())(*d.values()))
    return deserialized_message.fileName, deserialized_message.email


def stop_instance(instance_id):
    ec2 = boto3.resource('ec2',
                         aws_access_key_id=config.AWS_ACCESS_KEY_ID,
                         aws_secret_access_key=config.AWS_SECRET_ACCESS_KEY,
                         region_name=config.AWS_REGION)

    instance = ec2.Instance(instance_id)

    return instance.stop()
