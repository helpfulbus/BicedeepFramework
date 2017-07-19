import boto3
import config
import json
from collections import namedtuple


def send_report_completed_message(message_body):
    sqs = get_boto3_resource('sqs')
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
    sqs = get_boto3_resource('sqs')
    queue = sqs.get_queue_by_name(QueueName=config.REPORT_QUEUE_NAME)

    message_list = queue.receive_messages(MessageAttributeNames=['MessageGroupId'],
                                          MaxNumberOfMessages=1,
                                          WaitTimeSeconds=1)
    if len(message_list) > 0:
        return message_list[0]
    else:
        return None


def read_queue_queue():
    sqs = get_boto3_resource('sqs')
    queue = sqs.get_queue_by_name(QueueName=config.QUEUE_QUEUE_NAME)

    message_list = queue.receive_messages(MessageAttributeNames=['MessageGroupId'],
                                          MaxNumberOfMessages=1,
                                          WaitTimeSeconds=1)
    if len(message_list) > 0:
        return message_list[0]
    else:
        return None


def deserialize_message(message):
    deserialized_message = json.loads(message.body, object_hook=lambda d: namedtuple('X', d.keys())(*d.values()))
    return deserialized_message.email, deserialized_message.fileName, deserialized_message.selectedHeaders

def deserialize_queue_message(message):
    deserialized_message = json.loads(message.body, object_hook=lambda d: namedtuple('X', d.keys())(*d.values()))
    return deserialized_message.fileName, deserialized_message.content


def stop_instance(instance_id):
    ec2 = get_boto3_resource('ec2')
    instance = ec2.Instance(instance_id)

    return instance.stop()

def get_boto3_resource(resource_name):
    return boto3.resource(resource_name,
                          aws_access_key_id=config.AWS_ACCESS_KEY_ID,
                          aws_secret_access_key=config.AWS_SECRET_ACCESS_KEY,
                          region_name=config.AWS_REGION)
