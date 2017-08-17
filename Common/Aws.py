# <copyright company="Bicedeep, Inc.">
# Copyright (c) 2016-2018 All Rights Reserved
# </copyright>

import boto3
import config
import json
from collections import namedtuple
from Common import Logging


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

def try_to_delete_message(message, queueName):
    try:
        message.delete()
        Logging.Logging.write_log_to_file("Message Deleted")
    except Exception as e:
        delete_message_count = 10
        while(delete_message_count > 0):
            try:
                message_re_read = read_queue(queueName)
                if(message.body == message_re_read.body):
                    message_re_read.delete()
                    Logging.Logging.write_log_to_file("Message Deleted")
                return
            except Exception as e:
                delete_message_count = delete_message_count - 1
                Logging.Logging.write_log_to_file(str(e))


def send_exception_message(message_body, queueName):
    sqs = get_boto3_resource('sqs')
    # Get the queue
    queue = sqs.get_queue_by_name(QueueName=queueName)

    response = queue.send_messages(
        Entries=[
            {
                'Id': '1',
                'MessageBody': message_body,
                'MessageGroupId': 'ReportException'
            },
        ]
    )

    return response

def read_queue(queueName):
    sqs = get_boto3_resource('sqs')
    queue = sqs.get_queue_by_name(QueueName=queueName)

    message_list = queue.receive_messages(MessageAttributeNames=['MessageGroupId'],
                                          MaxNumberOfMessages=1,
                                          WaitTimeSeconds=1)
    if len(message_list) > 0:
        return message_list[0]
    else:
        return None


def add_message_to_exception_queue(message, sourceQueueName, exceptionQueueName):
    try:
        send_exception_message(message.body, exceptionQueueName)
    except Exception as e:
        Logging.Logging.write_log_to_file(str(e))
    try_to_delete_message(message, sourceQueueName)


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
