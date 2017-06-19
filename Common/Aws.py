import boto3
import datetime

def send_report_completed_message(file_name):
    sqs = boto3.resource('sqs')

    # Get the queue
    queue = sqs.get_queue_by_name(QueueName='BiceQueue1.fifo')

    response = queue.send_messages(
        Entries=[
            {
                'Id': '1',
                'MessageBody': file_name,
                'MessageGroupId': 'ReportCompleted'
            },
        ]
    )

    return response

def stop_instance(id):
    ec2 = boto3.resource('ec2')
    instance = ec2.Instance(id)

    return instance.stop()

