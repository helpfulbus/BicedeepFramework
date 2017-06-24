from __future__ import absolute_import
from google.cloud import storage
import os
import config


def _get_storage_client():
    return storage.Client(project=config.PROJECT_NAME)


def download_file(file_name):
    client = _get_storage_client()
    bucket = client.bucket(config.BUCKET_NAME)

    blob = bucket.get_blob(file_name)
    file_name_long = config.BUCKET_NAME + "/" + file_name

    dir = os.path.dirname(file_name_long)
    if not os.path.exists(dir):
        os.system("sudo mkdir " + dir)
    if not os.path.exists(dir):
        os.makedirs(dir)

    file = open(file_name_long, 'w')
    blob.download_to_filename(file_name_long, client)

    return file_name_long
