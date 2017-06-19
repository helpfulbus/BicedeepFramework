from __future__ import absolute_import
from google.cloud import storage
import os


BUCKET_NAME = "elated-discovery-5100"
def _get_storage_client():

    #credentials = GoogleCredentials.get_application_default()
    return storage.Client(project="dibioo-frontend")


def download_file(file_name):

    client = _get_storage_client()
    bucket = client.bucket(BUCKET_NAME)
    #bb = list(bucket.list_blobs())[3]

    blob = bucket.get_blob(file_name)
    file_name_long = BUCKET_NAME + "/" + file_name

    dir = os.path.dirname(file_name_long)
    if not os.path.exists(dir):
        os.system("sudo mkdir " + dir)
    if not os.path.exists(dir):
        os.makedirs(dir)

    file_name_short = file_name.split('/')[-1]
    file = open(file_name_long, 'w')
    blob.download_to_filename(file_name_long, client)
