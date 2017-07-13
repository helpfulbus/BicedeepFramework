from __future__ import absolute_import
from google.cloud import storage
import os
import config

reports_folder = "/reports/"
outputs_folder = "/outputs/"

def _get_storage_client():
    return storage.Client(project=config.PROJECT_NAME)


def download_file(file_name, email):
    client = _get_storage_client()
    bucket = client.bucket(config.BUCKET_NAME)

    blob = bucket.get_blob(file_name)
    file_name_long = config.BUCKET_NAME + "/" + file_name

    dir = os.path.dirname(file_name_long)
    if not os.path.exists(dir):
        os.system("sudo mkdir " + dir)
    if not os.path.exists(dir):
        os.makedirs(dir)

    reports_path_dir = os.path.dirname(config.BUCKET_NAME + "/" + email + reports_folder)
    outputs_path_dir = os.path.dirname(config.BUCKET_NAME + "/" + email + outputs_folder)

    if not os.path.exists(reports_path_dir):
        os.system("sudo mkdir -m 770 " + reports_path_dir)
    if not os.path.exists(reports_path_dir):
        os.makedirs(reports_path_dir)

    if not os.path.exists(outputs_path_dir):
        os.system("sudo mkdir -m 770 " + outputs_path_dir)
    if not os.path.exists(outputs_path_dir):
        os.makedirs(outputs_path_dir)

    file = open(file_name_long, 'w')
    blob.download_to_filename(file_name_long, client)

    return file_name_long, reports_path_dir, outputs_path_dir


def upload_results(email):
    client = _get_storage_client()
    bucket = client.bucket(config.BUCKET_NAME)

    reports_path_dir = os.path.dirname(config.BUCKET_NAME + "/" + email + reports_folder)
    outputs_path_dir = os.path.dirname(config.BUCKET_NAME + "/" + email + outputs_folder)

    for file_name in os.listdir(outputs_path_dir):
        dest_name = email + outputs_folder + file_name
        file_name_full = outputs_path_dir + "/" + file_name
        blob = bucket.blob(dest_name)
        blob.upload_from_filename(file_name_full)

    for file_name in os.listdir(reports_path_dir):
        dest_name = email + reports_folder + file_name
        file_name_full = reports_path_dir + "/" + file_name
        blob = bucket.blob(dest_name)
        blob.upload_from_filename(file_name_full)


def delete_local_dir():
    os.system("sudo rm -rf " + config.BUCKET_NAME)
