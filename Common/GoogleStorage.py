# <copyright company="Bicedeep, Inc.">
# Copyright (c) 2016-2018 All Rights Reserved
# </copyright>

from __future__ import absolute_import
from google.cloud import storage
import os
import config

reports_folder = "/reports/"
outputs_folder = "/outputs/"
logs_folder = "/logs/"

def _get_storage_client():
    return storage.Client(project=config.PROJECT_NAME)

def get_model_details_name_by_model_name(model_file_name):
    splitted_name = model_file_name.split("/")
    return splitted_name[0] + "/details/" + splitted_name[2] + ".json"

def download_query_file(file_name, model_file_name):
    client = _get_storage_client()
    bucket = client.bucket(config.BUCKET_NAME)

    file_name_long = config.BUCKET_NAME + "/" + file_name

    dir = os.path.dirname(file_name_long)
    if not os.path.exists(dir):
        os.system("sudo mkdir " + dir)
    if not os.path.exists(dir):
        os.makedirs(dir)


    model_name_long = config.BUCKET_NAME + "/" + model_file_name

    dir = os.path.dirname(model_name_long)
    if not os.path.exists(dir):
        os.system("sudo mkdir " + dir)
    if not os.path.exists(dir):
        os.makedirs(dir)

    model_details_name = get_model_details_name_by_model_name(model_file_name)
    model_details_long = config.BUCKET_NAME + "/" + model_details_name

    dir = os.path.dirname(model_details_long)
    if not os.path.exists(dir):
        os.system("sudo mkdir " + dir)
    if not os.path.exists(dir):
        os.makedirs(dir)


    query_blob = bucket.get_blob(file_name)
    model_blob = bucket.get_blob(model_file_name)
    model_details_blob = bucket.get_blob(model_details_name)

    query_file = open(file_name_long, 'w')
    model_file = open(model_name_long, 'w')
    model_details_file = open(model_details_long, 'w')

    query_blob.download_to_filename(file_name_long, client)
    model_blob.download_to_filename(model_name_long, client)
    model_details_blob.download_to_filename(model_details_long, client)

    return file_name_long, model_name_long, model_details_long


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


def upload_results(email, reported_file_name):
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

    for file_name in os.listdir('.'):
        if(file_name.split('.')[-1] == 'log'):
            dest_name = email + logs_folder + reported_file_name.split('/')[-1] + "/" + file_name
            file_name_full = "./" + file_name
            blob = bucket.blob(dest_name)
            blob.upload_from_filename(file_name_full)
            open(file_name, 'w').close()

def upload_query_results(query_file_name):
    client = _get_storage_client()
    bucket = client.bucket(config.BUCKET_NAME)

    blob = bucket.blob(query_file_name)
    blob.upload_from_filename(config.BUCKET_NAME + "/" + query_file_name)

    query_file_name_split = query_file_name.split('/')
    email = query_file_name_split[0]
    file_name = "queueserver.log"
    dest_name = email + logs_folder + query_file_name_split[-1] + "/" + file_name
    file_name_full = "./" + file_name
    blob = bucket.blob(dest_name)
    blob.upload_from_filename(file_name_full)
    open(file_name, 'w').close()

def delete_local_dir():
    os.system("sudo rm -rf " + config.BUCKET_NAME)
