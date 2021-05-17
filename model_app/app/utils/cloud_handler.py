import os
from google.cloud import storage

from app.utils.logger import logger

def get_model(bucket_name: str, source_file_name: str, env: str, ext="model"):
    """Chooses how to upload a model

    Args:
        bucket_name: gcp bucket name
        source_file_name: name of a model
        env: GCP or local
        ext: extension type

    Returns:
        file: file name
    """
    if env == "GCP":
        file = get_gcp_blob_content(bucket_name, source_file_name, ext)
    else:
        file = f"/tmp/{source_file_name}"
    return file


def get_gcp_blob_content(bucket_name, source_file_name, ext="model"):
    """Downloads a model from bucket and returns its name

    Args:
        bucket_name: gcp bucket name
        source_file_name: name of a model
        ext: extension type

    Returns:
        file: file name
    """
    file = f"/tmp/{source_file_name}"
    if os.path.exists(file):
        return file

    storage_client = storage.Client()
    bucket = storage_client.get_bucket(bucket_name)
    blob = bucket.blob(source_file_name)
    if ext in ["txt", "html", "xml", "htm", "xht"]:
        file = blob.download_as_string().decode("utf-8")
    else:
        with open(file, "wb") as ff:
            blob.download_to_file(ff)
        logger.info(f"Downloaded model into {file}")
    return file
