from google.cloud import storage
import os

def upload_blob(bucket_name, source_file_name, destination_folder):

    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(destination_folder)

    for root, dirs, files in os.walk(source_file_name):
        for file_name in files:
            local_path = os.path.join(root, file_name)
            relative_path = os.path.relpath(local_path, source_file_name)
            blob_path = os.path.join(destination_folder, relative_path)

            blob = bucket.blob(blob_path)
            blob.upload_from_filename(local_path)
            print(f"Uploaded {local_path} to {blob_path}")

upload_blob('igpt_fine_tuned_models', 'IGPT/fine_tune/saved_model_lora', 'llama-7b-emily')