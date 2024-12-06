from google.cloud import storage
import os

def upload_blob(bucket_name, source_file_name, destination_folder):
    """Uploads a file to the bucket."""
    # The ID of your GCS bucket
    # bucket_name = "your-bucket-name"
    # The path to your file to upload
    # source_file_name = "local/path/to/file"
    # The ID of your GCS object
    # destination_blob_name = "storage-object-name"

    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(destination_folder)

    # Optional: set a generation-match precondition to avoid potential race conditions
    # and data corruptions. The request to upload is aborted if the object's
    # generation number does not match your precondition. For a destination
    # object that does not yet exist, set the if_generation_match precondition to 0.
    # If the destination object already exists in your bucket, set instead a
    # generation-match precondition using its generation number.

    for root, dirs, files in os.walk(source_file_name):
        for file_name in files:
            local_path = os.path.join(root, file_name)
            relative_path = os.path.relpath(local_path, source_file_name)
            blob_path = os.path.join(destination_folder, relative_path)

            blob = bucket.blob(blob_path)
            blob.upload_from_filename(local_path)
            print(f"Uploaded {local_path} to {blob_path}")
    
    # generation_match_precondition = 0

    # blob.upload_from_filename(source_file_name, if_generation_match=generation_match_precondition)

    # print(
    #     f"File {source_file_name} uploaded to {destination_blob_name}."
    # )

upload_blob('igpt_fine_tuned_models', 'IGPT/fine_tune/saved_model_lora', 'llama-7b-emily')