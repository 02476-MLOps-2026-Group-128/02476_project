import os
from pathlib import Path

import typer
from google.cloud import storage
from loguru import logger


def clear_user_inputs() -> None:
    """Clear user-provided rows, that are saved as JSON files in the bucket."""
    storage_client = storage.Client()
    bucket_name = os.environ.get("DATA_STORAGE_BUCKET_NAME")
    if bucket_name:
        bucket = storage_client.bucket(bucket_name)
        prefix = "enriched/new_rows/"
        blobs = bucket.list_blobs(prefix=prefix)

        nb_blobs = 0
        for new_row_blob in blobs:
            new_row_blob.delete()
            nb_blobs += 1

        logger.info(f"Cleared {nb_blobs} user-provided rows from gs://{bucket_name}/{prefix}.")

    else:
        logger.warning("Env var DATA_STORAGE_BUCKET_NAME not set. Skipping enrichment of dataset.")
        return


if __name__ == "__main__":
    typer.run(clear_user_inputs)
