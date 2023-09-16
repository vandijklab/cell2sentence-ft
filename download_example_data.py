import gdown

from src import settings

FILE_ID = "1PYUM59fKclw-aeN79oL5ghCkU4kn6XvN"


def download():
    """Download example scRNA-seq data from Google Drive."""
    download_url = f"https://drive.google.com/uc?id={FILE_ID}"
    output_filepath = settings.DATA_DIR / "example_data.h5ad"
    gdown.download(download_url, output_filepath, quiet=True)


if __name__ == "__main__":
    download()
