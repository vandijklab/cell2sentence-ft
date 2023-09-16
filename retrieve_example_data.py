import gdown

from src import utils

FILE_ID = "1PYUM59fKclw-aeN79oL5ghCkU4kn6XvN"


def download():
    """Download example scRNA-seq data from Google Drive."""
    download_url = f"https://drive.google.com/uc?id={FILE_ID}"
    output_filepath = utils.DATA_DIR / "dominguez_sample.h5ad"
    gdown.download(download_url, str(output_filepath), quiet=True)


if __name__ == "__main__":
    download()
