import os

from azure.storage.blob import BlobServiceClient
from magic_pdf.config.enums import SupportedPdfParseMethod
from magic_pdf.data.dataset import PymuDocDataset
from magic_pdf.model.doc_analyze_by_custom_model import doc_analyze
from dotenv import load_dotenv
load_dotenv()
# Helper classes
class AzureBlobDataReader:
    def __init__(self, prefix, container_name, connection_string):
        self.prefix = prefix  # e.g., "unittest/tmp/"
        self.container_name = container_name
        self.connection_string = connection_string
        self.blob_service_client = BlobServiceClient.from_connection_string(connection_string)
        self.container_client = self.blob_service_client.get_container_client(container_name)

    def read(self, azure_blob_url):
        """
        Expects azure_blob_url in the format:
        "azure://{container_name}/{path/to/file.pdf}"
        """
        if not azure_blob_url.startswith("azure://"):
            raise ValueError("Azure blob url must start with 'azure://'")

        path = azure_blob_url[len("azure://"):]
        parts = path.split("/", 1)
        if len(parts) != 2:
            raise ValueError("Invalid azure blob url. Expected format: azure://<container>/<blob_path>")
        container, blob_path = parts

        # Get the blob client from the container client
        blob_client = self.container_client.get_blob_client(blob_path)
        downloader = blob_client.download_blob()
        data = downloader.readall()
        return data


class AzureBlobDataWriter:
    def __init__(self, prefix, container_name, connection_string):
        self.prefix = prefix  # e.g., "unittest/tmp" or "unittest/tmp/images"
        self.container_name = container_name
        self.connection_string = connection_string
        self.blob_service_client = BlobServiceClient.from_connection_string(connection_string)
        self.container_client = self.blob_service_client.get_container_client(container_name)

    def write(self, blob_name, data):
        """
        Upload data to a blob. The final blob path will be a combination of the prefix and blob_name.
        """
        full_blob_name = os.path.join(self.prefix, blob_name)
        blob_client = self.container_client.get_blob_client(full_blob_name)
        blob_client.upload_blob(data, overwrite=True)
        print(f"Uploaded {full_blob_name} to Azure Blob Storage")


#storage setup
container_name = os.getenv("AZURE_CONTAINER")
connection_string = os.getenv("AZURE_CONNECTION_STRING")
azure_path=os.getenv("HOME_DIR")
# Create reader and writers
reader = AzureBlobDataReader(azure_path, container_name, connection_string)
writer = AzureBlobDataWriter(azure_path, container_name, connection_string)
image_writer = AzureBlobDataWriter(f'{azure_path}/images', container_name, connection_string)
md_writer = AzureBlobDataWriter(azure_path, container_name, connection_string)

local_image_dir, local_md_dir = "output/images", "output"
image_dir = os.path.basename(local_image_dir)

# -----------------------------------------------------------------------------
# Processing Pipeline
# -----------------------------------------------------------------------------

# Path to the PDF in Azure Blob Storage. Use the format "azure://{container}/{blob_path}"
pdf_file_name = f"azure://{container_name}/{azure_path}bug5-11.pdf"

# Prepare local output paths and names
local_dir = "output"
name_without_suff = os.path.basename(pdf_file_name).split(".")[0]

pdf_bytes = reader.read(pdf_file_name)  # read the pdf content

ds = PymuDocDataset(pdf_bytes)

# choose OCR mode or text mode based on PDF classification
if ds.classify() == SupportedPdfParseMethod.OCR:
    infer_result = ds.apply(doc_analyze, ocr=True)
    # Pipeline for OCR mode: write images via image_writer
    pipe_result = infer_result.pipe_ocr_mode(image_writer)
else:
    infer_result = ds.apply(doc_analyze, ocr=False)
    # The text mode.
    pipe_result = infer_result.pipe_txt_mode(image_writer)

# Draw model result on each page locally
infer_result.draw_model(os.path.join(local_md_dir, f"{name_without_suff}_model.pdf"))

# Get model inference result (if needed)
model_inference_result = infer_result.get_infer_res()

# Draw layout and spans results on each page locally
pipe_result.draw_layout(os.path.join(local_md_dir, f"{name_without_suff}_layout.pdf"))
pipe_result.draw_span(os.path.join(local_md_dir, f"{name_without_suff}_spans.pdf"))

# Dump markdown and content list to Azure Blob Storage
pipe_result.dump_md(md_writer, f"{name_without_suff}.md", image_dir)
pipe_result.dump_content_list(md_writer, f"{name_without_suff}_content_list.json", image_dir)

# Retrieve markdown content and content list (if needed)
md_content = pipe_result.get_markdown(image_dir)
content_list_content = pipe_result.get_content_list(image_dir)

# Get and dump the middle JSON representation
middle_json_content = pipe_result.get_middle_json()
pipe_result.dump_middle_json(md_writer, f'{name_without_suff}_middle.json')
