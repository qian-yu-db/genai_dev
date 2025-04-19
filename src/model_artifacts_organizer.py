from mlflow.tracking import MlflowClient
import json
import os
from databricks.sdk.errors import PermissionDenied
import mlflow
from json2html import json2html
from markdownify import markdownify as md
import nbformat
from nbconvert import MarkdownExporter
import logging

logging.basicConfig()
logger = logging.getLogger("model_artifact_organizer")


class ModelArtifactOrganizer:
    def __init__(self, catalog: str, schema: str, model: str):
        self.model = model
        self.volume_path = f"/Volumes/{catalog}/{schema}/ml_documents"
        self.model_full_name = f"{catalog}.{schema}.{model}"
        self.client = MlflowClient()
        mlflow.set_registry_uri("databricks-uc")

    def collect_mlflow_artifacts(self) -> str | bytes:
        model_version = self.client.get_model_version_by_alias(
            name=self.model_full_name, alias="production"
        )
        run_id = model_version.run_id
        destination_path = self.create_uc_artifact_folder(
            volume_path=self.volume_path, folder_name=self.model
        )
        mlflow.artifacts.download_artifacts(
            artifact_uri=f"runs:/{run_id}/", dst_path=destination_path
        )
        return destination_path

    @staticmethod
    def create_uc_artifact_folder(volume_path, folder_name) -> str | bytes:
        folder_path = os.path.join(volume_path, folder_name)
        logger.info(f"Creating folder: {folder_path}")

        # Check if the folder exists and create it if it does not
        if not os.path.isdir(folder_path):
            try:
                os.makedirs(folder_path)
                logger.info(f"PASS: Folder `{folder_path}` created")
            except PermissionDenied:
                logger.info(f"FAIL: No permissions to create folder `{folder_path}`")
                raise ValueError(f"No permissions to create folder `{folder_path}`")
        else:
            logger.info(f"PASS: Folder `{folder_path}` already exists")
        return folder_path

    def create_model_attributes_md(self) -> str:
        dst_path = f"{self.volume_path}/{self.model}"
        model_version = self.client.get_model_version_by_alias(
            name=self.model_full_name, alias="production"
        )
        run_id = model_version.run_id
        run = self.client.get_run(run_id)

        # model properties
        model_flattened_json = json.dumps(
            {**run.data.params, **run.data.metrics, **run.data.tags}, indent=4
        )
        model_html = json2html.convert(json.loads(model_flattened_json))

        # run properties
        run_info_dict = {key: value for key, value in run.info.__dict__.items()}
        run_info_html = json2html.convert(run_info_dict)

        # data source
        dataset_input = run.inputs.to_dictionary()["dataset_inputs"][0]
        data_source = dataset_input["dataset"]
        data_source_html = json2html.convert(data_source)

        # consolidated information
        consolidated_md = (
            f"# model algorithm, model parameters, and model metrics table\n"
            f"{md(model_html)}\n"
            f"# model run information table\n"
            f"{md(run_info_html)}\n"
            f"# data source table\n"
            f"{md(data_source_html)}"
        )

        with open(f"{dst_path}/model_attribute_tables.md", "w") as file:
            logger.info(
                f"writing model attributes tables to {dst_path}/model_attribute_tables.md"
            )
            file.write(consolidated_md)

        return f"{dst_path}/model_attribute_tables.md"

    def image_file_to_md(self) -> str:
        folder_path = f"{self.volume_path}/{self.model}"
        png_files = [f for f in os.listdir(folder_path) if f.endswith(".png")]
        markdown_content = "\n\n".join(
            [
                f"![{os.path.splitext(f)[0]}]({os.path.join(folder_path, f)})"
                for f in png_files
            ]
        )
        output_file = f"{folder_path}/visuals.md"

        with open(output_file, "w") as md_file:
            md_file.write(markdown_content)

        return output_file

    def notebook_to_md(self) -> None:
        notebook_path = f"{self.volume_path}/{self.model}/notebooks"
        output_path = f"{self.volume_path}{self.model}/"

        files = [
            f
            for f in os.listdir(notebook_path)
            if f.endswith(".ipynb") or f.endswith(".py")
        ]
        for f in files:
            if f.endswith(".ipynb"):
                with open(f"{notebook_path}/{f}", "r", encoding="utf-8") as nb_file:
                    notebook_content = nbformat.read(nb_file, as_version=4)
            else:
                with open(f"{notebook_path}/{f}", "r", encoding="utf-8") as py_file:
                    notebook_content = py_file.read()

            # Convert to markdown
            markdown_exporter = MarkdownExporter()
            markdown_content, _ = markdown_exporter.from_notebook_node(notebook_content)

            # Write to markdown file
            output_file = f"{output_path}/{f.replace('.ipynb', '.md')}"
            with open(output_file, "w", encoding="utf-8") as md_file:
                md_file.write(markdown_content)
