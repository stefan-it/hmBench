import flair

from flair.datasets.sequence_labeling import ColumnCorpus
from flair.file_utils import cached_path

from pathlib import Path
from typing import Optional, Union


class HisGermaNER(ColumnCorpus):
    def __init__(
            self,
            base_path: Optional[Union[str, Path]] = None,
            in_memory: bool = True,
            **corpusargs,
    ) -> None:
        base_path = flair.cache_root / "datasets" if not base_path else Path(base_path)
        dataset_name = self.__class__.__name__.lower()
        data_folder = base_path / dataset_name
        data_path = flair.cache_root / "datasets" / dataset_name

        column_format = {0: "text", 1: "ner"}

        for split in ["train", "dev", "test"]:
            data_file = data_path / f"HisGermaNER_v0_{split}.tsv"

            if not data_file.is_file():
                cached_path(
                    f"https://huggingface.co/datasets/stefan-it/HisGermaNER/resolve/main/splits/HisGermaNER_v0_{split}.tsv",
                    data_path,
                )

        super().__init__(
            data_folder,
            column_format,
            column_delimiter="\t",
            in_memory=in_memory,
            comment_symbol="# ",
            document_separator_token="-DOCSTART-",
            skip_first_line=True,
            **corpusargs,
        )
