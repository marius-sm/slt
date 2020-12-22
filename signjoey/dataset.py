# coding: utf-8
"""
Data module
"""
from torchtext import data
from torchtext.data import Field, RawField
from typing import List, Tuple
import pickle
import gzip
import torch


def load_dataset_file(filename):
    
    print('Loading', filename)

    with gzip.open(filename, 'rb') as f:
        loaded_object = pickle.load(f)
        return loaded_object
    


class SignTranslationDataset(data.Dataset):
    """Defines a dataset for machine translation."""

    @staticmethod
    def sort_key(ex):
        return data.interleave_keys(len(ex.sgn), len(ex.txt))

    def __init__(
        self,
        fields: Tuple[RawField, RawField, Field, Field, Field],
        path: str = None,
        embeddings_path: str = None,
        annotations_path: str = None,
        **kwargs
    ):
        """Create a SignTranslationDataset given paths and fields.

        Arguments:
            path: Common prefix of paths to the data files for both languages.
            embeddings_path: If path is None, file from which to get the embeddings
            annotations_path: If path is None, file from which to get the annotations
            exts: A tuple containing the extension to path for each language.
            fields: A tuple containing the fields that will be used for data
                in each language.
            Remaining keyword arguments: Passed to the constructor of
                data.Dataset.
        """
        if not isinstance(fields[0], (tuple, list)):
            fields = [
                ("sequence", fields[0]),
                ("signer", fields[1]),
                ("sgn", fields[2]),
                ("gls", fields[3]),
                ("txt", fields[4]),
            ]

        if path is not None:
            iterator = [path] if not isinstance(path, list) else path
        else:
            assert embeddings_path is not None
            assert annotations_path is not None
            embeddings_path = [embeddings_path] if not isinstance(embeddings_path, list) else embeddings_path
            annotations_path = [annotations_path] if not isinstance(annotations_path, list) else annotations_path
            iterator = zip(embeddings_path, annotations_path)

        samples = {}
        for obj in iterator:
            if isinstance(obj, tuple):
                embeddings = load_dataset_file(obj[0])
                annotations = load_dataset_file(obj[1])
                for s in embeddings:
                    seq_id = s["name"]
                    if seq_id in samples:
                        assert samples[seq_id]["name"] == s["name"]
                        samples[seq_id]["sign"] = torch.cat(
                            [samples[seq_id]["sign"], s["sign"]], axis=1
                        )
                    else:
                        samples[seq_id] = {
                            "name": s["name"],
                            "sign": s["sign"],
                        }
                for s in annotations:
                    seq_id = s["name"]
                    samples[seq_id] = {
                        "name": s["name"],
                        "signer": s["signer"],
                        "gloss": s["gloss"],
                        "text": s["text"],
                        "sign": samples[seq_id]["sign"]
                    }

            else:
                tmp = load_dataset_file(obj)
                for s in tmp:
                    seq_id = s["name"]
                    if seq_id in samples:
                        assert samples[seq_id]["name"] == s["name"]
                        assert samples[seq_id]["signer"] == s["signer"]
                        assert samples[seq_id]["gloss"] == s["gloss"]
                        assert samples[seq_id]["text"] == s["text"]
                        samples[seq_id]["sign"] = torch.cat(
                            [samples[seq_id]["sign"], s["sign"]], axis=1
                        )
                    else:
                        samples[seq_id] = {
                            "name": s["name"],
                            "signer": s["signer"],
                            "gloss": s["gloss"],
                            "text": s["text"],
                            "sign": s["sign"],
                        }

        examples = []
        for s in samples:
            sample = samples[s]
            examples.append(
                data.Example.fromlist(
                    [
                        sample["name"],
                        sample["signer"],
                        # This is for numerical stability
                        sample["sign"] + 1e-8,
                        sample["gloss"].strip(),
                        sample["text"].strip(),
                    ],
                    fields,
                )
            )
        super().__init__(examples, fields, **kwargs)
