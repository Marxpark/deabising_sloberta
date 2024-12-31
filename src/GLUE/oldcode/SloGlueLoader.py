import datasets
# coding=utf-8
# Copyright 2020 The TensorFlow Datasets Authors and the HuggingFace Datasets Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Lint as: python3
import json
import os



class SloGlueConfig(datasets.BuilderConfig):
    """BuilderConfig for SuperGLUE."""

    def __init__(self, features, data_url, citation, url, label_classes=("False", "True"), **kwargs):
        """BuilderConfig for SuperGLUE.
        Args:
          features: `list[string]`, list of the features that will appear in the
            feature dict. Should not include "label".
          data_url: `string`, url to download the zip file from.
          citation: `string`, citation for the data set.
          url: `string`, url for information about the data set.
          label_classes: `list[string]`, the list of classes for the label if the
            label is present as a string. Non-string labels will be cast to either
            'False' or 'True'.
          **kwargs: keyword arguments forwarded to super.
        """
        # Version history:
        # 1.0.2: Fixed non-nondeterminism in ReCoRD.
        # 1.0.1: Change from the pre-release trial version of SuperGLUE (v1.9) to
        #        the full release (v2.0).
        # 1.0.0: S3 (new shuffling, sharding and slicing mechanism).
        # 0.0.2: Initial version.
        super(SloGlueConfig, self).__init__(version=datasets.Version("2.3.2"), **kwargs)
        self.features = features
        self.label_classes = label_classes
        self.data_url = data_url
        self.citation = citation
        self.url = url


class SloGlue(datasets.GeneratorBasedBuilder):
    """The SuperGLUE benchmark."""

    BUILDER_CONFIGS = [
        SloGlueConfig(
            name="boolq",
            description="testmate",
            features=["question", "passage"],
            data_url="/Users/marko/dev/maga/context-debias/data/SloSuperGLUE/SuperGLUE-HumanT/json/BoolQ.zip",
            citation="_BOOLQ_CITATION",
            url="https://github.com/google-research-datasets/boolean-questions",
        ),
        SloGlueConfig(
            name="cb",
            description="_CB_DESCRIPTION",
            features=["premise", "hypothesis"],
            label_classes=["entailment", "contradiction", "neutral"],
            data_url="/Users/marko/dev/maga/context-debias/data/SloSuperGLUE/SuperGLUE-HumanT/json/CB.zip",
            citation="_CB_CITATION",
            url="https://github.com/mcdm/CommitmentBank",
        ),
        SloGlueConfig(
            name="copa",
            description="_COPA_DESCRIPTION",
            label_classes=["choice1", "choice2"],
            # Note that question will only be the X in the statement "What's
            # the X for this?".
            features=["premise", "choice1", "choice2", "question"],
            data_url="/Users/marko/dev/maga/context-debias/data/SloSuperGLUE/SuperGLUE-HumanT/json/COPA.zip",
            citation="_COPA_CITATION",
            url="http://people.ict.usc.edu/~gordon/copa.html",
        ),
        SloGlueConfig(
            name="multirc",
            description="_MULTIRC_DESCRIPTION",
            features=["paragraph", "question", "answer"],
            data_url="/Users/marko/dev/maga/context-debias/data/SloSuperGLUE/SuperGLUE-HumanT/json/MultiRC.zip",
            citation="_MULTIRC_CITATION",
            url="https://cogcomp.org/multirc/",
        ),
        SloGlueConfig(
            name="record",
            description="_RECORD_DESCRIPTION",
            # Note that entities and answers will be a sequences of strings. Query
            # will contain @placeholder as a substring, which represents the word
            # to be substituted in.
            features=["passage", "query", "entities", "answers"],
            data_url="/Users/marko/dev/maga/context-debias/data/SloSuperGLUE/SuperGLUE-HumanT/json/ReCoRD.zip",
            citation="_RECORD_CITATION",
            url="https://sheng-z.github.io/ReCoRD-explorer/",
        ),
        SloGlueConfig(
            name="rte",
            description="_RTE_DESCRIPTION",
            features=["premise", "hypothesis"],
            label_classes=["entailment", "not_entailment"],
            data_url="/Users/marko/dev/maga/context-debias/data/SloSuperGLUE/SuperGLUE-HumanT/json/RTE.zip",
            citation="_RTE_CITATION",
            url="https://aclweb.org/aclwiki/Recognizing_Textual_Entailment",
        ),
        SloGlueConfig(
            name="wsc",
            description="_WSC_DESCRIPTION",
            # Note that span1_index and span2_index will be integers stored as
            # datasets.Value('int32').
            features=["text", "span1_index", "span2_index", "span1_text", "span2_text"],
            data_url="/Users/marko/dev/maga/context-debias/data/SloSuperGLUE/SuperGLUE-HumanT/json/WSC.zip",
            citation="_WSC_CITATION",
            url="https://cs.nyu.edu/faculty/davise/papers/WinogradSchemas/WS.html",
        ),
        SloGlueConfig(
            name="boolqGMT",
            description="testmate",
            features=["question", "passage"],
            data_url="/Users/marko/dev/maga/context-debias/data/SloSuperGLUE/SuperGLUE-GoogleMT/json/BoolQ.zip",
            citation="_BOOLQ_CITATION",
            url="https://github.com/google-research-datasets/boolean-questions",
        ),
        SloGlueConfig(
            name="cbGMT",
            description="_CB_DESCRIPTION",
            features=["premise", "hypothesis"],
            label_classes=["entailment", "contradiction", "neutral"],
            data_url="/Users/marko/dev/maga/context-debias/data/SloSuperGLUE/SuperGLUE-GoogleMT/json/CB.zip",
            citation="_CB_CITATION",
            url="https://github.com/mcdm/CommitmentBank",
        ),
        SloGlueConfig(
            name="copaGMT",
            description="_COPA_DESCRIPTION",
            label_classes=["choice1", "choice2"],
            # Note that question will only be the X in the statement "What's
            # the X for this?".
            features=["premise", "choice1", "choice2", "question"],
            data_url="/Users/marko/dev/maga/context-debias/data/SloSuperGLUE/SuperGLUE-GoogleMT/json/COPA.zip",
            citation="_COPA_CITATION",
            url="http://people.ict.usc.edu/~gordon/copa.html",
        ),
        SloGlueConfig(
            name="multircGMT",
            description="_MULTIRC_DESCRIPTION",
            features=["paragraph", "question", "answer"],
            data_url="/Users/marko/dev/maga/context-debias/data/SloSuperGLUE/SuperGLUE-GoogleMT/json/MultiRC.zip",
            citation="_MULTIRC_CITATION",
            url="https://cogcomp.org/multirc/",
        ),
        SloGlueConfig(
            name="recordGMT",
            description="_RECORD_DESCRIPTION",
            # Note that entities and answers will be a sequences of strings. Query
            # will contain @placeholder as a substring, which represents the word
            # to be substituted in.
            features=["passage", "query", "entities", "answers"],
            data_url="/Users/marko/dev/maga/context-debias/data/SloSuperGLUE/SuperGLUE-GoogleMT/json/ReCoRD.zip",
            citation="_RECORD_CITATION",
            url="https://sheng-z.github.io/ReCoRD-explorer/",
        ),
        SloGlueConfig(
            name="rteGMT",
            description="_RTE_DESCRIPTION",
            features=["premise", "hypothesis"],
            label_classes=["entailment", "not_entailment"],
            data_url="/Users/marko/dev/maga/context-debias/data/SloSuperGLUE/SuperGLUE-GoogleMT/json/RTE.zip",
            citation="_RTE_CITATION",
            url="https://aclweb.org/aclwiki/Recognizing_Textual_Entailment",
        ),
        SloGlueConfig(
            name="wscGMT",
            description="_WSC_DESCRIPTION",
            # Note that span1_index and span2_index will be integers stored as
            # datasets.Value('int32').
            features=["text", "span1_index", "span2_index", "span1_text", "span2_text"],
            data_url="/Users/marko/dev/maga/context-debias/data/SloSuperGLUE/SuperGLUE-GoogleMT/json/WSC.zip",
            citation="_WSC_CITATION",
            url="https://cs.nyu.edu/faculty/davise/papers/WinogradSchemas/WS.html",
        ),
        SloGlueConfig(
            name="wsc.fixed",
            description=(
                    "_WSC_DESCRIPTION" + "\n\nThis version fixes issues where the spans are not actually "
                                       "substrings of the text."
            ),
            # Note that span1_index and span2_index will be integers stored as
            # datasets.Value('int32').
            features=["text", "span1_index", "span2_index", "span1_text", "span2_text"],
            data_url="/Users/marko/dev/maga/context-debias/data/SloSuperGLUE/SuperGLUE-HumanT/json/WSC.zip",
            citation="_WSC_CITATION",
            url="https://cs.nyu.edu/faculty/davise/papers/WinogradSchemas/WS.html",
        ),
    ]

    def _info(self):
        features = {feature: datasets.Value("string") for feature in self.config.features}
        if self.config.name.startswith("wsc"):
            features["span1_index"] = datasets.Value("int32")
            features["span2_index"] = datasets.Value("int32")
        if self.config.name == "wic":
            features["start1"] = datasets.Value("int32")
            features["start2"] = datasets.Value("int32")
            features["end1"] = datasets.Value("int32")
            features["end2"] = datasets.Value("int32")
        if self.config.name == "multirc":
            features["idx"] = dict(
                {
                    "paragraph": datasets.Value("int32"),
                    "question": datasets.Value("int32"),
                    "answer": datasets.Value("int32"),
                }
            )
        elif self.config.name == "record":
            features["idx"] = dict(
                {
                    "passage": datasets.Value("int32"),
                    "query": datasets.Value("int32"),
                }
            )
        else:
            features["idx"] = datasets.Value("int32")

        if self.config.name == "record":
            # Entities are the set of possible choices for the placeholder.
            features["entities"] = datasets.features.Sequence(datasets.Value("string"))
            # Answers are the subset of entities that are correct.
            features["answers"] = datasets.features.Sequence(datasets.Value("string"))
        else:
            features["label"] = datasets.features.ClassLabel(names=self.config.label_classes)

        return datasets.DatasetInfo(
            description="_GLUE_DESCRIPTION" + self.config.description,
            features=datasets.Features(features),
            homepage=self.config.url,
            citation=self.config.citation + "\n" + "_SUPER_GLUE_CITATION",
        )

    def _split_generators(self, dl_manager):
        dl_dir = dl_manager.download_and_extract(self.config.data_url) or ""
        task_name = _get_task_name_from_data_url(self.config.data_url)
        dl_dir = os.path.join(dl_dir, task_name)
        if self.config.name in ["axb", "axg"]:
            return [
                datasets.SplitGenerator(
                    name=datasets.Split.TEST,
                    gen_kwargs={
                        "data_file": os.path.join(dl_dir, f"{task_name}.jsonl"),
                        "split": datasets.Split.TEST,
                    },
                ),
            ]
        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                gen_kwargs={
                    "data_file": os.path.join(dl_dir, "train.jsonl"),
                    "split": datasets.Split.TRAIN,
                },
            ),
            datasets.SplitGenerator(
                name=datasets.Split.VALIDATION,
                gen_kwargs={
                    "data_file": os.path.join(dl_dir, "val.jsonl"),
                    "split": datasets.Split.VALIDATION,
                },
            ),
            datasets.SplitGenerator(
                name=datasets.Split.TEST,
                gen_kwargs={
                    "data_file": os.path.join(dl_dir, "test.jsonl"),
                    "split": datasets.Split.TEST,
                },
            ),
        ]

    def _generate_examples(self, data_file, split):
        with open(data_file, encoding="utf-8") as f:
            for line in f:
                row = json.loads(line)

                if self.config.name == "multirc":
                    paragraph = row["passage"]
                    for question in paragraph["questions"]:
                        for answer in question["answers"]:
                            label = answer.get("label")
                            key = "%s_%s_%s" % (row["idx"], question["idx"], answer["idx"])
                            yield key, {
                                "paragraph": paragraph["text"],
                                "question": question["question"],
                                "answer": answer["text"],
                                "label": -1 if label is None else _cast_label(bool(label)),
                                "idx": {"paragraph": row["idx"], "question": question["idx"], "answer": answer["idx"]},
                            }
                elif self.config.name == "record":
                    passage = row["passage"]
                    for qa in row["qas"]:
                        yield qa["idx"], {
                            "passage": passage["text"],
                            "query": qa["query"],
                            "entities": _get_record_entities(passage),
                            "answers": _get_record_answers(qa),
                            "idx": {"passage": row["idx"], "query": qa["idx"]},
                        }
                else:
                    if self.config.name.startswith("wsc"):
                        row.update(row["target"])
                    example = {feature: row[feature] for feature in self.config.features}
                    if self.config.name == "wsc.fixed":
                        example = _fix_wst(example)
                    example["idx"] = row["idx"]

                    if "label" in row:
                        if self.config.name == "copa":
                            example["label"] = "choice2" if row["label"] else "choice1"
                        else:
                            example["label"] = _cast_label(row["label"])
                    else:
                        assert split == datasets.Split.TEST, row
                        example["label"] = -1
                    yield example["idx"], example


def _fix_wst(ex):
    """Fixes most cases where spans are not actually substrings of text."""

    def _fix_span_text(k):
        """Fixes a single span."""
        text = ex[k + "_text"]
        index = ex[k + "_index"]

        if text in ex["text"]:
            return

        if text in ("Kamenev and Zinoviev", "Kamenev, Zinoviev, and Stalin"):
            # There is no way to correct these examples since the subjects have
            # intervening text.
            return

        if "theyscold" in text:
            ex["text"].replace("theyscold", "they scold")
            ex["span2_index"] = 10
        # Make sure case of the first words match.
        first_word = ex["text"].split()[index]
        if first_word[0].islower():
            text = text[0].lower() + text[1:]
        else:
            text = text[0].upper() + text[1:]
        # Remove punctuation in span.
        text = text.rstrip(".")
        # Replace incorrect whitespace character in span.
        text = text.replace("\n", " ")
        ex[k + "_text"] = text
        assert ex[k + "_text"] in ex["text"], ex

    _fix_span_text("span1")
    _fix_span_text("span2")
    return ex


def _cast_label(label):
    """Converts the label into the appropriate string version."""
    if isinstance(label, str):
        return label
    elif isinstance(label, bool):
        return "True" if label else "False"
    elif isinstance(label, int):
        assert label in (0, 1)
        return str(label)
    else:
        raise ValueError("Invalid label format.")


def _get_record_entities(passage):
    """Returns the unique set of entities."""
    text = passage["text"]
    entities = set()
    for entity in passage["entities"]:
        entities.add(text[entity["start"]: entity["end"] + 1])
    return sorted(entities)


def _get_record_answers(qa):
    """Returns the unique set of answers."""
    if "answers" not in qa:
        return []
    answers = set()
    for answer in qa["answers"]:
        answers.add(answer["text"])
    return sorted(answers)


def _get_task_name_from_data_url(data_url):
    return data_url.split("/")[-1].split(".")[0]



# --model_name_or_path EMBEDDIA/sloberta --train_file ../../data/SloSuperGLUE/SuperGLUE-HumanT/json/RTE/train.jsonl --validation_file ../../data/SloSuperGLUE/SuperGLUE-HumanT/json/RTE/val.jsonl --test_file ../../data/SloSuperGLUE/SuperGLUE-HumanT/json/RTE/test.jsonl --do_train --do_eval --do_predict --max_seq_length 64 --per_device_train_batch_size 32 --per_device_eval_batch_size 32 --learning_rate 2e-5 --num_train_epochs 1.0 --output_dir ./GLUEOUTPUT/result.bin --overwrite_output_dir --max_predict_samples 1000 --max_eval_samples 1000 --max_train_samples 1000