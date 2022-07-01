# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

# Provide an argument parser and default command line options for using BLINK.
import argparse
import importlib
import os
import sys
import datetime


ENT_START_TAG = "[unused0]"
ENT_END_TAG = "[unused1]"
ENT_TITLE_TAG = "[unused2]"


class BlinkParser(argparse.ArgumentParser):
    """
    Provide an opt-producer and CLI arguement parser.

    More options can be added specific by paassing this object and calling
    ''add_arg()'' or add_argument'' on it.

    :param add_blink_args:
        (default True) initializes the default arguments for BLINK package.
    :param add_model_args:
        (default False) initializes the default arguments for loading models,
        including initializing arguments from the model.
    """

    def __init__(
        self, add_blink_args=True, add_model_args=False, description="BLINK parser",
    ):
        super().__init__(
            description=description,
            allow_abbrev=False,
            conflict_handler="resolve",
            formatter_class=argparse.HelpFormatter,
            add_help=add_blink_args,
        )
        self.blink_home = os.path.dirname(
            os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
        )
        os.environ["BLINK_HOME"] = self.blink_home

        self.add_arg = self.add_argument

        self.overridable = {}

        if add_blink_args:
            self.add_blink_args()
        if add_model_args:
            self.add_model_args()

    def add_blink_args(self, args=None):
        """
        Add common BLINK args across all scripts.
        """
        parser = self.add_argument_group("Common Arguments")
        parser.add_argument(
            "--silent", action="store_true", help="Whether to print progress bars."
        )
        parser.add_argument(
            "--debug",
            action="store_true",
            help="Whether to run in debug mode with only 200 samples.",
        )
        parser.add_argument(
            "--data_parallel",
            action="store_true",
            help="Whether to distributed the candidate generation process.",
        )
        parser.add_argument(
            "--no_cuda",
            action="store_true",
            help="Whether not to use CUDA when available",
        )
        parser.add_argument("--top_k", default=10, type=int)
        parser.add_argument(
            "--seed", type=int, default=52313, help="random seed for initialization"
        )
        parser.add_argument(
            "--zeshel",
            default=True,
            type=bool,
            help="Whether the dataset is from zeroshot.",
        )

    def add_model_args(self, args=None):
        """
        Add model args.
        """
        parser = self.add_argument_group("Model Arguments")
        parser.add_argument(
            "--max_seq_length",
            default=256,
            type=int,
            help="The maximum total input sequence length after WordPiece tokenization. \n"
            "Sequences longer than this will be truncated, and sequences shorter \n"
            "than this will be padded.",
        )
        parser.add_argument(
            "--max_context_length",
            default=128,
            type=int,
            help="The maximum total context input sequence length after WordPiece tokenization. \n"
            "Sequences longer than this will be truncated, and sequences shorter \n"
            "than this will be padded.",
        )
        parser.add_argument(
            "--max_cand_length",
            default=128,
            type=int,
            help="The maximum total label input sequence length after WordPiece tokenization. \n"
            "Sequences longer than this will be truncated, and sequences shorter \n"
            "than this will be padded.",
        )
        parser.add_argument(
            "--training_state_dir",
            default="training_state",
            type=str,
            required=False,
            help="Directory to save the training state, i.e, the model, optimizer, scheduler, etc",
        )
        parser.add_argument(
            "--training_state_file",
            default="training_state.pt",
            type=str,
            required=False,
            help="Name of the file having the training state (optimizer, scheduler, etc)",
        )

        # train_biencoder.py, biencoder.py, etc are using this. but dont use this from the command line if you want to resume training.
        parser.add_argument(
            "--path_to_model",
            default=None,
            type=str,
            required=False,
            help="The full path to the model to load.",
        )
        parser.add_argument(
            "--resume_training",
            action="store_true",
            help="Should the training be resumed from a previous checkpoint?",
        )

        parser.add_argument(
            "--entities_file",
            default=None,
            type=str,
            required=False,
            help="The full path to the file containing the list of entity descriptions",
        )
        parser.add_argument(
            "--hard_negatives_file",
            default=None,
            type=str,
            required=False,
            help="The full path to the file containing the list of hard negatives for each training example",
        )
        parser.add_argument(
            "--max_num_negatives",
            default=9,
            type=int,
            required=False,
            help="The number of negatives for each training example.",
        )

        parser.add_argument(
            "--bert_model",
            default="bert-base-uncased",
            type=str,
            help="Bert pre-trained model selected in the list: bert-base-uncased, "
            "bert-large-uncased, bert-base-cased, bert-base-multilingual, bert-base-chinese.",
        )
        parser.add_argument(
            "--pull_from_layer", type=int, default=-1, help="Layers to pull from BERT",
        )
        parser.add_argument(
            "--lowercase",
            action="store_true",
            help="Whether to lower case the input text. True for uncased models, False for cased models.",
        )
        parser.add_argument("--context_key", default="context", type=str)
        parser.add_argument(
            "--out_dim", type=int, default=1, help="Output dimention of bi-encoders.",
        )
        parser.add_argument(
            "--add_linear",
            action="store_true",
            help="Whether to add an additional linear projection on top of BERT.",
        )
        parser.add_argument(
            "--data_path",
            default="data/zeshel",
            type=str,
            help="The path to the train data.",
        )
        parser.add_argument('--eval_set_paths', action='append', help='The paths to the test data', type=str)
        parser.add_argument(
            "--eval_only",
            action="store_true",
            help="Skip training, just evaluate on the data in eval_set_paths",
        )
        parser.add_argument(
            "--output_path",
            default=None,
            type=str,
            required=True,
            help="The output directory where generated output file (model, etc.) is to be dumped.",
        )

    def add_training_args(self, args=None):
        """
        Add model training args.
        """
        parser = self.add_argument_group("Model Training Arguments")
        parser.add_argument(
            "--evaluate", action="store_true", help="Whether to run evaluation."
        )
        parser.add_argument(
            "--output_eval_file",
            default=None,
            type=str,
            help="The txt file where the the evaluation results will be written.",
        )

        parser.add_argument(
            "--tb",
            action="store_true",
            default=False,
            help="Enable tensorboard logging?",
        )

        parser.add_argument(
            "--processed_train_data_cache",
            default="train.cache",
            type=str,
            help="Preprocessed training data is stored here",
        )
        parser.add_argument(
            "--processed_val_data_cache",
            default="valid.cache",
            type=str,
            help="Preprocessed validation data is stored here",
        )
        parser.add_argument(
            "--train_batch_size",
            default=8,
            type=int,
            help="Total batch size for training.",
        )
        parser.add_argument(
            "--eval_batch_size",
            default=8,
            type=int,
            help="Total batch size for evaluation.",
        )
        parser.add_argument("--max_grad_norm", default=1.0, type=float)
        parser.add_argument(
            "--learning_rate",
            default=3e-5,
            type=float,
            help="The initial learning rate for Adam.",
        )
        parser.add_argument(
            "--num_train_epochs",
            default=1,
            type=int,
            help="Number of training epochs.",
        )
        parser.add_argument(
            "--print_interval", type=int, default=5, help="Interval of loss printing",
        )
        parser.add_argument(
            "--eval_interval",
            type=int,
            default=40,
            help="Interval for evaluation during training",
        )
        parser.add_argument(
            "--save_interval", type=int, default=200, help="Interval for model saving"
        )
        parser.add_argument(
            "--warmup_proportion",
            default=0.1,
            type=float,
            help="Proportion of training to perform linear learning rate warmup for. "
            "E.g., 0.1 = 10% of training.",
        )
        parser.add_argument(
            "--gradient_accumulation_steps",
            type=int,
            default=1,
            help="Number of updates steps to accumualte before performing a backward/update pass.",
        )
        parser.add_argument(
            "--type_optimization",
            type=str,
            default="all_encoder_layers",
            help="Which type of layers to optimize in BERT",
        )
        parser.add_argument(
            "--shuffle", type=bool, default=False, help="Whether to shuffle train data",
        )

    def add_eval_args(self, args=None):
        """
        Add model evaluation args.
        """
        parser = self.add_argument_group("Model Evaluation Arguments")
        parser.add_argument(
            "--mode", default="valid", type=str, help="Train / validation / test",
        )
        parser.add_argument(
            "--save_topk_result",
            action="store_true",
            help="Whether to save prediction results.",
        )
        parser.add_argument(
            "--encode_batch_size", default=8, type=int, help="Batch size for encoding."
        )
        parser.add_argument(
            "--cand_pool_path", default=None, type=str, help="Path for candidate pool",
        )
        parser.add_argument(
            "--cand_encode_path",
            default=None,
            type=str,
            help="Path for candidate encoding",
        )

    def add_type_args(self, args=None):
        """
        Add model evaluation args.
        """
        parser = self.add_argument_group("Arguments for type based model")
        parser.add_argument(
            "--type_model",
            default=1,
            type=int,
            help="Which type model should be used?",
        )
        parser.add_argument(
            "--positive_types",
            type=str,
            choices=["lflc", "lflc_ancestor"],
            default="lflc_ancestor",
        )
        parser.add_argument(
            "--max_type_list_len",
            default=30,
            type=int,
            help="Total number of positive + negative type samples per input",
        )
        parser.add_argument(
            "--num_types", default=60000, type=int, help="Length of type vocabulary",
        )
        parser.add_argument(
            "--types_key",
            default="fgetc_category_id",
            type=str,
            help="the dictionary key in the input file (train.jsonl) having the types list",
        )
        parser.add_argument(
            "--type_embedding_dim",
            default=100,
            type=int,
            help="Length of type vocabulary",
        )
        parser.add_argument(
            "--freeze_type_embeddings",
            action="store_true",
            default=False,
            help="Passing this flag will not train the type embeddings",
        )
        parser.add_argument(
            "--type_embeddings_path",
            default="",
            type=str,
            help="Path to the bert, glove, etc embeddings of the types",
        )
        parser.add_argument(
            "--ontology_file",
            default="",
            type=str,
            help="Path to a file containing the list of child to parent edges",
        )
        parser.add_argument(
            "--no_linear_after_type_embeddings",
            action="store_true",
            default=False,
            help="Should there not be a linear layer on top of type embeddings?",
        )
        parser.add_argument("--blink_loss_weight", default=1.0, type=float)
        parser.add_argument("--type_loss_weight", default=1.0, type=float)
        parser.add_argument("--type_loss_weight_positive", default=1.0, type=float)
        parser.add_argument("--type_loss_weight_negative", default=1.0, type=float)
        parser.add_argument(
            "--main_metric",
            type=str,
            choices=["entity", "type", "entity_and_type"],
            default="entity",
        )
        parser.add_argument(
            "--prob_aggregation",
            type=str,
            choices=["godel", "lukasiewicz", "weighted_lukasiewicz"],
            default="godel",
        )
        parser.add_argument(
            "--freeze_context_bert",
            action="store_true",
            default=False,
            help="Passing this flag will not train the context bert",
        )
        parser.add_argument(
            "--type_network_learning_rate",
            default=3e-5,
            type=float,
            help="The initial learning rate for Adam.",
        )
        parser.add_argument(
            "--type_task_importance_scheduling",
            type=str,
            choices=["none", "loss_weight", "grad_throttle"],
            default="none",
        )
        parser.add_argument(
            "--cls_start",
            default=1,
            type=int,
            help="Use CLS vectors from layers [cls_start: cls_end] to compute the context vector for type prediction",
        )
        parser.add_argument(
            "--cls_end",
            default=12,
            type=int,
            help="Use CLS vectors from layers [cls_start: cls_end] to compute the context vector for type prediction",
        )


