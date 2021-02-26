# Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
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

import argparse
import os

import paddle
from paddlenlp.transformers import GPT2ForGreedyGeneration, GPT2ChineseTokenizer, GPT2Tokenizer, GPT2ForPretraining

MODEL_CLASSES = {
    "gpt2-cn": (GPT2ForGreedyGeneration, GPT2ChineseTokenizer),
    "gpt2": (GPT2ForGreedyGeneration, GPT2Tokenizer),
    "gpt2-train": (GPT2ForPretraining, GPT2Tokenizer),
}


def parse_args():
    parser = argparse.ArgumentParser()
    # Required parameters
    parser.add_argument(
        "--model_type",
        default=None,
        type=str,
        required=True,
        help="Model type selected in the list: " +
        ", ".join(MODEL_CLASSES.keys()), )
    parser.add_argument(
        "--model_path",
        default=None,
        type=str,
        required=True,
        help="Path of the trained model to be exported.", )
    parser.add_argument(
        "--output_path",
        default=None,
        type=str,
        required=True,
        help="The output file prefix used to save the exported inference model.",
    )
    args = parser.parse_args()
    return args


def main():
    args = parse_args()

    args.model_type = args.model_type.lower()
    model_class, tokenizer_class = MODEL_CLASSES[args.model_type]

    # build model and load trained parameters
    model = model_class.from_pretrained(args.model_path)
    # switch to eval model
    model.eval()
    # convert to static graph with specific input description
    if args.model_type != 'gpt2-train':
        model = paddle.jit.to_static(
            model,
            input_spec=[
                paddle.static.InputSpec(
                    shape=[None, None], dtype="int64"),  # input_ids
            ])
    else:
        model = paddle.jit.to_static(
            model,
            input_spec=[
                paddle.static.InputSpec(
                    shape=[None, None], dtype="int64"),  # input_ids
                paddle.static.InputSpec(
                    shape=[None, None], dtype="int64"),  # position_ids
                paddle.static.InputSpec(
                    shape=[None, None], dtype="float32"),  # attention_mask
            ])

    # save converted static graph model
    paddle.jit.save(model, args.output_path)
    # also save tokenizer for inference usage
    tokenizer = tokenizer_class.from_pretrained(args.model_path)
    tokenizer.save_pretrained(os.path.dirname(args.output_path))


if __name__ == "__main__":
    main()