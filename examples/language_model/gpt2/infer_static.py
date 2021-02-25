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
import math
import collections
import itertools
import logging
import os
import random
import time
import h5py
from functools import partial

import numpy as np

import paddle
import paddle.distributed.fleet as fleet
from paddle.io import DataLoader, Dataset

from paddlenlp.data import Stack, Tuple, Pad
from paddlenlp.transformers import GPT2Model, GPT2ForPretraining, GPT2PretrainingCriterion
from paddlenlp.transformers import GPT2Tokenizer, GPT2ChineseTokenizer, GPT2ForGreedyGeneration
from paddlenlp.utils.log import logger
from data import GPT2Dataset
import lr

MODEL_CLASSES = {
    "gpt2": (GPT2ForGreedyGeneration, GPT2Tokenizer),
    "gpt2-cn": (GPT2ForGreedyGeneration, GPT2ChineseTokenizer),
}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_type",
        default=None,
        type=str,
        required=True,
        help="Model type selected in the list: " +
        ", ".join(MODEL_CLASSES.keys()), )
    parser.add_argument(
        "--model_name_or_path",
        default=None,
        type=str,
        required=True,
        help="Path to pre-trained model or shortcut name selected in the list: "
        + ", ".join(
            sum([
                list(classes[-1].pretrained_init_configuration.keys())
                for classes in MODEL_CLASSES.values()
            ], [])), )
    parser.add_argument(
        "--output_dir",
        default=None,
        type=str,
        required=True,
        help="The output directory where the model predictions and checkpoints will be written.",
    )

    args = parser.parse_args()
    return args


def reset_program_state_dict(model, state_dict):
    scale = model.initializer_range if hasattr(model, "initializer_range")\
        else model.gpt2.config["initializer_range"]

    new_state_dict = dict()
    for n, p in state_dict.items():
        if "layer_norm" not in p.name:
            dtype_str = "float32"
            if str(p.dtype) == "VarType.FP64":
                dtype_str = "float64"
            new_state_dict[p.name] = np.random.normal(
                loc=0.0, scale=scale, size=p.shape).astype(dtype_str)
    return new_state_dict


def copy_program_state_dict(static_dict, tensor_dict):
    new_state_dict = dict()
    for n, p in static_dict.items():
        new_state_dict[p.name] = tensor_dict[n]
    return new_state_dict


def load_pretrained_params(model, program):
    state_dict = model.state_dict()
    #Use the state dict to update the parameter
    tensor_dict = paddle.load(
        "/ssd1/zhonghui03/.paddlenlp/models/gpt2-medium-en/gpt2-medium-en.pdparams"
    )
    #    "/ssd1/zhonghui03/.paddlenlp/models/gpt2-base-cn/gpt2-base-cn.pdparams")
    reset_state_dict = copy_program_state_dict(state_dict, tensor_dict)
    paddle.static.set_program_state(program, reset_state_dict)


def create_data_holder(args):
    tokens = paddle.static.data(name="tokens", shape=[-1, -1], dtype="int64")
    return [tokens]


def do_train(args):
    # Initialize the paddle and paddle fleet execute enviroment
    paddle.enable_static()
    place = paddle.set_device("gpu")

    # Define the input data in the static mode
    main_program = paddle.static.default_main_program()
    startup_program = paddle.static.default_startup_program()

    model_class, tokenizer_class = MODEL_CLASSES[args.model_type]
    tokenizer = tokenizer_class.from_pretrained(args.model_name_or_path)
    config = model_class.pretrained_init_configuration[args.model_name_or_path]
    if config["vocab_size"] % 8 != 0:
        config["vocab_size"] += 8 - (config["vocab_size"] % 8)

    # create the model for the gpt model
    tokens = paddle.static.data(name="tokens", shape=[-1, -1], dtype="int64")
    position_ids = paddle.static.data(name="pos", shape=[-1, -1], dtype="int64")
    attention_mask = paddle.static.data(
        name="mask", shape=[-1, -1], dtype="float32")
    # model = GPT2ForGreedyGeneration(GPT2Model(**config))
    model = GPT2ForPretraining(GPT2Model(**config))
    preds = model(tokens, position_ids, attention_mask)
    # Define the Executor for running the static model
    exe = paddle.static.Executor(place)
    exe.run(startup_program)

    load_pretrained_params(model, main_program)
    test_program = main_program.clone(for_test=True)
    # state_dict = model.state_dict()
    #Use the state dict to update the parameter
    #reset_state_dict = reset_program_state_dict(model, state_dict)
    # tensor_dict = paddle.load("./layernorm_gpt2.pdparams")
    # reset_state_dict = copy_program_state_dict(state_dict, tensor_dict)
    # paddle.static.set_program_state(main_program, reset_state_dict)
    ds = ["问题：中国的首都是哪里？答案：北京。\n问题：百度的厂长是谁? 答案："]
    if args.model_type == "gpt2":
        ds = [
            "Question: Where is the capital of China? Answer: Beijing. \nQuestion: Who is the CEO of Apple? Answer:",
            "Question: Where is the capital of China? Answer: Beijing. \nQuestion: Who is the CEO of Facebook? Answer:",
            "Question: Where is the capital of China? Answer: Beijing. \nQuestion: How tall is the highest peak in the world? Answer:",
            "Question: Where is the capital of China? Answer: Beijing. \nQuestion: Who is the president of the united states? Answer:",
            "Question: Where is the capital of China? Answer: Beijing. \nQuestion: Where is the capital of France? Answer:",
            "Question: Where is the capital of China? Answer: Beijing. \nQuestion: What is the largest animal in the ocean? Answer:",
            "Question: Where is the capital of China? Answer: Beijing. \nQuestion: How many hours in a day? Answer:",
            "Question: Where is the capital of China? Answer: Beijing. \nQuestion: Who is the chancellor of Germany? Answer:",
        ]
    dataset = [
        np.array(tokenizer.encode(text)).astype("int64").reshape([1, -1])
        for text in ds
    ]
    for i, d in enumerate(dataset):
        loss_return = exe.run(test_program,
                              feed={
                                  "tokens": d,
                                  "pos": np.arange(d.shape[1]).reshape([1, -1]),
                                  "mask": np.zeros_like(d).astype("float32")
                              },
                              fetch_list=[preds])
        print(loss_return)
        exit(0)
        #c= [x for x in loss_return[0].reshape(-1)]
        print("输入token:", d)
        print("输出token:", c)
        print("输入文本:", ds[i])
        print('输出文本:', tokenizer.decode(c))
    output_dir = os.path.join(args.output_dir, "model_infer")
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    paddle.static.save_inference_model(
        output_dir,
        feeded_var_names=[
            tokens.name,
            loss_mask.name,
            attention_mask.name,
            position_ids.name,
            labels.name,
        ],
        target_vars=[loss],
        executor=exe)
    return

    files = [
        os.path.join(args.input_dir, f) for f in os.listdir(args.input_dir)
        if (os.path.isfile(os.path.join(args.input_dir, f)) and "npz_" not in
            str(f))
    ]
    #files.sort()
    num_files = len(files)
    for f_id in range(math.ceil(len(files))):
        data_file = files[0]
        train_data_loader = create_pretrained_dataset(
            args,
            data_file,
            None,
            0,
            eod_id=eod_id,
            places=paddle.static.cuda_places(),
            data_holders=data_holders)
        for step, batch in enumerate(train_data_loader):
            global_step += 1
            loss_return = exe.run(test_program, feed=batch, fetch_list=[loss])

            output_dir = os.path.join(args.output_dir, "model_%d" % global_step)
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)
            paddle.static.save_inference_model(
                output_dir,
                feeded_var_names=[
                    tokens.name,
                    loss_mask.name,
                    attention_mask.name,
                    position_ids.name,
                    labels.name,
                ],
                target_vars=[loss],
                executor=exe)
            del train_data_loader
            return


if __name__ == "__main__":
    args = parse_args()
    do_train(args)
