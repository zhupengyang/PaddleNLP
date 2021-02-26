# Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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
import re
import os
import json
import math
import time
import argparse

import numpy as np
import paddle
from paddle import inference
from paddle.io import DataLoader, Dataset

import sys
sys.path.append('../../../')
from paddlenlp.transformers import GPT2Model, GPT2ForPretraining
from paddlenlp.transformers import GPT2Tokenizer, GPT2ChineseTokenizer
from paddlenlp.transformers import GPT2Model
from paddlenlp.data import Stack, Tuple, Pad
from paddlenlp.utils.log import logger

MODEL_CLASSES = {
    "gpt2-cn": (GPT2ForPretraining, GPT2ChineseTokenizer),
    "gpt2": (GPT2ForPretraining, GPT2Tokenizer),
}

# yapf: disable
parser = argparse.ArgumentParser()
parser.add_argument(
    "--model_type",
    default=None,
    type=str,
    required=True,
    help="Model type selected in the list: " +
    ", ".join(MODEL_CLASSES.keys()), )
parser.add_argument("--model_path", default=None, type=str, required=True, help="Path to pre-trained model or shortcut name selected in the list: "
        + ", ".join(sum([list(classes[-1].pretrained_init_configuration.keys()) for classes in MODEL_CLASSES.values()], [])), )
parser.add_argument("--eval_path", default=None, type=str, required=True, help="The eval file path.", )
parser.add_argument('--cloze_eval', action='store_true', help='Evaluation dataset from `--eval_path` is a cloze task')
parser.add_argument('--overlapping_eval', type=int, default=32, help='Sliding window for overlapping eval ')
parser.add_argument("--init_checkpoint_path", default=None, type=str, help="The model checkpoint path.", )
parser.add_argument( "--batch_size", default=8, type=int, help="Batch size per GPU/CPU for training.", )
parser.add_argument('--seq_length', type=int, default=1024, help='Maximum sequence length to process for evaluation.')
parser.add_argument("--device", type=str, default="gpu", help="Select cpu, gpu, xpu devices.")
parser.add_argument("--logging_steps", type=int, default=100, help="Log every X updates steps.")
parser.add_argument("--out_file", type=str, default="out_xpu.txt", help="")

# yapf: enable


class LM_Eval_Dataset(paddle.io.Dataset):
    def __init__(self, tokens, seq_len, pad_idx, overlapping_eval=None):
        self.tokens = tokens
        self.seq_len = seq_len
        self.pad_idx = pad_idx
        self.overlapping_eval = overlapping_eval
        if self.overlapping_eval is None:
            self.overlapping_eval = self.seq_len
        self.overlapping_eval = max(1, self.overlapping_eval)

        self.total_targets = len(self.tokens) - 1
        # remove first sequence tokens
        targets = max(self.total_targets - self.overlapping_eval, 0)
        self.total_sequences = max(
            math.ceil(targets / self.overlapping_eval) + 1, 1)

    def __len__(self):
        return self.total_sequences

    def _construct_sample(self, tokens):
        tokens = np.array(tokens).astype("int64").tolist()
        labels = tokens[1:]
        tokens = tokens[:-1]
        seq_length = len(tokens)
        # attention mask for the attention calulate
        attention_mask = np.tri(seq_length, seq_length).reshape(
            (1, seq_length, seq_length))

        # the pad and eod tokens do not contribute the loss
        loss_mask = np.ones(seq_length, dtype="float32")
        loss_mask[np.where(np.array(tokens) == self.pad_idx)] = 0.0
        position_ids = np.arange(0, seq_length, dtype="int64")

        # -INF mask value as default
        attention_mask = (attention_mask - 1.0) * 1e9
        # Bool mask of attention
        attention_mask = attention_mask.astype("float32")
        return [tokens, loss_mask, attention_mask, position_ids, labels]

    def __getitem__(self, idx):
        start_idx = idx * self.overlapping_eval
        end_idx = start_idx + self.seq_len
        tokens = self.tokens[start_idx:end_idx + 1]
        num_tokens = len(tokens)
        if num_tokens < self.seq_len + 1:
            num_pad = (self.seq_len + 1 - num_tokens)
            tokens += [self.pad_idx] * num_pad
        [tokens, loss_mask, attention_mask, position_ids,
         labels] = self._construct_sample(tokens)
        if self.overlapping_eval != self.seq_len and idx != 0:
            loss_mask[:-self.overlapping_eval] *= 0

        return [tokens, loss_mask, attention_mask, position_ids, labels]


class Lambada_Eval_Dataset(paddle.io.Dataset):
    def __init__(self, tokens, labels, seq_len, pad_idx):
        self.seq_len = seq_len
        self.pad_idx = pad_idx
        self.tokens = tokens
        self.labels = labels

    def __len__(self):
        return len(self.tokens)

    def _construct_sample(self, tokens):
        tokens = np.array(tokens).astype("int64").tolist()
        #labels = tokens[1:]
        tokens = tokens[:-1]

        seq_length = len(tokens)
        # attention mask for the attention calulate
        attention_mask = np.tri(seq_length, seq_length).reshape(
            (1, seq_length, seq_length))

        # the pad and eod tokens do not contribute the loss
        position_ids = np.arange(0, seq_length, dtype="int64")

        # -INF mask value as default
        attention_mask = (attention_mask - 1.0) * 1e9
        # Bool mask of attention
        attention_mask = attention_mask.astype("float32")
        return [tokens, attention_mask, position_ids, labels]

    def __getitem__(self, idx):
        tokens = self.tokens[idx][:self.seq_len]
        labels = self.labels[idx]
        #tokens = tokens + labels
        #num_tokens = len(tokens)
        #if num_tokens < self.seq_len + 1:
        #    num_pad = (self.seq_len + 1 - num_tokens)
        #    tokens += [self.pad_idx] * num_pad
        #loss_mask = np.zeros(self.seq_len, dtype="float32")
        #loss_mask[num_tokens - len(labels) - 1:num_tokens - 1] = 1.
        #[tokens, attention_mask, position_ids, labels] = self._construct_sample(
        #    tokens)
        #return [tokens, loss_mask, attention_mask, position_ids, labels]
        return tokens, labels


def wikitext_detokenizer(string):
    # contractions
    string = string.replace("s '", "s'")
    string = re.sub(r"/' [0-9]/", r"/'[0-9]/", string)
    # number separators
    string = string.replace(" @-@ ", "-")
    string = string.replace(" @,@ ", ",")
    string = string.replace(" @.@ ", ".")
    # punctuation
    string = string.replace(" : ", ": ")
    string = string.replace(" ; ", "; ")
    string = string.replace(" . ", ". ")
    string = string.replace(" ! ", "! ")
    string = string.replace(" ? ", "? ")
    string = string.replace(" , ", ", ")
    # double brackets
    string = re.sub(r"\(\s*([^\)]*?)\s*\)", r"(\1)", string)
    string = re.sub(r"\[\s*([^\]]*?)\s*\]", r"[\1]", string)
    string = re.sub(r"{\s*([^}]*?)\s*}", r"{\1}", string)
    string = re.sub(r"\"\s*([^\"]*?)\s*\"", r'"\1"', string)
    string = re.sub(r"'\s*([^']*?)\s*'", r"'\1'", string)
    # miscellaneous
    string = string.replace("= = = =", "====")
    string = string.replace("= = =", "===")
    string = string.replace("= =", "==")
    string = string.replace(" " + chr(176) + " ", chr(176))
    string = string.replace(" \n", "\n")
    string = string.replace("\n ", "\n")
    string = string.replace(" N ", " 1 ")
    string = string.replace(" 's", "'s")
    return string


def get_tokens(tokenizer, text, strict=True):
    if not strict:
        tokens = tokenizer.encode(text)
        return tokens[:-1], [tokens[-1]]
    last_token = text.split()[-1]
    start_idx = text.rfind(last_token)
    beginning_tokens = tokenizer.encode(text[:start_idx].strip())
    last_token = tokenizer.encode(' ' + last_token)
    return beginning_tokens, last_token


def create_eval_dataset(args):
    val_dataloader = None
    eval_batch_size = args.batch_size
    seq_len = args.seq_length

    tokenizer = GPT2Tokenizer.from_pretrained(os.path.dirname(args.model_path))
    pad_token = tokenizer.command_name_map["pad"].Id

    if not args.cloze_eval:
        with open(args.eval_path, "rb") as reader:
            entire_data = reader.read().decode('utf-8')
        num_original_tokens = len(entire_data.strip().split(" "))
        entire_data = wikitext_detokenizer(entire_data)
        tokenized_data = tokenizer.encode(entire_data)
        num_tokenized_tokens = len(tokenized_data)
        print('Original Tokens: %d, Detokenized tokens: %d' %
              (num_tokenized_tokens, num_original_tokens))
        val_dataset = LM_Eval_Dataset(tokenized_data, seq_len, pad_token,
                                      args.overlapping_eval)
    else:
        tokenized_data = []
        tokenized_label = []
        with open(args.eval_path, 'r') as f:
            for line in f.readlines():
                text = json.loads(line)['text']
                tokens, labels = get_tokens(tokenizer, text)
                tokenized_data.append(tokens)
                tokenized_label.append(labels)
        val_dataset = Lambada_Eval_Dataset(tokenized_data, tokenized_label,
                                           seq_len, pad_token)
        num_tokenized_tokens = 0
        num_original_tokens = 0

    args.num_examples = len(val_dataset)
    args.num_original_tokens = num_original_tokens
    args.num_tokenized_tokens = num_tokenized_tokens
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=eval_batch_size,
        drop_last=False,
        collate_fn=Tuple(Stack(), Stack()))
    #collate_fn=Tuple(Stack(), Stack(), Stack(), Stack(), Stack()))

    return val_dataloader


class Predictor(object):
    def __init__(self, predictor, input_handles, output_handles):
        self.predictor = predictor
        self.input_handles = input_handles
        self.output_handles = output_handles

    @classmethod
    def create_predictor(cls, args):
        config = paddle.inference.Config(args.model_path + ".pdmodel",
                                         args.model_path + ".pdiparams")
        if args.device == "gpu":
            # set GPU configs accordingly
            config.enable_use_gpu(100, 0)
        elif args.select_device == "cpu":
            # set CPU configs accordingly,
            # such as enable_mkldnn, set_cpu_math_library_num_threads
            config.disable_gpu()
        elif args.select_device == "xpu":
            # set XPU configs accordingly
            config.enable_xpu(100)
        config.switch_use_feed_fetch_ops(False)
        predictor = paddle.inference.create_predictor(config)
        input_handles = [
            predictor.get_input_handle(name)
            for name in predictor.get_input_names()
        ]
        #print("input_handles", predictor.get_input_names())
        output_handles = [
            predictor.get_input_handle(name)
            for name in predictor.get_output_names()
        ]
        return cls(predictor, input_handles, output_handles)

    def predict_batch(self, data):
        for input_field, input_handle in zip(data, self.input_handles):
            # print(data)
            #print(data.shape)
            input_handle.copy_from_cpu(input_field.numpy() if isinstance(
                input_field, paddle.Tensor) else input_field)
        self.predictor.run()
        output = [
            output_handle.copy_to_cpu() for output_handle in self.output_handles
        ]
        return output

    def predict(self, dataset, batch_size=1):
        outputs = []
        for data in dataset:
            output = self.predict_batch(data)
            print(output)
            outputs.append(output)
        # print(outputs)
        #print(outputs[0])
        return outputs


def main():
    args = parse_args()
    predictor = Predictor.create_predictor(args)
    args.model_type = args.model_type.lower()
    model_class, tokenizer_class = MODEL_CLASSES[args.model_type]
    tokenizer = tokenizer_class.from_pretrained(
        os.path.dirname(args.model_path))
    ds = ["问题：中国的首都是哪里？答案：北京。\n问题：百度的厂长是谁? 答案："]
    if args.model_type == "gpt2":
        ds = [
            "Question: Where is the capital of China? Answer: Beijing. \nQuestion: Who is the CEO of Apple? Answer:",
        ]
    dataset = [
        np.array(tokenizer.encode(text)).astype("int64").reshape([1, 1, -1])
        for text in ds
    ]
    print(dataset[0])
    print(dataset[0].shape)
    for out in predictor.predict(dataset):
        pass


def do_eval(args):
    assert args.device in [
        "cpu", "gpu", "xpu"
    ], "Invalid device! Available device should be cpu, gpu, or xpu."
    #paddle.set_device(args.device)
    paddle.set_device("cpu")
    # predictor = Predictor.create_predictor(args)
    args.model_type = args.model_type.lower()

    model_class, tokenizer_class = MODEL_CLASSES[args.model_type]
    tokenizer = tokenizer_class.from_pretrained(
        os.path.dirname(args.model_path))

    tic_eval = time.time()
    eval_data_loader = create_eval_dataset(args)
    total_score = 0
    score_name = "loss" if not args.cloze_eval else "number correct"
    with paddle.no_grad():
        outs = np.loadtxt(args.out_file, 'int64', delimiter=' ')
        for step, batch in enumerate(eval_data_loader):
            # tokens, loss_mask, attention_mask, position_ids, labels = batch
            tokens, labels = batch
            # preds = predictor.predict_batch([tokens.numpy()])
            preds = [outs[step]]

            if not args.cloze_eval:
                masked_lm_loss = paddle.nn.functional.cross_entropy(
                    preds, labels, reduction="none")
                loss = paddle.sum(masked_lm_loss * loss_mask)
                total_score += loss.numpy() / (args.num_tokenized_tokens - 1)
            else:
                # outputs = paddle.argmax(preds, -1)
                # acc = paddle.cast(outputs == labels, 'float32')
                # acc = paddle.where(paddle.cast(loss_mask, 'bool'), acc, paddle.ones_like(acc))
                # acc = paddle.sum(paddle.prod(acc, -1))
                # total_score += acc.numpy()
                preds = preds[0].reshape([-1])
                labels = labels.reshape([-1])
                if len(labels) <= len(preds):
                    flag = 1
                    for a, b in zip(labels, preds):
                        if a != b:
                            flag = 0
                            break
                    total_score += flag

            if step % args.logging_steps == 0:
                logger.info("step %d, batch: %d, %s: %f, speed: %.2f step/s" %
                            (step, step, score_name, total_score,
                             args.logging_steps / (time.time() - tic_eval)))
                tic_eval = time.time()

    if not args.cloze_eval:
        total_loss = float(total_score)
        ppl = math.exp(min(20, total_loss))
        token_ratio = (args.num_tokenized_tokens - 1) / (
            args.num_original_tokens - 1)
        adjusted_ppl = math.exp(min(20, total_loss * token_ratio))
        string = ' validation results on {} | '.format(args.eval_path)
        string += 'avg loss: {:.4E} | '.format(total_loss)
        string += 'ppl: {:.4E} | '.format(ppl)
        string += 'adjusted ppl: {:.4E} | '.format(adjusted_ppl)
        string += 'token ratio: {} |'.format(token_ratio)
    else:
        num_correct = float(total_score)
        acc = float(num_correct / args.num_examples)
        string = ' validation results on {} | '.format(args.eval_path)
        string += 'number correct: {:.4E} | '.format(num_correct)
        string += 'total examples: {:.4E} | '.format(args.num_examples)
        string += 'avg accuracy: {:.4E}'.format(acc)
    logger.info(string)


if __name__ == "__main__":
    args = parser.parse_args()
    do_eval(args)
