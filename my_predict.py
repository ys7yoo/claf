# -*- coding: utf-8 -*-
from claf.config import args
# from claf.config.utils import set_global_seed
from claf.factory import TokenMakersFactory, DataReaderFactory
from claf.learn.experiment import Experiment
from claf.learn.mode import Mode
from claf.config.args import NestedNamespace

import torch
import numpy as np

from claf.tokens.text_handler import TextHandler
from claf.learn import utils


def predict(output_dict, context_text, helper):  # from class ReadingComprehension:
    """
    Inference by raw_feature

    * Args:
        output_dict: model's output dictionary consisting of
            - data_idx: question id
            - best_span: calculate the span_start_logits and span_end_logits to what is the best span
        arguments: arguments dictionary consisting of user_input
        helper: dictionary for helping get answer

    * Returns:
        span: predict best_span
    """
    span_start, span_end = list(output_dict["best_span"][0].data)
    word_start = span_start.item()
    word_end = span_end.item()

    text_span = helper["text_span"]
    char_start = text_span[word_start][0]
    char_end = text_span[word_end][1]

    #context_text = arguments["context"]
    answer_text = context_text[char_start:char_end]

    start_logit = output_dict["start_logits"][0]
    end_logit = output_dict["end_logits"][0]

    score = start_logit[span_start] + end_logit[span_end]
    score = score.item()

    return {"text": answer_text, "score": score}


def read_checkpoint(cuda_devices, checkpoint_path, prev_cuda_device_id=None):
    if cuda_devices == "cpu":
        return torch.load(checkpoint_path, map_location="cpu")  # use CPU

    if torch.cuda.is_available():
        checkpoint = torch.load(
            checkpoint_path,
            map_location={
                f"cuda:{prev_cuda_device_id}": f"cuda:{cuda_devices[0]}"
            },  # different cuda_device id case (save/load)
        )
    else:
        checkpoint = torch.load(checkpoint_path, map_location="cpu")  # use CPU
    return checkpoint

def get_saved_config(model_checkpoint): #, use_gpu, cuda_devices):
    saved_config_dict = model_checkpoint["config"]
    # self.config_dict = saved_config_dict

    # logger.info("Load saved_config ...")
    # logger.info(pretty_json_dumps(saved_config_dict))

    saved_config = NestedNamespace()
    saved_config.load_from_json(saved_config_dict)

    # is_use_gpu = self.config.use_gpu

    # self.config = saved_config
    # self.config.use_gpu = is_use_gpu
    # self.config.cuda_devices = cuda_devices

    return saved_config


if __name__ == "__main__":

    # load context and question from files
    with open('text/context_001.txt', 'r') as fc, open('text/question_001.txt', 'r') as fq:
        context = fc.readlines()
        question = fq.readlines()

    # set up experiment
    mode = Mode.PREDICT
    config = args.config(mode=mode)
    experiment = Experiment(mode, config)

    experiment.argument.context = ''.join(context)
    experiment.argument.question = ''.join(question)

    # set_global_seed(saved_config.seed_num)  # For Reproducible

    # set up predict mode (experiment.set_predict_mode())
    assert experiment.mode.endswith(Mode.PREDICT)

    # load model checkpoint
    cuda_devices = experiment.argument.cuda_devices
    checkpoint_path = experiment.argument.checkpoint_path
    prev_cuda_device_id = getattr(experiment.argument, "prev_cuda_device_id", None)

    # model_checkpoint = experiment._read_checkpoint(
    model_checkpoint = read_checkpoint(
        cuda_devices, checkpoint_path, prev_cuda_device_id=prev_cuda_device_id
    )

    saved_config = get_saved_config(model_checkpoint)
    saved_config.use_gpu = config.use_gpu
    saved_config.cuda_devices = cuda_devices
    
    # experiment._create_data_and_token_makers()
    token_makers = TokenMakersFactory().create(saved_config.token)
    tokenizers = token_makers["tokenizers"]
    del token_makers["tokenizers"]
    saved_config.data_reader.tokenizers = tokenizers

    data_reader = DataReaderFactory().create(saved_config.data_reader)




    # Token & Vocab
    vocabs = utils.load_vocabs(model_checkpoint)
    for token_name, token_maker in token_makers.items():
        token_maker.set_vocab(vocabs[token_name])
    text_handler = TextHandler(token_makers, lazy_indexing=False)

    # prepare raw feature and raw_to_tensor function
    raw_feature = {}
    for feature_name in data_reader.text_columns:
        feature = getattr(experiment.argument, feature_name, None)
        # if feature is None:
        # raise ValueError(f"--{feature_name} argument is required!")
        raw_feature[feature_name] = feature
    assert raw_feature is not None

    cuda_device = saved_config.cuda_devices[0] if saved_config.use_gpu else None
    raw_to_tensor_fn = text_handler.raw_to_tensor_fn(
        data_reader,
        cuda_device=cuda_device,
        helper=model_checkpoint.get("predict_helper", {})
    )
    assert raw_to_tensor_fn is not None

    # Model
    model = experiment._create_model(token_makers, checkpoint=model_checkpoint)
    model.eval()

    with torch.no_grad():
        print(raw_feature)
        tensor_feature, helper = raw_to_tensor_fn(raw_feature)
        print(tensor_feature)
        print(helper)

        output_dict = model(tensor_feature)

        for k, v in output_dict.items():
            value = v.cpu().detach().numpy()
            print(k, value)
            np.save(k, value)

        result = predict(output_dict, experiment.argument.context, helper)
        print(f"Predict: {result}")
