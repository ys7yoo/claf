# -*- coding: utf-8 -*-
from claf.config import args
# from claf.config.utils import set_global_seed
from claf.learn.experiment import Experiment
from claf.learn.mode import Mode

import torch
import numpy as np

if __name__ == "__main__":

    # load context and question from files
    with open('text/context_001.txt', 'r') as fc, open('text/question_001.txt', 'r') as fq:
        context = fc.readlines()
        question = fq.readlines()

    # set up experiment
    experiment = Experiment(Mode.PREDICT, args.config(mode=Mode.PREDICT))

    experiment.argument.context = ''.join(context)
    experiment.argument.question = ''.join(question)

    # set_global_seed(experiment.config.seed_num)  # For Reproducible

    assert experiment.mode.endswith(Mode.PREDICT)
    raw_feature, raw_to_tensor_fn, arguments = experiment.set_predict_mode()

    assert raw_feature is not None
    assert raw_to_tensor_fn is not None
    # result = experiment.trainer.predict(
    #     raw_feature,
    #     raw_to_tensor_fn,
    #     arguments,
    #     interactive=arguments.get("interactive", False),
    # )

    model = experiment.trainer.model

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

        result = model.predict(output_dict, arguments, helper)

        print(f"Predict: {result}")
