# -*- coding: utf-8 -*-
from claf.config import args
# from claf.config.utils import set_global_seed
from claf.learn.experiment import Experiment
from claf.learn.mode import Mode

import torch

if __name__ == "__main__":
    experiment = Experiment(Mode.PREDICT, args.config(mode=Mode.PREDICT))

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
        tensor_feature, helper = raw_to_tensor_fn(raw_feature)
        output_dict = model(tensor_feature)

        result = model.predict(output_dict, arguments, helper)

        print(f"Predict: {result}")
