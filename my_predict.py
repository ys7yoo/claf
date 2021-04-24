# -*- coding: utf-8 -*-
from claf.config import args
from claf.config.utils import set_global_seed
from claf.learn.experiment import Experiment
from claf.learn.mode import Mode

import torch


def predict(model, raw_feature, raw_to_tensor_fn, arguments, interactive=False):
    model.eval()
    with torch.no_grad():
        if interactive:  # pragma: no cover
            while True:
                for k in raw_feature:
                    raw_feature[k] = utils.get_user_input(k)

                tensor_feature, helper = raw_to_tensor_fn(raw_feature)
                output_dict = model(tensor_feature)

                arguments.update(raw_feature)
                predict = model.predict(output_dict, arguments, helper)
                print(f"Predict: {pretty_json_dumps(predict)} \n")
        else:
            tensor_feature, helper = raw_to_tensor_fn(raw_feature)
            output_dict = model(tensor_feature)

            return model.predict(output_dict, arguments, helper)
        

if __name__ == "__main__":
    experiment = Experiment(Mode.PREDICT, args.config(mode=Mode.PREDICT))

    set_global_seed(experiment.config.seed_num)  # For Reproducible

    assert experiment.mode.endswith(Mode.PREDICT)
    raw_features, raw_to_tensor_fn, arguments = experiment.set_predict_mode()

    assert raw_features is not None
    assert raw_to_tensor_fn is not None
    # result = experiment.trainer.predict(
    #     raw_features,
    #     raw_to_tensor_fn,
    #     arguments,
    #     interactive=arguments.get("interactive", False),
    # )
    model = experiment.trainer.model
    result = predict(model,
                     raw_features,
                     raw_to_tensor_fn,
                     arguments,
                     interactive=arguments.get("interactive", False),
    )

    print(f"Predict: {result}")
