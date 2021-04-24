# -*- coding: utf-8 -*-


from claf.config import args
from claf.config.utils import set_global_seed
from claf.learn.experiment import Experiment
from claf.learn.mode import Mode


if __name__ == "__main__":
    experiment = Experiment(Mode.PREDICT, args.config(mode=Mode.PREDICT))

    set_global_seed(experiment.config.seed_num)  # For Reproducible

    assert experiment.mode.endswith(Mode.PREDICT)
    raw_features, raw_to_tensor_fn, arguments = experiment.set_predict_mode()

    assert raw_features is not None
    assert raw_to_tensor_fn is not None
    result = experiment.trainer.predict(
        raw_features,
        raw_to_tensor_fn,
        arguments,
        interactive=arguments.get("interactive", False),
    )

    print(f"Predict: {result}")
