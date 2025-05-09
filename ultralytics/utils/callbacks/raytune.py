from ultralytics.utils import SETTINGS

try:
    assert SETTINGS["raytune"] is True  # verify integration is enabled
    import ray
    from ray import tune
    from ray.air import session

except (ImportError, AssertionError):
    tune = None


def on_fit_epoch_end(trainer):
    
    # If Ray Tune is available, send metrics
    if tune:
        metrics = trainer.metrics
        metrics["epoch"] = trainer.epoch
        session.report(metrics)
    else:
        # Otherwise, skip or log the metrics locally for Kaggle
        print(f"Metrics (epoch {trainer.epoch}): {trainer.metrics}")


# Define callbacks only if Ray Tune is available
callbacks = (
    {
        "on_fit_epoch_end": on_fit_epoch_end,
    }
    if tune
    else {}
)
