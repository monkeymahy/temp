from lightning.pytorch.callbacks import Callback
from torch_ema import ExponentialMovingAverage


class EMACallback(Callback):
    def __init__(
        self,
        decay: float = 0.9999,
    ):
        super().__init__()

        self.decay = decay
        self.ema = None

    def on_train_start(self, trainer, pl_module):
        # Initialize EMA with the model's parameters
        self.ema = ExponentialMovingAverage(
            pl_module.parameters(),
            decay=self.decay,
        )

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        # Update EMA after each training step
        self.ema.update()

    def on_validation_start(self, trainer, pl_module):
        # Temporarily use EMA weights for validation
        self.ema.store()  # Store original model parameters
        self.ema.copy_to()  # Copy EMA parameters to the model

    def on_validation_end(self, trainer, pl_module):
        # Restore original model parameters after validation
        self.ema.restore()

    # def on_test_start(self, trainer, pl_module):
    #     # Similarly for testing
    #     self.ema.store()  # Store original model parameters
    #     self.ema.copy_to()  # Copy EMA parameters to the model

    # def on_test_end(self, trainer, pl_module):
    #     self.ema.restore()

    def on_save_checkpoint(self, trainer, pl_module, checkpoint):
        # Save EMA state along with the model checkpoint

        # Persist EMA internal state (useful when loading full checkpoints).
        checkpoint["ema_state_dict"] = self.ema.state_dict()

    def on_load_checkpoint(self, trainer, pl_module, checkpoint):
        # Load EMA state from the checkpoint
        ema_state = checkpoint["ema_state_dict"]

        if self.ema is None:
            # 因为现在test和fit并不是在一个进程连续执行，因此需要重新初始化EMA对象
            # Re-initialize for correct parameter mapping
            self.ema = ExponentialMovingAverage(
                pl_module.parameters(),
                decay=self.decay,
            )

        self.ema.load_state_dict(ema_state)
