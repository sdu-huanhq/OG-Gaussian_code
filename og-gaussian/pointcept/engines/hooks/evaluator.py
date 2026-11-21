"""
Evaluate Hook

Author: Xiaoyang Wu (xiaoyang.wu.cs@gmail.com)
Please cite our work if the code is helpful to you.
"""

import torch

import pointcept.utils.comm as comm

from .default import HookBase
from .builder import HOOKS


@HOOKS.register_module()
class Evaluator(HookBase):
    def after_epoch(self):
        if self.trainer.cfg.evaluate and (self.trainer.epoch+1) % self.trainer.cfg.val_freq == 0:
            self.eval()

    def eval(self):
        self.trainer.logger.info(">>>>>>>>>>>>>>>> Start Evaluation >>>>>>>>>>>>>>>>")
        self.trainer.model.eval()
        error = None
        for i, input_dict in enumerate(self.trainer.val_loader):
            input_dict = comm.dict_to_gpu(input_dict)
            with torch.no_grad():
                output_dict = self.trainer.model(input_dict)    # dict(loss)
            
            loss = output_dict["loss"]
            self.trainer.storage.put_scalar("val_loss", loss.item())
            self.trainer.logger.info(
                "Test: [{iter}/{max_iter}] Loss {loss:.4f} ".format(
                    iter=i + 1, max_iter=len(self.trainer.val_loader), loss=loss.item()
                )
            )

            if i == len(self.trainer.val_loader)-1 and (self.trainer.epoch+1) % self.trainer.cfg.evaluate_freq == 0:
                with torch.no_grad():
                    output_dict = self.trainer.model(input_dict, mode="sample_goal")
                error = output_dict["error"]
                self.trainer.storage.put_scalar("val_error", error.item())
                self.trainer.logger.info(
                    "Error {error:.4f} ".format(error=error.item())
                )

        loss_avg = self.trainer.storage.history("val_loss").avg
        current_epoch = self.trainer.epoch + 1
        if self.trainer.writer is not None:
            self.trainer.writer.add_scalar("val/loss", loss_avg, current_epoch)
            if error is not None:
                self.trainer.writer.add_scalar("val/error", error, current_epoch)
        self.trainer.logger.info("<<<<<<<<<<<<<<<<< End Evaluation <<<<<<<<<<<<<<<<<")
