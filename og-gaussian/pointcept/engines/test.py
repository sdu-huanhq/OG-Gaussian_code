import os
from collections import OrderedDict
import torch
import torch.utils.data

from .defaults import create_ddp_model
import pointcept.utils.comm as comm
from pointcept.datasets import build_dataset
from pointcept.models import build_model
from pointcept.utils.logger import get_root_logger
from pointcept.utils.registry import Registry
from pointcept.utils.misc import make_dirs


TESTERS = Registry("testers")


@TESTERS.register_module()
class Tester:
    def __init__(
        self, 
        cfg, 
        model=None, 
        test_loader=None,
        apply_SVD=True, 
        verbose=False
    ) -> None:
        if comm.get_world_size() != 1:
            raise RuntimeError(
                "Tester supports single-GPU testing only. "
                "Run tools/test.py with --num-gpus 1 --num-machines 1."
            )
        torch.multiprocessing.set_sharing_strategy("file_system")
        self.logger = get_root_logger(
            log_file=os.path.join(cfg.save_path, f"test_{cfg.data.test.test_mode}.log"),
            file_mode="a" if cfg.resume else "w",
        )
        self.logger.info("=> Loading config ...")
        self.cfg = cfg
        self.apply_SVD = apply_SVD
        self.verbose = verbose
        if self.verbose:
            self.logger.info(f"Save path: {cfg.save_path}")
            self.logger.info(f"Config:\n{cfg.pretty_text}")
        if model is None:
            self.logger.info("=> Building model ...")
            self.model = self.build_model()
        else:
            self.model = model
        if test_loader is None:
            self.logger.info("=> Building test dataset & dataloader ...")
            self.test_loader = self.build_test_loader()
        else:
            self.test_loader = test_loader

    def build_model(self):
        model = build_model(self.cfg.model)
        n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
        self.logger.info(f"Num params: {n_parameters}")
        model = create_ddp_model(
            model.cuda(),
            broadcast_buffers=False,
            find_unused_parameters=self.cfg.find_unused_parameters,
        )
        if os.path.isfile(self.cfg.weight):
            self.logger.info(f"Loading weight at: {self.cfg.weight}")
            checkpoint = torch.load(self.cfg.weight)
            weight = OrderedDict()
            for key, value in checkpoint["state_dict"].items():
                if key.startswith("module."):
                    if comm.get_world_size() == 1:
                        key = key[7:]  # module.xxx.xxx -> xxx.xxx
                else:
                    if comm.get_world_size() > 1:
                        key = "module." + key  # xxx.xxx -> module.xxx.xxx
                weight[key] = value
            model.load_state_dict(weight, strict=True)
            self.logger.info(
                "=> Loaded weight '{}' (epoch {})".format(
                    self.cfg.weight, checkpoint["epoch"]
                )
            )
        else:
            raise RuntimeError("=> No checkpoint found at '{}'".format(self.cfg.weight))
        return model

    def build_test_loader(self):
        test_dataset = build_dataset(self.cfg.data.test)
        test_loader = torch.utils.data.DataLoader(
            test_dataset,
            batch_size=self.cfg.batch_size_test_per_gpu,
            shuffle=False,
            num_workers=self.cfg.batch_size_test_per_gpu,
            pin_memory=True,
            collate_fn=self.collate_fn,
        )
        return test_loader


    def test(self):
        assert self.test_loader.batch_size == 1
        logger = get_root_logger()
        logger.info(">>>>>>>>>>>>>>>> Start Evaluation >>>>>>>>>>>>>>>>")

        save_path = os.path.join(self.cfg.save_path, "result")
        make_dirs(save_path)
        comm.synchronize()
        record = dict(
            error_ave=0,
            error_min=1e6,
            error_max=0,
        )
        
        self.model.eval()
        # fragment inference
        for i, input_dict in enumerate(self.test_loader):
            input_dict = input_dict[0]
            # to cuda
            for key in input_dict.keys():
                if isinstance(input_dict[key], torch.Tensor):
                    input_dict[key] = input_dict[key].cuda(non_blocking=True)

            # predict
            with torch.no_grad():
                output_dict = self.model(
                    input_dict, mode="sample_goal", apply_SVD=self.apply_SVD)
                if self.cfg.empty_cache:
                    torch.cuda.empty_cache()
            
            error = output_dict['error'].cpu().numpy()

            logger.info(
                "Test: {}/{} >>> {} >>> Error: {}".format(
                    i + 1, len(self.test_loader), input_dict['sample_name'], error
                )
            )

            record['error_ave'] = (record['error_ave'] * i + error) / (i+1)
            record['error_min'] = min(record['error_min'], error)
            record['error_max'] = max(record['error_max'], error)

        logger.info("Syncing ...")
        comm.synchronize()

        if comm.is_main_process():
            logger.info(
                "Test result: Ave/Min/Max {:.6f}/{:.6f}/{:.6f}".format(
                    record['error_ave'], record['error_min'], record['error_max']
                )
            )
            logger.info("<<<<<<<<<<<<<<<<< End Evaluation <<<<<<<<<<<<<<<<<")

    @staticmethod
    def collate_fn(batch):
        return batch
