import sys, os
sys.path.append(os.curdir)

from pointcept.engines.defaults import (
    default_argument_parser,
    default_config_parser,
    default_setup,
)
from pointcept.engines.test import TESTERS
from pointcept.engines.launch import launch


def main_worker(cfg):
    cfg = default_setup(cfg)
    tester = TESTERS.build(dict(type=cfg.test.type, cfg=cfg))
    tester.test()


def main():
    parser = default_argument_parser()
    args = parser.parse_args()
    if args.num_gpus != 1 or args.num_machines != 1:
        parser.error("Testing supports only --num-gpus 1 --num-machines 1.")
    cfg = default_config_parser(args.config_file, args.options)

    launch(
        main_worker,
        num_gpus_per_machine=args.num_gpus,
        num_machines=args.num_machines,
        machine_rank=args.machine_rank,
        dist_url=args.dist_url,
        cfg=(cfg,),
    )


if __name__ == "__main__":
    main()


"""
cmd:
python tools/test.py \
    --config-file configs/kettle.py \
    --num-gpus 1 \
    --options save_path=exp/kettle \
    weight=exp/kettle/model/model_last.pth
"""
