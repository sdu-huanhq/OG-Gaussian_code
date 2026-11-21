"""
Dataset Builder

Author: Xiaoyang Wu (xiaoyang.wu.cs@gmail.com)
Please cite our work if the code is helpful to you.
"""

from pointcept.utils.registry import Registry

DATASETS = Registry("datasets")
ERRORS = Registry("errors")


def build_dataset(cfg):
    """Build datasets."""
    return DATASETS.build(cfg)

def build_error(cfg):
    """Build errors."""
    return ERRORS.build(cfg)
