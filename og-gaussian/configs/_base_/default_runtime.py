weight = None  # path to model weight
resume = False  # whether to resume training process
evaluate = True
test_only = False  # test process

seed = 42  # train process will init a random seed and record
save_path = "exp/default"
num_worker = 16  # total worker in all gpu
batch_size = 16  # total batch size in all gpu
batch_size_val = None  # auto adapt to bs 1 for each gpu
batch_size_test = None  # auto adapt to bs 1 for each gpu
epoch = 100
eval_epoch = 1

sync_bn = False
enable_amp = False
empty_cache = False
find_unused_parameters = False

mix_prob = 0
param_dicts = None  # example: param_dicts = [dict(keyword="block", lr_scale=0.1)]

# hook
hooks = [
    dict(type="CheckpointLoader"),
    dict(type="IterationTimer", warmup_iter=2),
    dict(type="InformationWriter"),
    dict(type="Evaluator"),
    dict(type="CodeSaver", file_list=["pointcept", "tools"]),
    dict(type="CheckpointSaver", save_freq=None),
    dict(type="PreciseEvaluator"),
]

# Trainer
train = dict(type="DefaultTrainer")

# Tester
test = dict(type="Tester", verbose=True)
