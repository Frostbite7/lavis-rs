from dinov2.eval.setup import setup_and_build_model
import argparse
import logging
import torch.nn as nn
from lavis.models.eva_vit import convert_weights_to_fp16


class DINOV2PretrainedModel(nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        args_parser = argparse.ArgumentParser()
        args = args_parser.parse_args()
        args.config_file = "/NAS6/Members/linchenxi/projects/DINOV2/model8/config.yaml"
        args.pretrained_weights = "/NAS6/Members/linchenxi/projects/DINOV2/model8/eval/training_24999/teacher_checkpoint.pth"
        args.output_dir = ""
        args.opts = []
        model, _, config = setup_and_build_model(args)
        self.model = model
        self.config = config
        self.num_features = 768

    def forward(self, input):
        return self.model(input, is_training=True)


def create_custom_vit(precision='fp16'):
    vit = DINOV2PretrainedModel()
    if precision == "fp16":
        convert_weights_to_fp16(vit.model)
    return vit

# def create_custom_vit(precision='fp16'):
#     description = "Load DINOv2 Pretrained Model"
#     args_parser = argparse.ArgumentParser()
#     args = args_parser.parse_args()
#     args.config_file = "/NAS6/Members/linchenxi/projects/DINOV2/model8/config.yaml"
#     args.pretrained_weights = "/NAS6/Members/linchenxi/projects/DINOV2/model8/eval/training_24999/teacher_checkpoint.pth"
#     args.output_dir = ""
#     args.opts = []
#     model, autocast_dtype = setup_and_build_model(args)
#     logger = logging.getLogger()
#     logger.info("Success!")
#
#     if precision == "fp16":
#         convert_weights_to_fp16(model)
#     return model
