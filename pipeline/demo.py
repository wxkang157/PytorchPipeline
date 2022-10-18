
from shutil import copyfile
from pathlib import Path
import datetime
from common.util import yaml_parser


def init_setting(cfg):
    timestr = str(datetime.datetime.now().strftime('%Y_%m%d_%H%M'))
    experiment_dir = Path(cfg.GLOBAL['save_result_dir'])
    experiment_dir.mkdir(exist_ok=True)     # 保存实验结果的总目录
    experiment_dir = experiment_dir.joinpath(cfg.GLOBAL['experiment_name'])
    experiment_dir.mkdir(exist_ok=True)
    experiment_dir = experiment_dir.joinpath(timestr)
    experiment_dir.mkdir(exist_ok=True)     # 每次实验的根目录
    checkpoints_dir = experiment_dir.joinpath('checkpoints/')
    checkpoints_dir.mkdir(exist_ok=True)    # 保存模型的目录
    log_dir = experiment_dir.joinpath('logs/')
    log_dir.mkdir(exist_ok=True)    # 保存日志的目录
    setting_dir = experiment_dir.joinpath('setting/')
    setting_dir.mkdir(exist_ok=True)  # 保存日志的目录

    copyfile('data/myClassDataset.py', str(setting_dir) + '/myClassDataset.py')
    copyfile('config/my.yaml', str(setting_dir) + '/my.yaml')
    copyfile('loss/build_loss.py', str(setting_dir) + '/build_loss.py')
    copyfile('model/build_model.py', str(setting_dir) + '/build_model.py')
    copyfile('train.py', str(setting_dir) + '/train.py')
    copyfile('test.py', str(setting_dir) + '/test.py')
    copyfile('val.py', str(setting_dir) + '/val.py')

    return experiment_dir, checkpoints_dir, log_dir


if __name__ == "__main__":
    cfg = yaml_parser('config/my.yaml') #解析配置文件
    init_setting(cfg)

