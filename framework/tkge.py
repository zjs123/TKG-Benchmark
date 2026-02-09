import argparse
import os

from tkge.task.task import Task
from tkge.task.train_task import TrainTask
from tkge.task.eval_task import EvaluateTask
from tkge.task.resume_task import ResumeTask
from tkge.common.config import Config

desc = 'Temporal KG Completion methods'
parser = argparse.ArgumentParser(description=desc)
parser.add_argument(
    "--version",
    "-V",
    action="version",
    version=f"work in progress"
)

# parser.add_argument('train', help='task type', type=bool)
# parser.add_argument('--config', help='configuration file', type=str)
# parser.add_argument('--help', help='help')

subparsers = parser.add_subparsers(title="task",
                                   description="valid tasks: train, eval",
                                   dest="task")

# subparser train
parser_train = TrainTask.parse_arguments(subparsers)
parser_eval = EvaluateTask.parse_arguments(subparsers)
parser_resume = ResumeTask.parse_arguments(subparsers)

args = parser.parse_args()

task_dict = {
    'train': TrainTask,
}

config_path = args.config
config = Config.create_from_yaml(config_path)  # TODO load_default is false

# Initialize working folder
if args.task in ['train']:
    config.create_experiment()
else:
    config.restore_experiment(args.experiment)

kwargs = {}
task = task_dict[args.task](config, **kwargs)
task.main()

# trainer = TrainTask(config)
# tester = TestTask(config)
