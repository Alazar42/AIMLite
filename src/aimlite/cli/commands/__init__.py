"""AIMLite CLI command modules."""

from aimlite.cli.commands.benchmark import run_benchmark
from aimlite.cli.commands.data import run_data_validate
from aimlite.cli.commands.doctor import run_doctor
from aimlite.cli.commands.evaluate import run_evaluate
from aimlite.cli.commands.init import run_init
from aimlite.cli.commands.install import run_install
from aimlite.cli.commands.serve import run_serve
from aimlite.cli.commands.train import run_train
from aimlite.cli.commands.update import run_update


__all__ = [
    "run_benchmark",
    "run_init",
    "run_install",
    "run_data_validate",
    "run_train",
    "run_evaluate",
    "run_serve",
    "run_doctor",
    "run_update",
]

