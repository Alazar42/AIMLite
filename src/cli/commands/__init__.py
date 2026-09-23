"""AIMLite CLI command modules."""

from cli.commands.benchmark import run_benchmark
from cli.commands.data import run_data_validate
from cli.commands.doctor import run_doctor
from cli.commands.evaluate import run_evaluate
from cli.commands.init import run_init
from cli.commands.install import run_install
from cli.commands.serve import run_serve
from cli.commands.train import run_train


__all__ = [
    "run_benchmark",
    "run_init",
    "run_install",
    "run_data_validate",
    "run_train",
    "run_evaluate",
    "run_serve",
    "run_doctor",
]

