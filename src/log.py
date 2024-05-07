import argparse
import logging


def add_log_arg(parser: argparse.ArgumentParser):
    parser.add_argument(
        "--log",
        action="store",
        dest="logging_level",
        type=str,
        choices=["CRITICAL", "FATAL", "ERROR", "WARNING", "INFO", "DEBUG"],
        default="INFO",
        help="Logging level, has effect on most of the logging",
    )


def process_log_arg(namespace: argparse.Namespace):
    logging_level = getattr(logging, namespace.logging_level)
    logging.basicConfig(level=logging_level, format="[%(name)s]:%(levelname)s:%(message)s")
