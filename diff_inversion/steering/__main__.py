"""python -m diff_inversion.steering {extract,fit,generate,edit} --help."""

import argparse

from .config import load_config
from .experiment import edit, extract, fit, generate


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="stage", required=True)
    for stage in ("extract", "fit", "generate", "edit"):
        command = commands.add_parser(stage)
        command.add_argument("--output", required=True, help="New output directory")
        if stage != "fit":
            command.add_argument("--config", required=True, help="Experiment YAML")
        if stage == "fit":
            command.add_argument("--features", required=True)
        if stage in ("generate", "edit"):
            command.add_argument("--vectors", required=True)
            command.add_argument(
                "--layers",
                help="Override the steering layers (profile or comma-separated module prefixes)",
            )
    args = parser.parse_args()
    if args.stage == "fit":
        fit(args.features, args.output)
    elif args.stage == "extract":
        extract(load_config(args.config), args.output)
    elif args.stage == "generate":
        generate(load_config(args.config), args.vectors, args.output, args.layers)
    else:
        edit(load_config(args.config), args.vectors, args.output, args.layers)


if __name__ == "__main__":
    main()
