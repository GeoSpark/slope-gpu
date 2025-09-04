from pathlib import Path

import click
from loguru import logger
import moderngl

from geometry import calc_slope


@click.group()
def cli():
    pass


@click.command(name="info")
def info():
    ctx = moderngl.create_standalone_context(backend="egl")  # noqa
    logger.info(f'Using GPU: {ctx.info["GL_RENDERER"]}; {ctx.info["GL_VENDOR"]}; {ctx.info["GL_VERSION"]}')
    logger.info(f"Maximum tile size: {ctx.info['GL_MAX_TEXTURE_SIZE']}x{ctx.info['GL_MAX_TEXTURE_SIZE']}")


@click.command(name="slope")
@click.argument("input_file_path", type=click.Path(exists=True, path_type=Path))
@click.argument("output_file_path", type=click.Path(exists=True, path_type=Path))
@click.option("--threshold", default=10.0, help="The threshold for slope in percent.")
def slope(input_file_path: Path, output_file_path: Path, threshold: float):
    calc_slope(input_file_path, output_file_path, threshold)


if __name__ == "__main__":
    cli.add_command(info)
    cli.add_command(slope)
    cli()
