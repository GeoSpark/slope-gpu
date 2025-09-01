from pathlib import Path

import click
import moderngl
import numpy as np
import rasterio
from rasterio.windows import Window
from loguru import logger

compute_shader_workgroup_size = 8

@click.group()
def cli():
    pass


@click.command(name="info")
def info():
    ctx = moderngl.create_standalone_context()
    logger.info(f'Using GPU: {ctx.info["GL_RENDERER"]}; {ctx.info["GL_VENDOR"]}; {ctx.info["GL_VERSION"]}')
    logger.info(f"Maximum tile size: {ctx.info['GL_MAX_TEXTURE_SIZE']}x{ctx.info['GL_MAX_TEXTURE_SIZE']}")


@click.command(name="slope")
@click.argument("input_file_path", type=click.Path(exists=True, path_type=Path))
@click.argument("output_file_path", type=click.Path(path_type=Path))
@click.option("--input-band", type=int, default=1, help="Band to read from input file. Default: 1")
@click.option("--overwrite/--no-overwrite", default=False, help="Overwrite output file if it already exists. Default: --no-overwrite")
def slope(input_file_path: Path, output_file_path: Path, input_band: int = 1, overwrite: bool = False):
    if output_file_path.exists():
        if overwrite:
            logger.warning(f"Output file \"{output_file_path}\" already exists, overwriting.")
            output_file_path.unlink()
        else:
            ctx = click.get_current_context()
            ctx.fail(f"Output file \"{output_file_path}\" already exists, exiting.")

    ctx = moderngl.create_standalone_context()
    logger.info(f'Using GPU: {ctx.info["GL_RENDERER"]}; {ctx.info["GL_VENDOR"]}; {ctx.info["GL_VERSION"]}')

    # Ensure we have enough VRAM for two tiles. There's no generic way of getting maximum VRAM, so we assume that the maximum texture
    # is the same size as the maximum VRAM. Because we divide both dimensions by two, we should only use at most half the VRAM. This
    # could possibly be made better by using extensions GL_NVX_gpu_memory_info or GL_ATI_meminfo, then falling back to this naieve
    # version for GPUs that don't support those extensions.
    max_tile_size = ctx.info['GL_MAX_TEXTURE_SIZE'] // 2
    logger.info(f"Maximum tile size: {ctx.info['GL_MAX_TEXTURE_SIZE']}x{ctx.info['GL_MAX_TEXTURE_SIZE']}")

    with open("algorithms/slope.glsl", "rt") as f:
        compute_shader = ctx.compute_shader(source=f.read())

    with rasterio.open(input_file_path) as src:
        logger.info(f"Loading \"{input_file_path}\" @ {src.width}x{src.height}")

        # Get some basic input file info and use the largest texture size this GPU can deal with
        # to split the file into tiles.
        img_width = src.width
        img_height = src.height
        tile_width = min(img_width, max_tile_size - 2)
        tile_height = min(img_height, max_tile_size - 2)
        workgroups_x = (tile_width + 2 + compute_shader_workgroup_size - 1) // compute_shader_workgroup_size
        workgroups_y = (tile_height + 2 + compute_shader_workgroup_size - 1) // compute_shader_workgroup_size
        tiles_x = (img_width + tile_width - 1) // tile_width
        tiles_y = (img_height + tile_height - 1) // tile_height

        logger.info(f"tile size: {tile_width}x{tile_height}")

        dst_profile = src.profile
        dst_profile.update({
            "count": 1,
            "dtype": "float32",
            "compress": "lzw"
        })

        nodata = -32767.0 if src.nodata is None else src.nodata

        # Create appropriately-sized buffers on the GPU and the CPU.
        gpu_buf_in = ctx.texture((tile_width + 2, tile_height + 2), components=1, dtype='f4')
        gpu_buf_in.bind_to_image(0, read=True, write=False)
        gpu_buf_out = ctx.texture((tile_width + 2, tile_height + 2), components=1, dtype='f4')
        gpu_buf_out.bind_to_image(1, read=False, write=True)
        cpu_buf_in = np.full((tile_height + 2, tile_width + 2), nodata, dtype=np.float32)

        logger.info(f"Writing {tiles_x}x{tiles_y} tiles to {output_file_path}")

        with rasterio.open(output_file_path, "w", **dst_profile) as dst:
            for y0 in range(0, img_height, tile_height):
                for x0 in range(0, img_width, tile_width):
                    logger.info(f"Processing tile {x0 // tile_width}:{y0 // tile_height}")

                    # Compute actual tile size (may be truncated at edges)
                    actual_tile_width = min(tile_width, img_width - x0)
                    actual_tile_height = min(tile_height, img_height - y0)

                    # Add 1-pixel overlap if possible
                    read_x0 = max(x0 - 1, 0)
                    read_y0 = max(y0 - 1, 0)
                    read_x1 = min(x0 + actual_tile_width + 1, img_width)
                    read_y1 = min(y0 + actual_tile_height + 1, img_height)

                    read_w = read_x1 - read_x0
                    read_h = read_y1 - read_y0

                    pad_l = 1 if read_x0 == 0 else 0
                    pad_t = 1 if read_y0 == 0 else 0

                    window = Window(read_x0, read_y0, read_w, read_h)
                    arr = cpu_buf_in[pad_t:read_h + pad_t, pad_l:read_w + pad_l]
                    arr[:, :] = nodata
                    src.read(input_band, window=window, out=arr)

                    gpu_buf_in.write(data=cpu_buf_in.tobytes())

                    compute_shader["no_data"] = nodata

                    # Assumes no z-rotation.
                    compute_shader["texel_size"] = [abs(src.transform.a), abs(src.transform.e)]

                    logger.info(f"Calculating slope")
                    compute_shader.run(workgroups_x, workgroups_y, 1)
                    ctx.finish()

                    logger.info(f"Writing result to {output_file_path}")
                    result = np.frombuffer(gpu_buf_out.read(), dtype=np.float32).reshape(tile_height + 2, tile_width + 2)
                    result = result[1:actual_tile_height + 1, 1:actual_tile_width + 1]
                    write_window = Window(x0, y0, actual_tile_width, actual_tile_height)
                    dst.write(result, 1, window=write_window)

    logger.info("Done")


if __name__ == "__main__":
    cli.add_command(info)
    cli.add_command(slope)
    cli()
