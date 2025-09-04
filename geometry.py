from pathlib import Path
import math
import os

import numpy as np
from dotenv import load_dotenv
import geopandas as gpd
from loguru import logger
import moderngl
import rasterio
from rasterio.enums import Resampling
from rasterio.features import geometry_mask, shapes
from rasterio.transform import array_bounds, from_origin
from rasterio.vrt import WarpedVRT
from rasterio.warp import calculate_default_transform, transform_bounds
from rasterio.windows import Window, from_bounds
from rasterio.io import MemoryFile
from sqlalchemy import create_engine

load_dotenv()
conn_str = f"postgresql+psycopg://{os.getenv("PGUSER")}:{os.getenv("PGPASSWORD")}@{os.getenv("PGHOST")}:{os.getenv("PGPORT")}/{os.getenv("PGDATABASE")}"
engine = create_engine(conn_str)
compute_shader_workgroup_size = 8
max_tile_size = 0
nodata = -32768.0
shaders = {}


def calc_slope(input_file_path: Path, output_file_path: Path, threshold: float):
    ctx = _init_moderngl()

    dst_profile = {
        "driver": "GTiff",
        "blockxsize": 256,
        "blockysize": 256,
        "tiled": True,
        "count": 1,
        "dtype": "float32",
        "compress": "lzw"
    }

    grid_ids = [255]

    for grid_id in grid_ids:
        logger.info(f"Fetching grid {grid_id}")
        src, mask = _get_polygons(input_file_path, grid_id)
        dst_profile["height"] = src.height
        dst_profile["width"] = src.width
        dst_profile["crs"] = src.crs
        dst_profile["transform"] = src.transform

        with MemoryFile() as memfile:
            with memfile.open(**dst_profile) as dst:
                _process(ctx, dst, src, mask, threshold)

            with memfile.open() as dst:
                slope_polygons = _polygonize(dst.read(1), dst.transform, src.crs, 90)
                slope_polygons.to_file(output_file_path / f"grid_{grid_id}.gpkg", driver="GPKG")

        logger.info("Done")


def _init_moderngl():
    global max_tile_size

    ctx = moderngl.create_standalone_context(backend="egl")  # noqa
    logger.info(f'Using GPU: {ctx.info["GL_RENDERER"]}; {ctx.info["GL_VENDOR"]}; {ctx.info["GL_VERSION"]}')

    # Ensure we have enough VRAM for two tiles. There's no generic way of getting maximum VRAM, so we assume that the maximum texture
    # is the same size as the maximum VRAM. Because we divide both dimensions by two, we should only use at most half the VRAM. This
    # could possibly be made better by using extensions GL_NVX_gpu_memory_info or GL_ATI_meminfo, then falling back to this naieve
    # version for GPUs that don't support those extensions.
    max_tile_size = ctx.info['GL_MAX_TEXTURE_SIZE'] // 2
    logger.info(f"Maximum tile size: {max_tile_size}x{max_tile_size}")

    with open("algorithms/slope.glsl", "rt") as f:
        shaders["slope"] = ctx.compute_shader(source=f.read())
        shaders["slope"]["no_data"] = nodata

    with open("algorithms/threshold.glsl", "rt") as f:
        shaders["threshold"] = ctx.compute_shader(source=f.read())
        shaders["threshold"]["no_data"] = nodata

    with open("algorithms/median_filter.glsl", "rt") as f:
        shaders["median_filter"] = ctx.compute_shader(source=f.read())
        shaders["median_filter"]["no_data"] = nodata

    with open("algorithms/dilation_filter.glsl", "rt") as f:
        shaders["dilation_filter"] = ctx.compute_shader(source=f.read())
        shaders["dilation_filter"]["no_data"] = nodata

    with open("algorithms/erosion_filter.glsl", "rt") as f:
        shaders["erosion_filter"] = ctx.compute_shader(source=f.read())
        shaders["erosion_filter"]["no_data"] = nodata

    with open("algorithms/mask_zeros.glsl", "rt") as f:
        shaders["mask_zeros"] = ctx.compute_shader(source=f.read())
        shaders["mask_zeros"]["no_data"] = nodata

    return ctx


def _process(ctx: moderngl.Context, dst, src, mask, threshold: float):
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

    # Create appropriately-sized buffers on the GPU and the CPU.
    gpu_buf_a = ctx.texture((tile_width + 2, tile_height + 2), components=1, dtype='f4')
    gpu_buf_b = ctx.texture((tile_width + 2, tile_height + 2), components=1, dtype='f4')
    cpu_buf = np.full((tile_height + 2, tile_width + 2), nodata, dtype=np.float32)

    logger.info(f"Writing {tiles_x}x{tiles_y} tiles")

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
            arr = cpu_buf[pad_t:read_h + pad_t, pad_l:read_w + pad_l]
            src.read(1, window=window, out=arr)
            w_mask = mask[read_y0:read_y1, read_x0:read_x1]
            arr[w_mask] = src.nodata

            gpu_buf_a.write(data=cpu_buf.tobytes())

            # ----------------
            logger.info("Calculating slope")
            gpu_buf_a.bind_to_image(0, read=True, write=False)
            gpu_buf_b.bind_to_image(1, read=False, write=True)
            # Assumes no z-rotation.
            shaders["slope"]["texel_size"] = [abs(src.transform.a), abs(src.transform.e)]
            shaders["slope"].run(workgroups_x, workgroups_y, 1)
            ctx.finish()
            ctx.memory_barrier()
            # ----------------
            logger.info(f"Thresholding slope <{threshold}%")
            # Swap buffers.
            gpu_buf_b.bind_to_image(0, read=True, write=False)
            gpu_buf_a.bind_to_image(1, read=False, write=True)
            shaders["threshold"]["threshold"] = threshold
            shaders["threshold"].run(workgroups_x, workgroups_y, 1)
            ctx.finish()
            ctx.memory_barrier()
            # ----------------
            logger.info("Median filter")
            # Swap buffers.
            gpu_buf_a.bind_to_image(0, read=True, write=False)
            gpu_buf_b.bind_to_image(1, read=False, write=True)
            shaders["median_filter"].run(workgroups_x, workgroups_y, 1)
            ctx.finish()
            ctx.memory_barrier()
            # ----------------
            logger.info("Closing (Dilation filter)")
            # Swap buffers.
            gpu_buf_b.bind_to_image(0, read=True, write=False)
            gpu_buf_a.bind_to_image(1, read=False, write=True)
            shaders["dilation_filter"].run(workgroups_x, workgroups_y, 1)
            ctx.finish()
            ctx.memory_barrier()
            # ----------------
            logger.info("Closing (Erosion filter)")
            # Swap buffers.
            gpu_buf_a.bind_to_image(0, read=True, write=False)
            gpu_buf_b.bind_to_image(1, read=False, write=True)
            shaders["erosion_filter"].run(workgroups_x, workgroups_y, 1)
            ctx.finish()
            ctx.memory_barrier()
            # ----------------
            logger.info("Masking zeros")
            # Swap buffers.
            gpu_buf_b.bind_to_image(0, read=True, write=False)
            gpu_buf_a.bind_to_image(1, read=False, write=True)
            shaders["mask_zeros"].run(workgroups_x, workgroups_y, 1)
            ctx.finish()
            ctx.memory_barrier()
            # ----------------

            logger.info(f"Storing result")
            result = np.frombuffer(gpu_buf_a.read(), dtype=np.float32).reshape(tile_height + 2, tile_width + 2)
            result = result[1:actual_tile_height + 1, 1:actual_tile_width + 1]
            write_window = Window(x0, y0, actual_tile_width, actual_tile_height)

            dst.write(result, 1, window=write_window)


def _polygonize(mask_bool, transform, crs, min_area_m2):
    logger.info("Polygonizing")

    feats = (
        {"properties": {}, "geometry": geom}
        for geom, val in shapes(mask_bool, transform=transform, connectivity=8)
        if val == 1
    )

    gdf = gpd.GeoDataFrame.from_features(feats, crs=crs)

    logger.info("Removing small polygons")

    if not gdf.empty:
        gdf = gdf[gdf.geometry.area >= min_area_m2]

    return gdf


def _get_polygons(input_file_path: Path, grid_id: int) -> tuple[WarpedVRT, np.ndarray]:
    sql = f'SELECT geom FROM "phase2"."land" WHERE grid_id=%(grid_id)s and subtype=%(subtype)s and class=%(class)s'
    gdf_mask = gpd.read_postgis(sql, con=engine, geom_col="geom", params={"grid_id": grid_id, "subtype": "land", "class": "land"})

    bounds = gdf_mask.total_bounds

    with rasterio.open(input_file_path) as src:
        logger.info(f"Loading \"{src.name}\" @ {src.width}x{src.height}")

        xmin_3832, ymin_3832, xmax_3832, ymax_3832 = bounds
        xres, yres = _suggest_resolution_from_src_window(src, gdf_mask.crs, bounds)

        border_px = 1
        xmin_pad = xmin_3832 - border_px * xres
        xmax_pad = xmax_3832 + border_px * xres
        ymin_pad = ymin_3832 - border_px * yres
        ymax_pad = ymax_3832 + border_px * yres

        # Build the exact target grid
        width  = max(1, int(math.ceil((xmax_pad - xmin_pad) / xres)))
        height = max(1, int(math.ceil((ymax_pad - ymin_pad) / yres)))
        dst_transform = from_origin(xmin_pad, ymax_pad, xres, yres)

        vrt = WarpedVRT(
            src,
            crs=gdf_mask.crs,
            transform=dst_transform,
            width=width,
            height=height,
            resampling=Resampling.bilinear,
            src_nodata=src.nodata,
            nodata=src.nodata,
            dtype="float32",
        )

        outside_mask = geometry_mask(
            geometries=gdf_mask.geometry,
            out_shape=(height, width),
            transform=dst_transform,
            invert=False,  # False => mask=True outside the shapes
            all_touched=True
        )

    return vrt, outside_mask

def _suggest_resolution_from_src_window(src, dst_crs, bbox_3832):
    xmin_3832, ymin_3832, xmax_3832, ymax_3832 = bbox_3832

    # 1) Convert the target bbox -> source CRS
    xmin_s, ymin_s, xmax_s, ymax_s = transform_bounds(dst_crs, src.crs,
                                                      xmin_3832, ymin_3832, xmax_3832, ymax_3832,
                                                      densify_pts=21)

    # 2) Build a source window covering that area, then derive its exact bounds
    win = from_bounds(xmin_s, ymin_s, xmax_s, ymax_s, src.transform).round_offsets().round_lengths()
    w = max(1, int(win.width))
    h = max(1, int(win.height))
    src_win_transform = rasterio.windows.transform(win, src.transform)
    left, bottom, right, top = array_bounds(h, w, src_win_transform)

    # 3) Let GDAL pick a default target transform for *this* source window
    dst_transform, dst_w, dst_h = calculate_default_transform(
        src_crs=src.crs, dst_crs=dst_crs,
        width=w, height=h,
        left=left, bottom=bottom, right=right, top=top
    )

    # Extract pixel sizes
    xres = abs(dst_transform.a)
    yres = abs(dst_transform.e)

    return xres, yres
