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
from sqlalchemy import create_engine, text

load_dotenv()
conn_str = f'postgresql+psycopg://{os.getenv("PGUSER")}:{os.getenv("PGPASSWORD")}@{os.getenv("PGHOST")}:{os.getenv("PGPORT")}/{os.getenv("PGDATABASE")}'
engine = create_engine(conn_str)
compute_shader_workgroup_size = 8
overlap = 4
halo = 1
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
        "compress": "lzw",
        "nodata": nodata,
    }

    srtm_vrt = rasterio.open(input_file_path)
    logger.info(f"Loading \"{srtm_vrt.name}\" @ {srtm_vrt.width}x{srtm_vrt.height}")

    sql = 'SELECT DISTINCT grid_id FROM "phase2"."land_tiles"'
    with engine.connect() as conn:
        grid_ids = list(conn.execute(text(sql)))

    # Unravel the list of tuples into a simple tuple.
    grid_ids = list(zip(*grid_ids))[0]
    # grid_ids = [255, 270]

    for grid_id in grid_ids:
        logger.info(f"Fetching grid {grid_id}")

        with _get_polygons(srtm_vrt, grid_id).open() as src:
            dst_profile["height"] = src.height
            dst_profile["width"] = src.width
            dst_profile["crs"] = src.crs
            dst_profile["transform"] = src.transform

            with rasterio.open(output_file_path / f"grid_{grid_id}.tif", "w", **dst_profile) as dst:
                _process(ctx, dst, src, threshold)

            # with MemoryFile() as memfile:
            #     with memfile.open(**dst_profile) as dst:
            #         _process(ctx, dst, src, threshold)

                # with memfile.open() as dst:
                #     slope_polygons = _polygonize(dst.read(1), dst.transform, src.crs, 90)
                #     slope_polygons.to_file(output_file_path / f"grid_{grid_id}.gpkg", driver="GPKG")

            logger.info("Done")

    srtm_vrt.close()


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


def _process(ctx: moderngl.Context, dst, src, threshold: float):
    # Get some basic input file info and use the largest texture size this GPU can deal with
    # to split the file into tiles.
    workgroups_x = (max_tile_size + compute_shader_workgroup_size - 1) // compute_shader_workgroup_size
    workgroups_y = (max_tile_size + compute_shader_workgroup_size - 1) // compute_shader_workgroup_size
    tiles_x = (src.width + max_tile_size - 1) // max_tile_size
    tiles_y = (src.height + max_tile_size - 1) // max_tile_size

    logger.info(f"tile size: {max_tile_size}x{max_tile_size}")

    # Create appropriately-sized buffers on the GPU and the CPU.
    gpu_buf_a = ctx.texture((max_tile_size, max_tile_size), components=1, dtype='f4')
    gpu_buf_b = ctx.texture((max_tile_size, max_tile_size), components=1, dtype='f4')
    cpu_buf = np.full((max_tile_size, max_tile_size), nodata, dtype=np.float32)

    logger.info(f"Writing {tiles_x}x{tiles_y} tiles")

    # Because maths is hard, we're keeping track of the currently processed tile the dumb way.
    tx = 0
    ty = 0

    for y0 in range(-overlap, src.height, max_tile_size - (overlap * 2)):
        tx = 0

        for x0 in range(-overlap, src.width, max_tile_size - (overlap * 2)):
            logger.info(f"Processing tile {tx}:{ty}")
            tx += 1

            window = Window(x0, y0, max_tile_size, max_tile_size)
            src.read(1, window=window, out=cpu_buf, boundless=True, masked=True, fill_value=nodata)

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
            result = np.frombuffer(gpu_buf_a.read(), dtype=np.float32).reshape((max_tile_size, max_tile_size))
            x = x0 + overlap
            y = y0 + overlap
            w = min(max_tile_size - overlap, src.width - x)
            h = min(max_tile_size - overlap, src.height - y)
            write_window = Window(x, y, w, h)
            result = result[overlap:overlap + h, overlap:overlap + w]
            dst.write(result, 1, window=write_window)

        ty += 1


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


def _get_polygons(src, grid_id: int) -> MemoryFile:
    sql = f'SELECT geom FROM "phase2"."land_tiles" WHERE grid_id=%(grid_id)s'
    gdf_mask = gpd.read_postgis(sql, con=engine, geom_col="geom", params={"grid_id": grid_id})

    bounds = gdf_mask.total_bounds

    xmin_3832, ymin_3832, xmax_3832, ymax_3832 = bounds
    xres, yres = _suggest_resolution_from_src_window(src, gdf_mask.crs, bounds)

    xmin_pad = xmin_3832 - overlap * xres
    xmax_pad = xmax_3832 + overlap * xres
    ymin_pad = ymin_3832 - overlap * yres
    ymax_pad = ymax_3832 + overlap * yres

    # Build the exact target grid
    width  = max(1, int(math.ceil((xmax_pad - xmin_pad) / xres)))
    height = max(1, int(math.ceil((ymax_pad - ymin_pad) / yres)))
    dst_transform = from_origin(xmin_pad, ymax_pad, xres, yres)

    logger.info(f"Extracting SRTM data")

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

    data = vrt.read(1)

    logger.info(f"Masking SRTM data")

    mask = geometry_mask(
        geometries=gdf_mask.geometry,
        out_shape=(height, width),
        transform=dst_transform,
        invert=False,  # False => mask=True outside the shapes
        all_touched=True
    )

    data[mask] = src.nodata

    profile = vrt.profile
    profile["driver"] = "GTiff"
    profile["tiled"] = True
    profile["blockxsize"] = 256
    profile["blockysize"] = 256

    masked_data = MemoryFile()
    with masked_data.open(**profile) as memfile:
        memfile.write(data, 1)

    return masked_data


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
