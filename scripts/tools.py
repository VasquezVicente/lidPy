##tools
import os
import laspy
import numpy as np
from typing import List, Tuple
from pathlib import Path
import PDAl


def cloud_check(path):
    with laspy.open(path) as f:
        header = f.header
        las_version = f"{header.version.major}.{header.version.minor}"
        point_format = header.point_format.id
        file_size = os.path.getsize(path) / (1024 * 1024)  # MB

        # Extent
        min_x, max_x = header.mins[0], header.maxs[0]
        min_y, max_y = header.mins[1], header.maxs[1]
        extent = (min_x, max_x, min_y, max_y)

        # Point count and density
        countLAS = header.point_count
        area = (max_x - min_x) * (max_y - min_y)
        density = countLAS / area if area > 0 else 0

        # CRS (if WKT is stored)
        crs = header.parse_crs()
        crs_info = crs.to_wkt() if crs else "NA"

        # Print results
        print(f"Class        : LAS (v{las_version} format {point_format})")
        print(f"Memory       : {file_size:.1f} Mb")
        print(f"Extent       : {min_x}, {max_x}, {min_y}, {max_y} (xmin, xmax, ymin, ymax)")
        print(f"Coord. ref.  : {crs_info}")
        print(f"Points       : {countLAS:,} points")
        print(f"Density      : {density:.2f} points/units²")
        print()

        # Extra dimensions
        if header.point_format.extra_dimensions:
            print("Extra Dimensions:")
            for dim in header.point_format.extra_dimensions:
                print(f"  - {dim.name}")
            print()

        # VLRs
        print("VLRs:")
        for i, vlr in enumerate(header.vlrs):
            print(f"  VLR {i}:")
            print(f"    Description       : {vlr.description}")
            print(f"    Record ID         : {vlr.record_id}")
            print(f"    User ID           : {vlr.user_id}")
            print()

