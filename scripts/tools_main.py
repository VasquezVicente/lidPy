import json
import numpy as np
import pdal
import os

class PointCloudReader:
    def __init__(self, filename):
        self.filename = os.path.abspath(filename)
        self._validate_file()
        self.metadata = None
        self.points = None
    
    def _validate_file(self):
        if not os.path.exists(self.filename):
            raise FileNotFoundError(f"File not found: {self.filename}")
        if not self.filename.lower().endswith(('.las', '.laz')):
            raise ValueError("Unsupported file type. Only .las/.laz supported.")

    def read(self, fields=None, filters=None):
        # Create the PDAL pipeline JSON
        pipeline_def = [{
            "type": "readers.las",
            "filename": self.filename
        }]

        if filters:
            pipeline_def.append({
                "type": "filters.range",
                "limits": filters
            })

        # Create the pipeline and execute
        pipeline = pdal.Pipeline(json.dumps(pipeline_def))
        pipeline.execute()

        # Get point data from the pipeline
        point_data = pipeline.arrays[0]
        return point_data

    def summary(self):
        # Generate summary metadata about the point cloud
        pipeline_json = {
            "pipeline": [self.filename]
        }
        pipeline = pdal.Pipeline(json.dumps(pipeline_json))
        pipeline.validate()  # check if pipeline is correct
        pipeline.execute()

        metadata = pipeline.metadata

        # Extracting the header information
        header = metadata.get("metadata", {}).get("pdal", {}).get("header", {})

        # Safely access 'scale' and 'offset' with a default if they are missing
        scale = header.get("scale", "Not available")
        offset = header.get("offset", "Not available")
        bounds = header.get("bounds", "Not available")

        # Add other fields to summary as needed
        summary = {
            "scale": scale,
            "offset": offset,
            "bounds": bounds
        }

        return summary

    def get_xyz(self):
        if self.points is None:
            raise RuntimeError("No points loaded. Use .read() first.")
        return np.vstack((self.points['X'], self.points['Y'], self.points['Z'])).T


def read_laz(filename, fields=None, filters=None):
    """
    Load a .laz or .las file into a NumPy structured array.

    Parameters:
    - filename: path to the .laz/.las file.
    - fields: list of field names to load (e.g., ['X', 'Y', 'Z', 'Intensity']).
    - filters: PDAL filter string (e.g., "Classification[2:2]" for ground points).

    Returns:
    - points: NumPy structured array of point data.
    - metadata: dictionary with file metadata (bounds, scale, offsets, etc.)
    """
    reader = PointCloudReader(filename)
    points = reader.read(fields=fields, filters=filters)
    return points

# Example usage within your script
filename = r"\\stri-sm01\ForestLandscapes\UAVSHARE\BCNM Lidar Raw Data\TLS\plot1\retile_-5_-5.laz"
fields = ["X", "Y", "Z"]  # Select fields to load
# Call the function to load the .laz file
points= read_laz(filename, fields=fields)

# Print a summary or inspect the data
print(f"Loaded {len(points)} points.")
print(f"First 5 points: {points[:5]}")