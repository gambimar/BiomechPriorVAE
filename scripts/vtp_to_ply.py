"""Convert OpenSim VTP geometry to the adjacent ``*.vtp.ply`` files."""

import argparse
from pathlib import Path


def convert_vtp_to_ply(geometry_path: str | Path) -> None:
    """Convert each VTP file, preserving existing PLY files."""
    import vtk

    for vtp_file in Path(geometry_path).glob("*.vtp"):
        ply_file = vtp_file.with_suffix(".vtp.ply")
        if ply_file.exists():
            continue
        print(f"Converting {vtp_file.name} to PLY...")
        reader = vtk.vtkXMLPolyDataReader()
        reader.SetFileName(str(vtp_file))
        reader.Update()
        writer = vtk.vtkPLYWriter()
        writer.SetFileName(str(ply_file))
        writer.SetInputConnection(reader.GetOutputPort())
        if writer.Write() != 1:
            raise OSError(f"Failed to write {ply_file}")
        print(f"Created: {ply_file.name}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("geometry_path", type=Path)
    convert_vtp_to_ply(parser.parse_args().geometry_path)
