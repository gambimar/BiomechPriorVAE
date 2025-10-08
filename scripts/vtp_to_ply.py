import vtk
import os
import glob

def convert_vtp_to_ply(geometry_path):
    vtp_files = glob.glob(os.path.join(geometry_path, "*.vtp"))
    
    for vtp_file in vtp_files:
        ply_file = vtp_file + ".ply"
        
        if os.path.exists(ply_file):
            continue
        
        print(f"Converting {os.path.basename(vtp_file)} to PLY...")
        
        # 读取 VTP 文件
        reader = vtk.vtkXMLPolyDataReader()
        reader.SetFileName(vtp_file)
        reader.Update()
        
        # 写入 PLY 文件
        writer = vtk.vtkPLYWriter()
        writer.SetFileName(ply_file)
        writer.SetInputConnection(reader.GetOutputPort())
        writer.Write()
        
        print(f"Created: {os.path.basename(ply_file)}")

geometry_path = "/home/kishgard/projects/BiomechPriorVAE/data/model/Geometry"
convert_vtp_to_ply(geometry_path)