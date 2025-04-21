import os
import shutil
import subprocess

import pyvista as pv
import torch
import yaml
from PIL import Image
from pytorch3d.io import load_obj
from pytorch3d.renderer import (
    FoVPerspectiveCameras,
    RasterizationSettings,
    MeshRenderer,
    MeshRasterizer,
    SoftPhongShader,
    PointLights,
    look_at_view_transform
)
from pytorch3d.renderer import TexturesVertex
from pytorch3d.structures import Meshes
from tqdm import tqdm

from datasets import read_label_map


# 解压tar.gz文件
def extract_tar_gz(file_path, extract_path):
    print(f'Extracting {file_path.split("/")[-1]} file')
    if os.path.exists(extract_path):
        shutil.rmtree(extract_path)
    os.makedirs(extract_path, exist_ok=True)
    subprocess.call(['tar', '-xzf', file_path, '-C', extract_path])
    print(f'Extracting {file_path.split("/")[-1]} files to {extract_path} directory completed')
# vtk格式文件转off格式文件
def vtk_to_off(vtk_file, off_file):
    # 读取 VTK 文件
    mesh = pv.read(vtk_file)
    # 获取顶点
    points = mesh.points
    # 获取面索引
    faces = mesh.faces.reshape(-1, 4)[:, 1:]  # VTK 面格式包含面点数，去掉第一列
    # 写入 OFF 文件
    with open(off_file, "w") as f:
        f.write("OFF\n")
        f.write(f"{len(points)} {len(faces)} 0\n")
        # 写入顶点
        for p in points:
            f.write(f"{p[0]} {p[1]} {p[2]}\n")
        # 写入面
        for face in faces:
            f.write(f"3 {face[0]} {face[1]} {face[2]}\n")
# vtk格式文件转obj格式文件
def vtk_to_obj(vtk_file, obj_file):
    # 读取 VTK 文件
    mesh = pv.read(vtk_file)
    # 使用 PyVista 直接导出 OBJ 文件
    mesh.save(obj_file)
# 遍历目录
def traverse_files(path, dirs):
    for file in os.listdir(path):
        file_path = os.path.join(path, file)
        if os.path.isfile(file_path):
            dirs.append(file_path)
        elif os.path.isdir(file_path):
            traverse_files(file_path, dirs)
def convert_file_format(data_path, off_data_path, obj_data_path):
    data_dir = data_path.replace('.tar', '').replace('.gz', '')
    # off_data_dir = off_data_path +  data_dir.split('/')[-1]
    obj_data_dir = obj_data_path +  data_dir.split('/')[-1]
    # os.makedirs(off_data_dir, exist_ok=True)
    os.makedirs(obj_data_dir, exist_ok=True)
    extract_tar_gz(data_path, data_dir)
    vtk_paths = []
    traverse_files(data_dir, vtk_paths)
    for i in tqdm(range(len(vtk_paths)), desc='convert_file_format'):
        # off_data_file = off_data_dir + '/' + vtk_paths[i].split('/')[-1].split('.')[0] + ".off"
        obj_data_file = obj_data_dir + '/' + vtk_paths[i].split('/')[-1].split('.')[0] + ".obj"
        # vtk_to_off(vtk_paths[i], off_data_file)
        vtk_to_obj(vtk_paths[i], obj_data_file)

def load_obj_with_color(obj_filename, device):
    verts, faces, _ = load_obj(obj_filename)
    faces_idx = faces.verts_idx

    # 将模型中心平移到原点
    center = verts.mean(0)
    verts = verts - center

    # 将模型尺寸统一到 [-1, 1]
    scale = (verts.abs().max())  # 最大绝对值（让所有点落在 -1 到 1）
    verts = verts / scale

    # 设置每个顶点颜色为灰色
    verts_rgb = torch.ones_like(verts)[None] * torch.tensor([0.7, 0.7, 0.7])
    mesh = Meshes(verts=[verts.to(device)],
                  faces=[faces_idx.to(device)],
                  textures=TexturesVertex(verts_features=verts_rgb.to(device)))
    return mesh
def render_obj_multiview(obj_path, out_dir, num_views=8, image_size=256, device='cuda'):
    # 加载 mesh（无纹理）
    mesh = load_obj_with_color(obj_path, device)
    # 多角度相机位置
    R, T = look_at_view_transform(dist=2.0, elev=20, azim=torch.linspace(0, 360, num_views), device=device)
    # 灯光和着色器
    lights = PointLights(device=device, location=[[2.0, 2.0, 2.0]])
    raster_settings = RasterizationSettings(image_size=image_size)
    file_order = 0
    cameras = FoVPerspectiveCameras(device=device, R=R, T=T)
    renderer = MeshRenderer(
        rasterizer=MeshRasterizer(cameras=cameras, raster_settings=raster_settings),
        shader=SoftPhongShader(device=device, cameras=cameras, lights=lights)
    )
    # 批量渲染
    images = renderer(mesh.extend(num_views), cameras=cameras, lights=lights)
    # 保存图像
    for i, image in enumerate(images):
        file_order += 1
        img = (image[..., :3] * 255).cpu().byte().numpy()
        im = Image.fromarray(img)
        im.save(os.path.join(out_dir, f'{out_dir.split("/")[-1]}_{out_dir.split("/")[-2]}_view_{file_order:03d}.png'))
def pre_deal(split='train'):
    classification_image_num = {}
    file_label = {}
    split = split.upper()
    # 读取配置
    with open('../config/vtk.yaml', 'r') as config_file:
        config_data = yaml.safe_load(config_file)
        data_path = config_data[split]['DATA_PATH']
        obj_data_path = config_data[split]['OBJ_PATH']
        off_data_path = config_data[split]['OFF_PATH']
        label_path = config_data[split]['LABEL_PATH']
        pic_path = config_data[split]['PIC_PATH']
    # 转换文件格式 .vtk --> .obj & .off
    # convert_file_format(data_path, off_data_path, obj_data_path)
    # 读取csv文件
    protein_ids, class_ids = read_label_map(label_path)
    if len(protein_ids) != len(class_ids):
        raise ValueError(f'The number of files and tags does not match, Please check the {label_path} file')
    classification_num = {}
    print('The number of files corresponding to each category.')
    for i in range(len(class_ids)):
        key = 'cls_' + str(class_ids[i])
        if key in classification_num:
            classification_num[key] += 1
        else:
            classification_num[key] = 1
    classification_num = sorted(classification_num.items())
    # 对于分类数据少的，尽量多打印一些,
    for i in range(len(classification_num)):
        if classification_num[i][1] < 10:
            classification_image_num[classification_num[i][0]] = 100 // classification_num[i][1]
        else:
            classification_image_num[classification_num[i][0]] = 8
        print(f'{classification_num[i][0]}:{classification_num[i][1]}\t\t', end='')
        if (i + 1) % 5 == 0:
            print()

    for i in range(len(protein_ids)):
        file_label[str(protein_ids[i])] = class_ids[i]
    obj_file_dir = []
    traverse_path = os.path.join(obj_data_path, data_path.split('/')[-1].split('.')[0])
    traverse_files(traverse_path, obj_file_dir)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    for i in tqdm(range(len(obj_file_dir)), desc="generate 2d pic"):
        obj_path = obj_file_dir[i]
        filename = obj_path.split("/")[-1].split(".")[0]
        if filename not in file_label:
            print(f'{filename} not found in {file_label}')
            continue
        class_dir = str(file_label[filename])
        class_dir_full = os.path.join(pic_path, class_dir)
        if not os.path.exists(class_dir_full):
            os.makedirs(class_dir_full)
        pic_dir_full = os.path.join(str(class_dir_full), filename)
        num_views = classification_image_num['cls_' + class_dir]
        if split == 'test':
            num_views = 8
        if not os.path.exists(pic_dir_full):
            os.makedirs(pic_dir_full)
        if os.path.exists(pic_dir_full) and len(os.listdir(pic_dir_full)) >= num_views:
            continue

        render_obj_multiview(obj_path, pic_dir_full, num_views=num_views, image_size=256, device=device)
if __name__ == '__main__':
    pre_deal(split='train')
    pre_deal(split='test')
