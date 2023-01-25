import numpy as np
import pylab
import pickle
import numpy as np
import torch
import torch.nn.functional as F
import struct
import zlib
import imageio

torch.pi = torch.acos(torch.zeros(1)).item() * 2 # which is 3.1415927410125732
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DEBUG = False





##################################################
##################################################
##################################################
def quaternion2rotation(q):
    qw, qx, qy, qz = q
    R = np.array([[1 - 2*qy**2 - 2*qz**2, 2*qx*qy - 2*qz*qw, 2*qx*qz + 2*qy*qw], 
                  [2*qx*qy + 2*qz*qw, 1 - 2*qx**2 - 2*qz**2, 2*qy*qz - 2*qx*qw], 
                  [2*qx*qz - 2*qy*qw, 2*qy*qz + 2*qx*qw, 1 - 2*qx**2 - 2*qy**2]])
    return R



def make_M_from_tqs(t, q, s):
    # q = np.quaternion(q[0], q[1], q[2], q[3])
    T = np.eye(4)
    T[0:3, 3] = t
    R = np.eye(4)
    R[0:3, 0:3] = quaternion2rotation(q) # quaternion.as_rotation_matrix(q)
    S = np.eye(4)
    S[0:3, 0:3] = np.diag(s)
    M = T.dot(R).dot(S)
    return M



def load_matrix_from_txt(path, shape=(4, 4)):
    with open(path) as f:
        txt = f.readlines()
    txt = ''.join(txt).replace('\n', ' ')
    matrix = [float(v) for v in txt.split()]
    return np.array(matrix).reshape(shape)



def loadOBJ(fliePath):
    vertices = []
    for line in open(fliePath, "r"):
        vals = line.split()
        if len(vals) == 0:
            continue
        if vals[0] == "v":
            v = np.array([float(v_i) for v_i in vals[1:4]])
            vertices.append(v)
    return np.array(vertices)



def inverse_dict(d):
    return {v:k for k,v in d.items()}
##################################################
##################################################
##################################################





def pickle_dump(obj, path):
    with open(path, mode='wb') as f:
        pickle.dump(obj,f)



def pickle_load(path):
    with open(path, mode='rb') as f:
        data = pickle.load(f)
        return data



def polar2xyz(theta, phi, r=1):
    x = r * torch.sin(theta) * torch.cos(phi)
    y = r * torch.sin(theta) * torch.sin(phi)
    z = r * torch.cos(theta)
    return x, y, z



def xyz2polar(x, y, z):
    r = torch.sqrt(x**2 + y**2 + z**2)
    theta = torch.arccos(z/r)
    phi = torch.sgn(y) * torch.arccos(x/torch.sqrt(x**2+y**2))
    return r, theta, phi



def check_map_torch(image, path = 'tes.png', figsize=[10,10]):
    fig = pylab.figure(figsize=figsize)

    ax1 = fig.add_subplot(1, 1, 1)
    ax1.set_title('result')
    ax1.imshow(image.to('cpu').detach().numpy().copy())

    ax1.xaxis.set_ticklabels([])
    ax1.yaxis.set_ticklabels([])
    fig.savefig(path, dpi=300)
    pylab.close()



def check_map_np(image, path = 'tes.png', figsize=[10,10]):
    fig = pylab.figure(figsize=figsize)

    ax1 = fig.add_subplot(1, 1, 1)
    ax1.set_title('result')
    ax1.imshow(image)

    ax1.xaxis.set_ticklabels([])
    ax1.yaxis.set_ticklabels([])
    fig.savefig(path, dpi=300)
    pylab.close()



def vec2skew(v):
    """
    :param v:  (3, ) torch tensor
    :return:   (3, 3)
    """
    zero = torch.zeros(1, dtype=torch.float32, device=v.device)
    skew_v0 = torch.cat([ zero,    -v[2:3],   v[1:2]])  # (3, 1)
    skew_v1 = torch.cat([ v[2:3],   zero,    -v[0:1]])
    skew_v2 = torch.cat([-v[1:2],   v[0:1],   zero])
    skew_v = torch.stack([skew_v0, skew_v1, skew_v2], dim=0)  # (3, 3)
    return skew_v  # (3, 3)



def Exp(r):
    """so(3) vector to SO(3) matrix
    :param r: (3, ) axis-angle, torch tensor
    :return:  (3, 3)
    """
    skew_r = vec2skew(r)  # (3, 3)
    norm_r = r.norm() + 1e-15
    eye = torch.eye(3, dtype=torch.float32, device=r.device)
    R = eye + (torch.sin(norm_r) / norm_r) * skew_r + ((1 - torch.cos(norm_r)) / norm_r**2) * (skew_r @ skew_r)
    return R



def get_ray_direction(size, fov, c2w=False):
    fov = torch.deg2rad(torch.tensor(fov, dtype=torch.float))
    x_coord = torch.linspace(-torch.tan(fov*.5), torch.tan(fov*.5), size)[None].expand(size, -1)
    y_coord = x_coord.T
    rays_d_cam = torch.stack([x_coord, y_coord, torch.ones_like(x_coord)], dim=2)
    rays_d_cam = F.normalize(rays_d_cam, dim=-1)
    if c2w is False:
        return rays_d_cam.unsqueeze(0).detach() # H, W, 3:xyz
    else:
        rays_d_cam = rays_d_cam.unsqueeze(0).to(c2w.device).detach()
        rays_d_wrd = torch.sum(rays_d_cam[:, :, :, None, :] * c2w[:, None, None, :, :], -1)
        return rays_d_wrd # batch, H, W, 3:xyz



# Sampling view points on a sphere uniformaly by the fibonacci sampling.
from numpy import arange, pi, cos, sin, tan, dot, arccos
def sample_fibonacci_views(n):
    i = arange(0, n, dtype=float)
    phi = arccos(1 - 2*i/n)
    goldenRatio = (1 + 5**0.5)/2
    theta = 2 * pi * i / goldenRatio
    X, Y, Z = cos(theta) * sin(phi), sin(theta) * sin(phi), cos(phi)
    xyz = np.stack([X, Y, Z], axis=1)
    return xyz



def sample_pose(v, look_at, look_up_randm='normal'):
    pos = v
    look_at = look_at - v
    look_at = look_at / np.linalg.norm(look_at)

    if np.abs(np.dot(look_at, np.array([0, 1, 0]))) < 0.999:
        up = np.array([0, 1, 0])
    else:
        ang = np.radians(360*np.random.rand())
        up = np.array([np.cos(ang), 0, np.sin(ang)])

    right = np.cross(look_at, up)
    up = np.cross(right, look_at)

    right = right / np.linalg.norm(right)
    up = up / np.linalg.norm(up)

    rot = np.array([right, up, -look_at])
    return pos, rot



def get_OSMap_obj(distance_map, mask, rays_d_cam, w2c, cam_pos_wrd, o2w, obj_pos_wrd, obj_scale):
    OSMap_cam = distance_map[..., None] * rays_d_cam
    OSMap_wrd = torch.sum(OSMap_cam[..., None, :]*w2c.permute(0, 2, 1)[..., None, None, :, :], dim=-1) + cam_pos_wrd[..., None, None, :]
    OSMap_obj = torch.sum((OSMap_wrd - obj_pos_wrd[..., None, None, :])[..., None, :]*o2w.permute(0, 2, 1)[..., None, None, :, :], dim=-1) / obj_scale[..., None, None, :]
    OSMap_obj[torch.logical_not(mask)] = 0.
    OSMap_obj_wMask = torch.cat([OSMap_obj, mask.to(OSMap_obj.dtype).unsqueeze(-1)], dim=-1)
    return OSMap_obj_wMask



def get_diffOSMap_obj(obs_distance_map, est_distance_map, obs_mask, est_mask, rays_d_cam, w2c, o2w, obj_scale):
    diff_or_mask = torch.logical_or(obs_mask, est_mask)
    diff_xor_mask = torch.logical_xor(obs_mask, est_mask)

    diff_distance_map = obs_distance_map - est_distance_map
    diffOSMap_cam = diff_distance_map[..., None] * rays_d_cam
    diffOSMap_wrd = torch.sum(diffOSMap_cam[..., None, :]*w2c.permute(0, 2, 1)[..., None, None, :, :], dim=-1)
    diffOSMap_obj = torch.sum(diffOSMap_wrd[..., None, :]*o2w.permute(0, 2, 1)[..., None, None, :, :], dim=-1)
    diffOSMap_obj = diffOSMap_obj / obj_scale[..., None, None, :]

    diffOSMap_obj[torch.logical_not(diff_or_mask)] = 0.
    diffOSMap_obj_wMask = torch.cat([diffOSMap_obj, diff_xor_mask.to(diffOSMap_obj).unsqueeze(-1)], dim=-1)
    return diffOSMap_obj_wMask



def txt2list(txt_file):
    result_list = []
    with open(txt_file, 'r') as f:
        lines = f.read().splitlines()
        for line in lines:
            result_list.append(line.rstrip('\n'))
    return result_list



def list2txt(result_list, txt_file):
    with open(txt_file, 'a') as f:
        for result in result_list:
            f.write(result + '\n')



class RGBDFrame():

  def load(self, file_handle):
    self.camera_to_world = np.asarray(struct.unpack('f'*16, file_handle.read(16*4)), dtype=np.float32).reshape(4, 4)
    self.timestamp_color = struct.unpack('Q', file_handle.read(8))[0]
    self.timestamp_depth = struct.unpack('Q', file_handle.read(8))[0]
    self.color_size_bytes = struct.unpack('Q', file_handle.read(8))[0]
    self.depth_size_bytes = struct.unpack('Q', file_handle.read(8))[0]
    self.color_data = b''.join(struct.unpack('c'*self.color_size_bytes, file_handle.read(self.color_size_bytes)))
    self.depth_data = b''.join(struct.unpack('c'*self.depth_size_bytes, file_handle.read(self.depth_size_bytes)))


  def decompress_depth(self, compression_type):
    if compression_type == 'zlib_ushort':
       return self.decompress_depth_zlib()
    else:
       raise


  def decompress_depth_zlib(self):
    return zlib.decompress(self.depth_data)


  def decompress_color(self, compression_type):
    if compression_type == 'jpeg':
       return self.decompress_color_jpeg()
    else:
       raise


  def decompress_color_jpeg(self):
    return imageio.imread(self.color_data)