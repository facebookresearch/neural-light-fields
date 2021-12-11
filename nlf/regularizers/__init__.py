from .ray_depth import RayDepthRegularizer
from .ray_depth_occ_dir import RayDepthOccDirRegularizer
from .multiple_ray_depth import MultipleRayDepthRegularizer
from .inverse_ray_depth import InverseRayDepthRegularizer
from .ray_blending import RayDepthBlendingRegularizer
from .ray_interpolation import RayInterpolationRegularizer
from .depth_classification import DepthClassificationRegularizer
from .ray_bundle import RayBundleRegularizer
from .fourier import FourierRegularizer
from .teacher import TeacherRegularizer, BlurryTeacherRegularizer
from .voxel_sparsity import VoxelSparsityRegularizer
from .warp import WarpRegularizer, WarpLevelSetRegularizer
from .visualization import EPIVisualizer, FocusVisualizer

regularizer_dict = {
    'ray_depth': RayDepthRegularizer,
    'ray_depth_occ_dir': RayDepthOccDirRegularizer,
    'multiple_ray_depth': MultipleRayDepthRegularizer,
    'inverse_ray_depth': InverseRayDepthRegularizer,
    'ray_depth_blending': RayDepthBlendingRegularizer,
    'ray_interpolation': RayInterpolationRegularizer,
    'depth_classification': DepthClassificationRegularizer,
    'fourier': FourierRegularizer,
    'ray_bundle': RayBundleRegularizer,
    'teacher': TeacherRegularizer,
    'blurry_teacher': BlurryTeacherRegularizer,
    'voxel_sparsity': VoxelSparsityRegularizer,
    'warp': WarpRegularizer,
    'warp_level': WarpLevelSetRegularizer,
    'epi_vis': EPIVisualizer,
    'focus_vis': FocusVisualizer,
}
