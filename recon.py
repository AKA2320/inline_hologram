import matplotlib.pyplot as plt
import imageio.v3 as iio
import bm3d
import numpy as np
from pyDHM.numericalPropagation import angularSpectrum
from skimage.restoration import unwrap_phase
from skimage.measure import marching_cubes
import trimesh


def recon():
    # Step 1: Load Data
    hologramFile = 'pond/pond hologram.tif'
    backgroundFile = 'pond/background.tif'

    I_pond_raw = iio.imread(hologramFile).astype(np.float32)
    I_bg_raw = iio.imread(backgroundFile).astype(np.float32)
    print(f"Pond data shape: {I_pond_raw.shape}")
    print(f"Background shape: {I_bg_raw.shape}")
    I_bg_avg = np.mean(I_bg_raw, axis=-1)
    I_holo_avg = np.mean(I_pond_raw, axis=0)

    # 2. Background Subtraction and Normalization
    I_int = I_holo_avg - I_bg_avg # Interferogram (I_holo - I_bg)
    # Normalize to 0-1 range for stable propagation/reconstruction
    I_int_norm = (I_int - I_int.min()) / (I_int.max() - I_int.min())
    # Final Interferogram ready for Step 3
    I_int_to_use = I_int_norm

    # 3. Denoising using BM3D
    sigma_psd = 0.2 # Estimated noise standard deviation
    # Apply BM3D denoising
    I_denoised = bm3d.bm3d(I_int_norm, sigma_psd=sigma_psd)
    I_int_to_use = I_denoised # Use the denoised image for the next step

    # 4. Numerical Propagation and Reconstruction
    lambda_m = 450e-9            # Wavelength in meters
    pixel_pitch_m = 0.44e-6      # Camera pixel size in meters
    z_min = 0.0001               # Minimum depth to sweep (meters)
    z_max = 0.0005               # Maximum depth to sweep (meters)
    num_z_steps = 50             # Number of depths to sweep
    I = I_int_to_use # Preprocessed (1024, 1280) interferogram
    U0 = I - 0.5 
    # The object wave U0 is approximated as a real field. We cast it to complex:
    U0 = U0.astype(np.complex128) 
    z_values = np.linspace(z_min, z_max, num_z_steps) 
    reconstructions = []
    for z in z_values:
        Uz = angularSpectrum(U0, wavelength=lambda_m,
                            dx=pixel_pitch_m, dy=pixel_pitch_m, z=z)
        amp = np.abs(Uz)
        phase = np.angle(Uz)
        # Unwrap the phase for a smoother volume
        phase_u = unwrap_phase(phase) 
        reconstructions.append((amp, phase_u, z))

    # 5. 3D Volume Construction
    amp_volume = np.stack([amp for amp, _, _ in reconstructions], axis=-1)
    # Normalize the volume again before Marching Cubes
    amp_volume = (amp_volume - amp_volume.min()) / (amp_volume.max() - amp_volume.min())
    # Extract the Mesh
    verts, faces, normals, _ = marching_cubes(amp_volume, level=0.5)
    # Create and Export Mesh
    mesh = trimesh.Trimesh(verts, faces, vertex_normals=normals)
    vertex_count = len(mesh.vertices)
    face_count = len(mesh.faces)
    print(f"  Vertices: {vertex_count:,}")
    print(f"  Faces:    {face_count:,}")
    mesh.export("inline_recon.glb") 


if __name__ == "__main__":
    recon()