import open3d as o3d
import numpy as np
import os

# 强制使用 CPU 渲染（如果显卡驱动有问题，这行可能救命）
os.environ['LIBGL_ALWAYS_SOFTWARE'] = '1'

try:
    print(f"Open3D Version: {o3d.__version__}")
    
    # 1. 创建随机点云
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(np.random.rand(100, 3))
    
    # 2. 尝试初始化可视化器
    vis = o3d.visualization.Visualizer()
    
    # 关键点：在 WSL 中，如果这步报错或卡住，说明没有 X Server
    created = vis.create_window(visible=False, width=800, height=600)
    if not created:
        print("❌ Error: Failed to create window. WSL graphics not configured.")
        exit(1)
        
    vis.add_geometry(pcd)
    vis.poll_events()
    vis.update_renderer()
    
    # 3. 尝试截图
    img = vis.capture_screen_float_buffer()
    vis.destroy_window()
    
    if img is None:
        print("❌ Error: Capture returned None (Rendering failed).")
    else:
        print(f"✅ Success! Captured image shape: {np.asarray(img).shape}")

except Exception as e:
    print(f"❌ Exception occurred: {e}")