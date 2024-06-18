import cv2
import numpy as np
import pyrealsense2 as rs
import open3d as o3d #pip install open3d

# Función para detectar y emparejar características
def detect_and_match_features(img1, img2):
    orb = cv2.ORB_create()
    kp1, des1 = orb.detectAndCompute(img1, None)
    kp2, des2 = orb.detectAndCompute(img2, None)
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(des1, des2)
    matches = sorted(matches, key=lambda x: x.distance)
    return kp1, kp2, matches

# Función para estimar la pose
def estimate_pose(kp1, kp2, matches, K):
    pts1 = np.float32([kp1[m.queryIdx].pt for m in matches])
    pts2 = np.float32([kp2[m.trainIdx].pt for m in matches])
    E, mask = cv2.findEssentialMat(pts1, pts2, K, method=cv2.FM_LMEDS, prob=0.999, threshold=3.0)
    _, R, t, mask = cv2.recoverPose(E, pts1, pts2, K)
    return R, t, mask, pts1, pts2, mask

# Función para triangular puntos
def triangulate_points(R, t, pts1, pts2, K):
    # Construir matrices de proyección
    P1 = np.hstack((R, t))  # Matriz de proyección de la cámara 1
    P2 = np.hstack((np.eye(3), np.zeros((3, 1))))  # Matriz de proyección de la cámara 2

    # Aplicar la matriz de calibración K a las matrices de proyección
    P1 = K @ P1
    P2 = K @ P2

    # Convertir a tipo np.float32 para cv2.triangulatePoints
    P1 = P1.astype(np.float32)
    P2 = P2.astype(np.float32)
    pts1 = pts1.astype(np.float32)
    pts2 = pts2.astype(np.float32)

    # Triangulación de puntos
    pts4D = cv2.triangulatePoints(P1, P2, pts1.T, pts2.T)
    pts3D = pts4D / pts4D[3]  # Normalizar puntos 4D a coordenadas 3D
    
    # Devolver puntos 3D
    return pts3D[:3].T

# Obtener los parámetros intrínsecos de la RealSense
pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
profile = pipeline.start(config)
intr = profile.get_stream(rs.stream.color).as_video_stream_profile().get_intrinsics()
K = np.array([[intr.fx, 0, intr.ppx],
              [0, intr.fy, intr.ppy],
              [0, 0, 1]], dtype=np.float32)



# Capturar imágenes de la RealSense
def capture_image(pipeline):
    frames = pipeline.wait_for_frames()
    color_frame = frames.get_color_frame()
    if not color_frame:
        return None
    color_image = np.asanyarray(color_frame.get_data())
    return color_image

print("Capturando imagen 1...")
img1 = capture_image(pipeline)
if img1 is None:
    print("Error al capturar la imagen 1")
    exit()
cv2.imshow("Image 1", img1)
cv2.waitKey(1000)  # Esperar 1 segundo para simular el tiempo entre capturas

print("Capturando imagen 2...")
img2 = capture_image(pipeline)
if img2 is None:
    print("Error al capturar la imagen 2")
    exit()
cv2.imshow("Image 2", img2)
cv2.waitKey(1000)  # Esperar 1 segundo para simular el tiempo entre capturas

pipeline.stop()

# Convertir a escala de grises
img1_gray = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
img2_gray = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

# Paso 1: Detectar y emparejar características
kp1, kp2, matches = detect_and_match_features(img1_gray, img2_gray)

# Paso 2: Estimar la pose relativa entre las dos primeras imágenes
R, t, mask, pts1, pts2, mask = estimate_pose(kp1, kp2, matches, K)



# Mostrar número de correspondencias y máscara
print("Número de correspondencias:", len(matches))
print("Máscara de inliers de RANSAC:", mask)

# Paso 3: Triangular puntos para obtener la estructura 3D inicial
pts1_valid = pts1[mask.ravel() == 1]
pts2_valid = pts2[mask.ravel() == 1]

#if len(pts1_valid) < 6 or len(pts2_valid) < 6:
    #print("No hay suficientes puntos válidos para la triangulación")
    #exit()  

points_3D = triangulate_points(R, t, pts1, pts2, K)

print("3D Points:", points_3D)

pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(points_3D)

# Visualizar la nube de puntos
o3d.visualization.draw_geometries([pcd])

