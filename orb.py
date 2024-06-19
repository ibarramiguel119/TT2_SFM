import cv2
import numpy as np
import open3d as o3d
import plotly.graph_objects as go
import pyrealsense2 as rs

def capture_image(pipeline):
    frames = pipeline.wait_for_frames()
    color_frame = frames.get_color_frame()
    if not color_frame:
        return None
    color_image = np.asanyarray(color_frame.get_data())
    return color_image

def get_intrinsics(profile):
    intrinsics = profile.get_stream(rs.stream.color).as_video_stream_profile().get_intrinsics()
    K = np.array([[intrinsics.fx, 0, intrinsics.ppx],
                  [0, intrinsics.fy, intrinsics.ppy],
                  [0, 0, 1]])
    
    print("Matriz Intrínseca (K):", K)
    return K

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

def main():
    # Configurar la pipeline de Intel RealSense
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
    profile = pipeline.start(config)

    try:
        all_keypoints = []  # Lista para almacenar todos los puntos clave de cada imagen
        all_descriptors = []  # Lista para almacenar todos los descriptores de cada imagen

        for i in range(5):
            print(f"Capturando imagen {i+1}...")
            img = capture_image(pipeline)
            cv2.imshow(f"Imagen {i+1}", img)
            cv2.waitKey(1000)  # Esperar 1 segundo entre capturas

            gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            # Crear una instancia del detector SIFT con parámetros personalizados
            sift = cv2.SIFT_create(nfeatures=1000, contrastThreshold=0.03, sigma=1.0)

            # Detectar y computar características para la imagen actual
            keypoints, descriptors = sift.detectAndCompute(gray_img, None)

            # Guardar los puntos clave y descriptores en la lista
            all_keypoints.append(keypoints)
            all_descriptors.append(descriptors)

        # Comparar cada par de imágenes consecutivas y encontrar coincidencias
        MAX_MATCHES = 500  # Ajusta este valor según tus necesidades
        all_matches = []
        for i in range(4):  # Compara 1 y 2, 2 y 3, 3 y 4, 4 y 5
            keypoints1 = all_keypoints[i]
            keypoints2 = all_keypoints[i + 1]
            descriptors1 = all_descriptors[i]
            descriptors2 = all_descriptors[i + 1]

            # Crear un objeto BFMatcher
            bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)

            # Realizar la coincidencia entre los descriptores SIFT de ambas imágenes
            matches = bf.match(descriptors1, descriptors2)

            # Ordenar las coincidencias en función de la distancia (las mejores coincidencias primero)
            matches = sorted(matches, key=lambda x: x.distance)

            # Tomar las mejores coincidencias
            matches = matches[:MAX_MATCHES]

            # Obtener puntos clave para las mejores coincidencias
            pts1 = np.float32([keypoints1[m.queryIdx].pt for m in matches])
            pts2 = np.float32([keypoints2[m.trainIdx].pt for m in matches])

            # Guardar las correspondencias
            all_matches.append((pts1, pts2))

        # Obtener la matriz intrínseca K de la cámara
        K = get_intrinsics(profile)

        # Variables para almacenar los puntos filtrados para la triangulación
        pts1_accumulated = np.empty((0, 2), dtype=np.float32)
        pts2_accumulated = np.empty((0, 2), dtype=np.float32)

        # Triangulación de puntos para obtener la nube 3D
        for pts1, pts2 in all_matches:
            # Encontrar la matriz fundamental utilizando RANSAC
            F, mask = cv2.findFundamentalMat(pts1, pts2, cv2.FM_RANSAC)

            # Seleccionar solo los puntos inliers
            pts1_filter = pts1[mask.ravel() == 1]
            pts2_filter = pts2[mask.ravel() == 1]

            # Dibujar puntos de correspondencia sobre las imágenes originales (opcional)
            # for pt1, pt2 in zip(pts1_filter, pts2_filter):
            #     cv2.circle(img1_pts, tuple(pt1.astype(int)), 5, (0, 0, 255), -1)
            #     cv2.circle(img2_pts, tuple(pt2.astype(int)), 5, (0, 0, 255), -1)

            # Calcular la matriz esencial E
            E = K.T.dot(F).dot(K)

            # Descomponer la matriz esencial para obtener R y t
            _, R, t, mask = cv2.recoverPose(E, pts1_filter, pts2_filter, K)

            # Triangulación de puntos para obtener la nube 3D
            points3d = triangulate_points(R, t, pts1_filter, pts2_filter, K)
            print(f"Puntos 3D para imágenes {i+1} y {i+2}:")
            print(points3d)

            # Agregar los puntos filtrados a los acumulados
            pts1_accumulated = np.vstack((pts1_accumulated, pts1_filter))
            pts2_accumulated = np.vstack((pts2_accumulated, pts2_filter))

        # Visualizar la nube de puntos 3D con Open3D
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points3d)
        o3d.visualization.draw_geometries([pcd])

        # Visualizar la nube de puntos 3D con Plotly
        fig = go.Figure(data=[go.Scatter3d(
            x=points3d[:, 0],
            y=points3d[:, 1],
            z=points3d[:, 2],
            mode='markers',
            marker=dict(
                size=3,
                color=points3d[:, 2],  # Color por coordenada Z
                colorscale='Viridis',
                opacity=0.8
            )
        )])
        fig.update_layout(scene=dict(aspectmode='data'))
        fig.show()

    finally:
        pipeline.stop()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
