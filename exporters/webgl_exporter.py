import os

WEBGL_TEMPLATE = """<!DOCTYPE html>
<html>
<head>
    <title>3D Factory Map</title>
    <script src="https://cdnjs.cloudflare.com/ajax/libs/three.js/r128/three.min.js"></script>
    <script src="https://cdnjs.cloudflare.com/ajax/libs/three.js/r128/examples/js/loaders/OBJLoader.js"></script>
    <style>
        body { margin: 0; overflow: hidden; }
        canvas { display: block; }
    </style>
</head>
<body>
    <script>
        const scene = new THREE.Scene();
        scene.background = new THREE.Color(0xf0f0f0);
        
        const camera = new THREE.PerspectiveCamera(75, window.innerWidth/window.innerHeight, 0.1, 1000);
        const renderer = new THREE.WebGLRenderer({antialias: true});
        renderer.setSize(window.innerWidth, window.innerHeight);
        document.body.appendChild(renderer.domElement);
        
        // 조명 추가
        const ambientLight = new THREE.AmbientLight(0x404040, 1.5);
        scene.add(ambientLight);
        
        const directionalLight = new THREE.DirectionalLight(0xffffff, 1);
        directionalLight.position.set(1, 1, 1);
        scene.add(directionalLight);
        
        // 모델 로드
        const loader = new THREE.OBJLoader();
        loader.load('{model_path}', (object) => {
            scene.add(object);
            camera.position.z = 5;
            
            // 오브젝트 중심 계산
            const box = new THREE.Box3().setFromObject(object);
            const center = box.getCenter(new THREE.Vector3());
            const size = box.getSize(new THREE.Vector3());
            
            // 카메라 위치 조정
            const maxDim = Math.max(size.x, size.y, size.z);
            camera.position.z = center.z + maxDim * 2;
            camera.lookAt(center);
            
            // 애니메이션
            function animate() {
                requestAnimationFrame(animate);
                object.rotation.y += 0.01;
                renderer.render(scene, camera);
            }
            animate();
        });
        
        // 반응형 크기 조정
        window.addEventListener('resize', () => {
            camera.aspect = window.innerWidth / window.innerHeight;
            camera.updateProjectionMatrix();
            renderer.setSize(window.innerWidth, window.innerHeight);
        });
    </script>
</body>
</html>"""

def export_to_webgl(obj_path, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    html_path = os.path.join(output_dir, "viewer.html")
    
    with open(html_path, 'w') as f:
        f.write(WEBGL_TEMPLATE.format(model_path=os.path.basename(obj_path)))
    
    return html_path