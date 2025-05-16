import os

WEBGL_TEMPLATE = """<!DOCTYPE html>
<html>
<head>
    <title>3D Factory Map</title>
    <script src="https://cdnjs.cloudflare.com/ajax/libs/three.js/r128/three.min.js"></script>
    <script src="https://cdn.jsdelivr.net/npm/three@0.128.0/examples/js/loaders/OBJLoader.js"></script>
</head>
<body>
    <script>
        const scene = new THREE.Scene();
        const camera = new THREE.PerspectiveCamera(75, window.innerWidth/window.innerHeight, 0.1, 1000);
        const renderer = new THREE.WebGLRenderer();
        renderer.setSize(window.innerWidth, window.innerHeight);
        document.body.appendChild(renderer.domElement);
        new THREE.OBJLoader().load('{model_path}', (object) => {
            scene.add(object);
            camera.position.z = 300;
            
            function animate() {
                requestAnimationFrame(animate);
                object.rotation.y += 0.01;
                renderer.render(scene, camera);
            }
            animate();
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