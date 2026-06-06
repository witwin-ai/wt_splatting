import { THREE } from '../runtime';
import type * as ThreeTypes from 'three';
import { depthKeyFragmentShader, depthKeyVertexShader } from '../shaders/gpuSortShaders';
import type { SplatDataTextures } from './SplatDataTextures';
import { beginFullscreenTargetPass, finishFullscreenTargetPass } from './renderPassState';

export class DepthKeyPass {
  readonly target: ThreeTypes.WebGLRenderTarget;
  private readonly scene: ThreeTypes.Scene;
  private readonly camera: ThreeTypes.Camera;
  private readonly mesh: ThreeTypes.Mesh;
  private readonly material: ThreeTypes.RawShaderMaterial;

  constructor(private readonly textures: SplatDataTextures, private readonly splatCount: number) {
    this.target = new THREE.WebGLRenderTarget(textures.width, textures.height, {
      type: THREE.FloatType,
      format: THREE.RGBAFormat,
      minFilter: THREE.NearestFilter,
      magFilter: THREE.NearestFilter,
      generateMipmaps: false,
      depthBuffer: false,
      stencilBuffer: false,
    });
    this.camera = new THREE.Camera();
    this.scene = new THREE.Scene();
    this.material = new THREE.RawShaderMaterial({
      glslVersion: THREE.GLSL3,
      vertexShader: depthKeyVertexShader,
      fragmentShader: depthKeyFragmentShader,
      uniforms: {
        centerOpacityTex: { value: textures.centerOpacityTex },
        textureSize: { value: new THREE.Vector2(textures.width, textures.height) },
        splatCount: { value: splatCount },
        modelViewMatrixForSplats: { value: new THREE.Matrix4() },
      },
    });
    this.mesh = new THREE.Mesh(fullscreenTriangle(), this.material);
    this.scene.add(this.mesh);
  }

  update(renderer: ThreeTypes.WebGLRenderer, modelView: ThreeTypes.Matrix4): ThreeTypes.Texture {
    this.material.uniforms.modelViewMatrixForSplats.value.copy(modelView);
    const state = beginFullscreenTargetPass(renderer, this.target);
    try {
      renderer.render(this.scene, this.camera);
    } finally {
      finishFullscreenTargetPass(renderer, state);
    }
    return this.target.texture;
  }

  dispose(): void {
    this.target.dispose();
    this.material.dispose();
    this.mesh.geometry.dispose();
  }
}

function fullscreenTriangle(): ThreeTypes.BufferGeometry {
  const geometry = new THREE.BufferGeometry();
  geometry.setAttribute('position', new THREE.Float32BufferAttribute([-1, -1, 0, 3, -1, 0, -1, 3, 0], 3));
  return geometry;
}
