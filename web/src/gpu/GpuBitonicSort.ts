import { THREE } from '../runtime';
import type * as ThreeTypes from 'three';
import { bitonicSortFragmentShader, depthKeyVertexShader } from '../shaders/gpuSortShaders';
import { beginFullscreenTargetPass, finishFullscreenTargetPass } from './renderPassState';

export class GpuBitonicSort {
  private readonly targetA: ThreeTypes.WebGLRenderTarget;
  private readonly targetB: ThreeTypes.WebGLRenderTarget;
  private readonly scene: ThreeTypes.Scene;
  private readonly camera: ThreeTypes.Camera;
  private readonly material: ThreeTypes.RawShaderMaterial;
  private readonly mesh: ThreeTypes.Mesh;
  private output: ThreeTypes.WebGLRenderTarget;

  constructor(private readonly width: number, private readonly height: number) {
    const options = {
      type: THREE.FloatType,
      format: THREE.RGBAFormat,
      minFilter: THREE.NearestFilter,
      magFilter: THREE.NearestFilter,
      generateMipmaps: false,
      depthBuffer: false,
      stencilBuffer: false,
    };
    this.targetA = new THREE.WebGLRenderTarget(width, height, options);
    this.targetB = new THREE.WebGLRenderTarget(width, height, options);
    this.output = this.targetA;
    this.camera = new THREE.Camera();
    this.scene = new THREE.Scene();
    this.material = new THREE.RawShaderMaterial({
      glslVersion: THREE.GLSL3,
      vertexShader: depthKeyVertexShader,
      fragmentShader: bitonicSortFragmentShader,
      uniforms: {
        inputTex: { value: null },
        textureSize: { value: new THREE.Vector2(width, height) },
        sortTexelCount: { value: width * height },
        stage: { value: 0 },
        passOfStage: { value: 0 },
      },
    });
    this.mesh = new THREE.Mesh(fullscreenTriangle(), this.material);
    this.scene.add(this.mesh);
  }

  run(renderer: ThreeTypes.WebGLRenderer, input: ThreeTypes.Texture, splatCount: number): ThreeTypes.Texture {
    let source = input;
    const sortTexelCount = nextPowerOfTwo(Math.max(1, splatCount));
    this.material.uniforms.sortTexelCount.value = sortTexelCount;
    const stages = Math.ceil(Math.log2(sortTexelCount));
    let state = null as ReturnType<typeof beginFullscreenTargetPass> | null;
    try {
      for (let stage = 0; stage < stages; stage += 1) {
        for (let passOfStage = stage; passOfStage >= 0; passOfStage -= 1) {
          this.material.uniforms.inputTex.value = source;
          this.material.uniforms.stage.value = stage;
          this.material.uniforms.passOfStage.value = passOfStage;
          this.output = this.output === this.targetA ? this.targetB : this.targetA;
          if (state) {
            finishFullscreenTargetPass(renderer, state);
            state = null;
          }
          state = beginFullscreenTargetPass(renderer, this.output);
          renderer.render(this.scene, this.camera);
          source = this.output.texture;
        }
      }
    } finally {
      if (state) {
        finishFullscreenTargetPass(renderer, state);
      }
    }
    return source;
  }

  dispose(): void {
    this.targetA.dispose();
    this.targetB.dispose();
    this.material.dispose();
    this.mesh.geometry.dispose();
  }
}

function nextPowerOfTwo(value: number): number {
  let result = 1;
  while (result < value) result *= 2;
  return result;
}

function fullscreenTriangle(): ThreeTypes.BufferGeometry {
  const geometry = new THREE.BufferGeometry();
  geometry.setAttribute('position', new THREE.Float32BufferAttribute([-1, -1, 0, 3, -1, 0, -1, 3, 0], 3));
  return geometry;
}
