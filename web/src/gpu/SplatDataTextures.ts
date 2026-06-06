import { THREE } from '../runtime';
import type * as ThreeTypes from 'three';
import type { TwoDGaussianSplats } from '../data/load2dgs';

export class SplatDataTextures {
  readonly width: number;
  readonly height: number;
  readonly sortTexelCount: number;
  readonly centerOpacityTex: ThreeTypes.DataTexture;
  readonly basisUTex: ThreeTypes.DataTexture;
  readonly basisVTex: ThreeTypes.DataTexture;
  readonly colorTex: ThreeTypes.DataTexture;

  constructor(data: TwoDGaussianSplats) {
    this.sortTexelCount = nextPowerOfTwo(Math.max(1, data.splatCount));
    this.width = sortTextureWidth(this.sortTexelCount);
    this.height = this.sortTexelCount / this.width;
    this.centerOpacityTex = textureFrom(data.splatCount, this.width, this.height, (i, out) => {
      out[0] = data.centers[i * 3 + 0];
      out[1] = data.centers[i * 3 + 1];
      out[2] = data.centers[i * 3 + 2];
      out[3] = data.opacity[i];
    });
    this.basisUTex = textureFrom(data.splatCount, this.width, this.height, (i, out) => {
      out[0] = data.tangentU[i * 3 + 0];
      out[1] = data.tangentU[i * 3 + 1];
      out[2] = data.tangentU[i * 3 + 2];
      out[3] = data.sigmaUv[i * 2 + 0];
    });
    this.basisVTex = textureFrom(data.splatCount, this.width, this.height, (i, out) => {
      out[0] = data.tangentV[i * 3 + 0];
      out[1] = data.tangentV[i * 3 + 1];
      out[2] = data.tangentV[i * 3 + 2];
      out[3] = data.sigmaUv[i * 2 + 1];
    });
    this.colorTex = textureFrom(data.splatCount, this.width, this.height, (i, out) => {
      out[0] = data.colorDc[i * 3 + 0];
      out[1] = data.colorDc[i * 3 + 1];
      out[2] = data.colorDc[i * 3 + 2];
      out[3] = 1.0;
    });
  }

  dispose(): void {
    this.centerOpacityTex.dispose();
    this.basisUTex.dispose();
    this.basisVTex.dispose();
    this.colorTex.dispose();
  }
}

function nextPowerOfTwo(value: number): number {
  let result = 1;
  while (result < value) result *= 2;
  return result;
}

function sortTextureWidth(sortTexelCount: number): number {
  return 2 ** Math.floor(Math.log2(Math.sqrt(sortTexelCount)));
}

function textureFrom(
  splatCount: number,
  width: number,
  height: number,
  write: (index: number, out: Float32Array) => void
): ThreeTypes.DataTexture {
  const data = new Float32Array(width * height * 4);
  const out = new Float32Array(4);
  for (let i = 0; i < splatCount; i += 1) {
    out.fill(0);
    write(i, out);
    data.set(out, i * 4);
  }
  const texture = new THREE.DataTexture(data, width, height, THREE.RGBAFormat, THREE.FloatType);
  texture.needsUpdate = true;
  texture.magFilter = THREE.NearestFilter;
  texture.minFilter = THREE.NearestFilter;
  return texture;
}
