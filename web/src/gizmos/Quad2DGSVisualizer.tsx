import { parseW2gs, type TwoDGaussianSplats } from '../data/load2dgs';
import type * as ReactTypes from 'react';
import type * as ThreeTypes from 'three';
import type { SplatRendererData } from '../data/extractSplatComponent';
import type { ViewportAssetsApi } from '../sdk/types';
import { React, THREE, useEffect, useFrame, useMemo, useRef, useState, useThree } from '../runtime';
import { checkGpuCapabilities } from '../gpu/checkGpuCapabilities';
import { DepthKeyPass } from '../gpu/DepthKeyPass';
import { GpuBitonicSort } from '../gpu/GpuBitonicSort';
import { SplatDataTextures } from '../gpu/SplatDataTextures';
import { quad2dgsFragmentShader, quad2dgsVertexShader } from '../shaders/quad2dgsShaders';
import { Marker, disableRaycast } from './utils';

export default function Quad2DGSVisualizer({ data, assets }: { data: SplatRendererData; assets: ViewportAssetsApi }) {
  const { gl, camera } = useThree();
  const groupRef = useRef<ThreeTypes.Group>(null);
  const [payload, setPayload] = useState<TwoDGaussianSplats | null>(null);
  const [error, setError] = useState<string | null>(data.error || null);
  const capabilities = useMemo(() => checkGpuCapabilities(gl), [gl]);

  useEffect(() => {
    const controller = new AbortController();
    setPayload(null);
    setError(data.error || null);
    assets
      .fetchBinary(data.ref, { signal: controller.signal })
      .then((buffer) => setPayload(parseW2gs(buffer)))
      .catch((err) => {
        if (!controller.signal.aborted) setError(err instanceof Error ? err.message : String(err));
      });
    return () => controller.abort();
  }, [assets, data.ref.assetId, data.ref.contentHash, data.error]);

  if (error) return React.createElement(Marker, { message: error, color: '#ff5a5f' });
  if (!capabilities.supported) return React.createElement(Marker, { message: capabilities.reason || '2DGS GPU path unsupported', color: '#ffb020' });
  if (!payload) return React.createElement(Marker, { message: 'Loading 2DGS...', color: '#8aa4ff' });
  if (payload.splatCount > data.max2dSplats && data.sortMode !== 'gpu_weighted') {
    return React.createElement(Marker, { message: '2DGS exact GPU sort budget exceeded', color: '#ffb020' });
  }
  return React.createElement(Quad2DGSPrimitive, { data, payload, gl, camera, groupRef });
}

function Quad2DGSPrimitive({
  data,
  payload,
  gl,
  camera,
  groupRef,
}: {
  data: SplatRendererData;
  payload: TwoDGaussianSplats;
  gl: ThreeTypes.WebGLRenderer;
  camera: ThreeTypes.Camera;
  groupRef: ReactTypes.MutableRefObject<ThreeTypes.Group | null>;
}) {
  const resources = useMemo(() => {
    const textures = new SplatDataTextures(payload);
    const depth = new DepthKeyPass(textures, payload.splatCount);
    const sorter = data.sortMode === 'gpu_weighted' ? null : new GpuBitonicSort(textures.width, textures.height);
    const geometry = new THREE.InstancedBufferGeometry();
    geometry.instanceCount = payload.splatCount;
    geometry.setAttribute(
      'position',
      new THREE.Float32BufferAttribute([-3, -3, 0, 3, -3, 0, -3, 3, 0, 3, 3, 0], 3)
    );
    geometry.setIndex([0, 1, 2, 2, 1, 3]);
    const material =
      data.sortMode === 'gpu_weighted'
        ? buildWeightedBlendMaterial(data, textures, depth.target.texture)
        : buildDepthSortedMaterial(data, textures, depth.target.texture);
    const mesh = new THREE.Mesh(geometry, material);
    mesh.frustumCulled = false;
    disableRaycast(mesh);
    return { textures, depth, sorter, geometry, material, mesh };
  }, [payload, data.opacityScale, data.scaleModifier, data.sortMode]);

  useFrame(() => {
    const group = groupRef.current;
    const modelView = new THREE.Matrix4();
    if (group) modelView.multiplyMatrices(camera.matrixWorldInverse, group.matrixWorld);
    const keyTexture = resources.depth.update(gl, modelView);
    resources.material.uniforms.sortedIndexTex.value = resources.sorter
      ? resources.sorter.run(gl, keyTexture, payload.splatCount)
      : keyTexture;
  });

  useEffect(() => {
    return () => {
      resources.textures.dispose();
      resources.depth.dispose();
      resources.sorter?.dispose();
      resources.geometry.dispose();
      resources.material.dispose();
    };
  }, [resources]);

  return React.createElement('group' as any, { ref: groupRef }, React.createElement('primitive' as any, { object: resources.mesh }));
}

function buildDepthSortedMaterial(
  data: SplatRendererData,
  textures: SplatDataTextures,
  sortedIndexTex: ThreeTypes.Texture
): ThreeTypes.RawShaderMaterial {
  return buildQuadMaterial(data, textures, sortedIndexTex, {
    weightedMode: false,
    blending: THREE.NormalBlending,
  });
}

function buildWeightedBlendMaterial(
  data: SplatRendererData,
  textures: SplatDataTextures,
  sortedIndexTex: ThreeTypes.Texture
): ThreeTypes.RawShaderMaterial {
  return buildQuadMaterial(data, textures, sortedIndexTex, {
    weightedMode: true,
    blending: THREE.CustomBlending,
    blendEquation: THREE.AddEquation,
    blendSrc: THREE.OneFactor,
    blendDst: THREE.OneFactor,
  });
}

function buildQuadMaterial(
  data: SplatRendererData,
  textures: SplatDataTextures,
  sortedIndexTex: ThreeTypes.Texture,
  options: ThreeTypes.ShaderMaterialParameters & { weightedMode: boolean }
): ThreeTypes.RawShaderMaterial {
  return new THREE.RawShaderMaterial({
    glslVersion: THREE.GLSL3,
    transparent: true,
    depthWrite: false,
    side: THREE.DoubleSide,
    vertexShader: quad2dgsVertexShader,
    fragmentShader: quad2dgsFragmentShader,
    uniforms: {
      sortedIndexTex: { value: sortedIndexTex },
      centerOpacityTex: { value: textures.centerOpacityTex },
      basisUTex: { value: textures.basisUTex },
      basisVTex: { value: textures.basisVTex },
      colorTex: { value: textures.colorTex },
      textureSize: { value: new THREE.Vector2(textures.width, textures.height) },
      opacityScale: { value: data.opacityScale },
      scaleModifier: { value: data.scaleModifier },
      weightedMode: { value: options.weightedMode },
    },
    ...options,
  });
}
