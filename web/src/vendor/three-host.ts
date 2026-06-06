import type * as ThreeTypes from 'three';

type WitwinGlobals = { __witwin?: { three?: typeof ThreeTypes } };

const hostThree = (globalThis as WitwinGlobals).__witwin?.three;
if (!hostThree) {
  throw new Error('[wt_splatting] Three.js host singleton is unavailable');
}

export const THREE = hostThree;
export default hostThree;

export const Box3 = hostThree.Box3;
export const BufferAttribute = hostThree.BufferAttribute;
export const BufferGeometry = hostThree.BufferGeometry;
export const Camera = hostThree.Camera;
export const Clock = hostThree.Clock;
export const Color = hostThree.Color;
export const ColorManagement = hostThree.ColorManagement;
export const CubeCamera = hostThree.CubeCamera;
export const Data3DTexture = hostThree.Data3DTexture;
export const DataArrayTexture = hostThree.DataArrayTexture;
export const DataTexture = hostThree.DataTexture;
export const DoubleSide = hostThree.DoubleSide;
export const DynamicDrawUsage = hostThree.DynamicDrawUsage;
export const Euler = hostThree.Euler;
export const FileLoader = hostThree.FileLoader;
export const Float32BufferAttribute = hostThree.Float32BufferAttribute;
export const FloatType = hostThree.FloatType;
export const GLSL3 = hostThree.GLSL3;
export const InstancedBufferAttribute = hostThree.InstancedBufferAttribute;
export const InstancedBufferGeometry = hostThree.InstancedBufferGeometry;
export const LinearFilter = hostThree.LinearFilter;
export const LinearMipMapLinearFilter = hostThree.LinearMipMapLinearFilter;
export const LinearSRGBColorSpace = hostThree.LinearSRGBColorSpace;
export const Loader = hostThree.Loader;
export const Matrix2 = (hostThree as any).Matrix2;
export const Matrix3 = hostThree.Matrix3;
export const Matrix4 = hostThree.Matrix4;
export const Mesh = hostThree.Mesh;
export const MeshStandardMaterial = hostThree.MeshStandardMaterial;
export const NearestFilter = hostThree.NearestFilter;
export const Object3D = hostThree.Object3D;
export const OrthographicCamera = hostThree.OrthographicCamera;
export const PMREMGenerator = hostThree.PMREMGenerator;
export const PerspectiveCamera = hostThree.PerspectiveCamera;
export const Quaternion = hostThree.Quaternion;
export const RawShaderMaterial = hostThree.RawShaderMaterial;
export const Raycaster = hostThree.Raycaster;
export const REVISION = hostThree.REVISION;
export const RGBAFormat = hostThree.RGBAFormat;
export const RGBAIntegerFormat = (hostThree as any).RGBAIntegerFormat;
export const RGIntegerFormat = (hostThree as any).RGIntegerFormat;
export const SRGBColorSpace = hostThree.SRGBColorSpace;
export const ShaderChunk = hostThree.ShaderChunk;
export const ShaderMaterial = hostThree.ShaderMaterial;
export const Scene = hostThree.Scene;
export const Texture = hostThree.Texture;
export const UnsignedByteType = hostThree.UnsignedByteType;
export const UnsignedIntType = hostThree.UnsignedIntType;
export const UnsignedShortType = hostThree.UnsignedShortType;
export const Vector2 = hostThree.Vector2;
export const Vector3 = hostThree.Vector3;
export const Vector4 = hostThree.Vector4;
export const WebGLArrayRenderTarget = (hostThree as any).WebGLArrayRenderTarget;
export const WebGLCubeRenderTarget = hostThree.WebGLCubeRenderTarget;
export const WebGLRenderTarget = hostThree.WebGLRenderTarget;
