export interface TwoDGaussianSplats {
  splatCount: number;
  shDegree: number;
  attributes: Record<string, unknown>;
  centers: Float32Array;
  tangentU: Float32Array;
  tangentV: Float32Array;
  sigmaUv: Float32Array;
  opacity: Float32Array;
  colorDc: Float32Array;
}

const HEADER_SIZE = 24;
const BYTES_PER_SPLAT = (3 + 3 + 3 + 2 + 1 + 3) * 4;

export function parseW2gs(buffer: ArrayBuffer): TwoDGaussianSplats {
  if (buffer.byteLength < HEADER_SIZE) {
    throw new Error('2DGS payload is too small');
  }
  const magic = new TextDecoder().decode(new Uint8Array(buffer, 0, 4));
  if (magic !== 'W2GS') {
    throw new Error('Invalid 2DGS payload: bad magic number');
  }
  const view = new DataView(buffer);
  const version = view.getUint32(4, true);
  if (version !== 1) {
    throw new Error(`Unsupported 2DGS payload version: ${version}`);
  }
  const splatCount = view.getUint32(8, true);
  const shDegree = view.getUint32(16, true);
  const attrsLength = view.getUint32(20, true);
  const attrsStart = HEADER_SIZE;
  const attrsEnd = attrsStart + attrsLength;
  const payloadOffset = align4(attrsEnd);
  const payloadEnd = payloadOffset + splatCount * BYTES_PER_SPLAT;
  if (buffer.byteLength < payloadEnd) {
    throw new Error('2DGS payload is truncated');
  }
  const attributes = attrsLength
    ? JSON.parse(new TextDecoder().decode(new Uint8Array(buffer, attrsStart, attrsLength)))
    : {};
  let offset = payloadOffset;
  const centers = copyFloats(buffer, offset, splatCount * 3);
  offset += splatCount * 3 * 4;
  const tangentU = copyFloats(buffer, offset, splatCount * 3);
  offset += splatCount * 3 * 4;
  const tangentV = copyFloats(buffer, offset, splatCount * 3);
  offset += splatCount * 3 * 4;
  const sigmaUv = copyFloats(buffer, offset, splatCount * 2);
  offset += splatCount * 2 * 4;
  const opacity = copyFloats(buffer, offset, splatCount);
  offset += splatCount * 4;
  const colorDc = copyFloats(buffer, offset, splatCount * 3);

  return { splatCount, shDegree, attributes, centers, tangentU, tangentV, sigmaUv, opacity, colorDc };
}

function copyFloats(buffer: ArrayBuffer, byteOffset: number, count: number): Float32Array {
  return new Float32Array(new Float32Array(buffer, byteOffset, count));
}

function align4(value: number): number {
  return (value + 3) & ~3;
}
