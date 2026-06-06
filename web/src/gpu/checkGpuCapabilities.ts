import { THREE } from '../runtime';
import type * as ThreeTypes from 'three';

export interface SplatGpuCapabilities {
  supported: boolean;
  reason?: string;
}

export function checkGpuCapabilities(gl: ThreeTypes.WebGLRenderer): SplatGpuCapabilities {
  const context = gl.getContext();
  if (!(context instanceof WebGL2RenderingContext)) {
    return { supported: false, reason: '2DGS requires WebGL2.' };
  }
  const floatTarget = context.getExtension('EXT_color_buffer_float');
  if (!floatTarget) {
    return { supported: false, reason: '2DGS requires EXT_color_buffer_float.' };
  }
  if (!context.getParameter(context.MAX_VERTEX_TEXTURE_IMAGE_UNITS)) {
    return { supported: false, reason: '2DGS requires vertex texture fetch.' };
  }
  return { supported: true };
}
