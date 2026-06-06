import { THREE } from '../runtime';
import type * as ThreeTypes from 'three';

interface FullscreenTargetPassState {
  renderTarget: ThreeTypes.WebGLRenderTarget | null;
  viewport: ThreeTypes.Vector4;
  scissor: ThreeTypes.Vector4;
  scissorTest: boolean;
  autoClear: boolean;
  xrEnabled: boolean;
}

export function beginFullscreenTargetPass(
  renderer: ThreeTypes.WebGLRenderer,
  target: ThreeTypes.WebGLRenderTarget
): FullscreenTargetPassState {
  const state: FullscreenTargetPassState = {
    renderTarget: renderer.getRenderTarget(),
    viewport: renderer.getViewport(new THREE.Vector4()),
    scissor: renderer.getScissor(new THREE.Vector4()),
    scissorTest: renderer.getScissorTest(),
    autoClear: renderer.autoClear,
    xrEnabled: renderer.xr.enabled,
  };
  renderer.xr.enabled = false;
  renderer.setRenderTarget(target);
  renderer.setViewport(0, 0, target.width, target.height);
  renderer.setScissor(0, 0, target.width, target.height);
  renderer.setScissorTest(false);
  renderer.autoClear = false;
  return state;
}

export function finishFullscreenTargetPass(
  renderer: ThreeTypes.WebGLRenderer,
  state: FullscreenTargetPassState
): void {
  renderer.setRenderTarget(state.renderTarget);
  renderer.setViewport(state.viewport);
  renderer.setScissor(state.scissor);
  renderer.setScissorTest(state.scissorTest);
  renderer.autoClear = state.autoClear;
  renderer.xr.enabled = state.xrEnabled;
}
