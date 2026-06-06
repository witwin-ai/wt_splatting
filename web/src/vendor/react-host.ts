import type * as ReactTypes from 'react';

type WitwinGlobals = { __witwin?: { react?: typeof ReactTypes } };

const React = (globalThis as WitwinGlobals).__witwin?.react;
if (!React) {
  throw new Error('[wt_splatting] React host singleton is unavailable');
}

export default React;
export const createElement = React.createElement;
export const Fragment = React.Fragment;
export const useCallback = React.useCallback;
export const useEffect = React.useEffect;
export const useMemo = React.useMemo;
export const useRef = React.useRef;
export const useState = React.useState;
