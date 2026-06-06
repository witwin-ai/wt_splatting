import type * as FiberTypes from '@react-three/fiber';

type WitwinGlobals = { __witwin?: { fiber?: typeof FiberTypes } };

const fiber = (globalThis as WitwinGlobals).__witwin?.fiber;
if (!fiber) {
  throw new Error('[wt_splatting] R3F host singleton is unavailable');
}

export const useFrame = fiber.useFrame;
export const useThree = fiber.useThree;
