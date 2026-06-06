import type * as DreiTypes from '@react-three/drei';

type WitwinGlobals = { __witwin?: { drei?: typeof DreiTypes } };

const drei = (globalThis as WitwinGlobals).__witwin?.drei;
if (!drei?.Text) {
  throw new Error('[wt_splatting] Drei host singleton is incomplete');
}

export const Html = drei.Html;
export const Text = drei.Text;
