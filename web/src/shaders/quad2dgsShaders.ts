export const quad2dgsVertexShader = `precision highp float;
in vec3 position;
uniform sampler2D sortedIndexTex;
uniform sampler2D centerOpacityTex;
uniform sampler2D basisUTex;
uniform sampler2D basisVTex;
uniform sampler2D colorTex;
uniform vec2 textureSize;
uniform mat4 modelViewMatrix;
uniform mat4 projectionMatrix;
uniform float scaleModifier;
uniform bool weightedMode;
out vec2 vLocalUv;
out vec4 vColorOpacity;

ivec2 texelForIndex(float indexValue) {
  int index = int(indexValue + 0.5);
  int x = index - (index / int(textureSize.x)) * int(textureSize.x);
  int y = index / int(textureSize.x);
  return ivec2(x, y);
}

void main() {
  ivec2 splatTexel;
  if (weightedMode) {
    splatTexel = texelForIndex(float(gl_InstanceID));
  } else {
    ivec2 sortedTexel = texelForIndex(float(gl_InstanceID));
    vec4 sorted = texelFetch(sortedIndexTex, sortedTexel, 0);
    if (sorted.z < 0.5) {
      gl_Position = vec4(2.0, 2.0, 2.0, 1.0);
      return;
    }
    splatTexel = texelForIndex(sorted.y);
  }
  vec4 centerOpacity = texelFetch(centerOpacityTex, splatTexel, 0);
  vec4 basisU = texelFetch(basisUTex, splatTexel, 0);
  vec4 basisV = texelFetch(basisVTex, splatTexel, 0);
  vec4 color = texelFetch(colorTex, splatTexel, 0);
  vec2 localUv = position.xy;
  vec3 offset = basisU.xyz * basisU.w * localUv.x * scaleModifier + basisV.xyz * basisV.w * localUv.y * scaleModifier;
  vLocalUv = localUv;
  vColorOpacity = vec4(color.rgb, centerOpacity.w);
  gl_Position = projectionMatrix * modelViewMatrix * vec4(centerOpacity.xyz + offset, 1.0);
}
`;

export const quad2dgsFragmentShader = `precision highp float;
uniform float opacityScale;
uniform bool weightedMode;
in vec2 vLocalUv;
in vec4 vColorOpacity;
out vec4 outColor;
void main() {
  float gaussian = exp(-0.5 * dot(vLocalUv, vLocalUv));
  float alpha = vColorOpacity.a * opacityScale * gaussian;
  if (alpha < 0.01) discard;
  if (weightedMode) {
    outColor = vec4(vColorOpacity.rgb * alpha, alpha);
    return;
  }
  outColor = vec4(vColorOpacity.rgb, alpha);
}
`;
