export const depthKeyVertexShader = `precision highp float;
in vec3 position;
void main() {
  gl_Position = vec4(position.xy, 0.0, 1.0);
}
`;

export const depthKeyFragmentShader = `precision highp float;
uniform sampler2D centerOpacityTex;
uniform mat4 modelViewMatrixForSplats;
uniform vec2 textureSize;
uniform float splatCount;
out vec4 outKey;
void main() {
  ivec2 pixel = ivec2(gl_FragCoord.xy);
  int index = pixel.y * int(textureSize.x) + pixel.x;
  if (float(index) >= splatCount) {
    outKey = vec4(-3.402823e38, -1.0, 0.0, 0.0);
    return;
  }
  vec4 centerOpacity = texelFetch(centerOpacityTex, pixel, 0);
  vec4 viewCenter = modelViewMatrixForSplats * vec4(centerOpacity.xyz, 1.0);
  outKey = vec4(-viewCenter.z, float(index), 1.0, 0.0);
}
`;

export const bitonicSortFragmentShader = `precision highp float;
uniform sampler2D inputTex;
uniform vec2 textureSize;
uniform float sortTexelCount;
uniform float stage;
uniform float passOfStage;
out vec4 outKey;

vec4 readAt(int index) {
  if (index < 0 || float(index) >= sortTexelCount) {
    return vec4(-3.402823e38, -1.0, 0.0, 0.0);
  }
  int x = index - (index / int(textureSize.x)) * int(textureSize.x);
  int y = index / int(textureSize.x);
  return texelFetch(inputTex, ivec2(x, y), 0);
}

void main() {
  ivec2 pixel = ivec2(gl_FragCoord.xy);
  int index = pixel.y * int(textureSize.x) + pixel.x;
  int pairDistance = 1 << int(passOfStage);
  int blockSize = 1 << int(stage + 1.0);
  int partner = index ^ pairDistance;
  vec4 selfValue = readAt(index);
  vec4 partnerValue = readAt(partner);
  bool descendingBlock = (index & blockSize) == 0;
  bool pairIsLowerIndex = index < partner;
  bool takePartnerForLower = descendingBlock ? partnerValue.x > selfValue.x : partnerValue.x < selfValue.x;
  bool takePartnerForUpper = descendingBlock ? partnerValue.x < selfValue.x : partnerValue.x > selfValue.x;
  bool takePartner = pairIsLowerIndex ? takePartnerForLower : takePartnerForUpper;
  outKey = takePartner ? partnerValue : selfValue;
}
`;
