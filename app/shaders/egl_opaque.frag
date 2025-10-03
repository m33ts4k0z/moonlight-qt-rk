#version 300 es
#extension GL_OES_EGL_image_external_essl3 : require
precision mediump float;
out vec4 FragColor;

in vec2 vTextCoord;

uniform samplerExternalOES uTexture;

void main() {
        FragColor = texture(uTexture, vTextCoord);
}
