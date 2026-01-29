// vim: noai:ts=4:sw=4:softtabstop=4:expandtab
#include "eglvid.h"

#include "path.h"
#include "utils.h"
#include "streaming/session.h"
#include "streaming/streamutils.h"

#include <QDir>

#include <Limelight.h>
#include <unistd.h>
#include <dlfcn.h>
#include <cstring>

#include <SDL_syswm.h>

// These are extensions, so some platform headers may not provide them
#ifndef GL_UNPACK_ROW_LENGTH_EXT
#define GL_UNPACK_ROW_LENGTH_EXT 0x0CF2
#endif

typedef struct _VERTEX
{
    float x, y;
    float u, v;
} VERTEX, *PVERTEX;

/* TODO:
 *  - handle more pixel formats
 *  - handle software decoding
 */

/* DOC/misc:
 *  - https://kernel-recipes.org/en/2016/talks/video-and-colorspaces/
 *  - http://www.brucelindbloom.com/
 *  - https://learnopengl.com/Getting-started/Shaders
 *  - https://github.com/stunpix/yuvit
 *  - https://en.wikipedia.org/wiki/YCbCr#ITU-R_BT.601_conversion
 *  - https://www.renesas.com/eu/en/www/doc/application-note/an9717.pdf
 *  - https://www.xilinx.com/support/documentation/application_notes/xapp283.pdf
 *  - https://www.itu.int/dms_pubrec/itu-r/rec/bt/R-REC-BT.709-6-201506-I!!PDF-E.pdf
 *  - https://www.khronos.org/registry/OpenGL/extensions/OES/OES_EGL_image_external.txt
 *  - https://gist.github.com/rexguo/6696123
 *  - https://wiki.libsdl.org/CategoryVideo
 */

#define EGL_LOG(Category, ...) SDL_Log ## Category(\
        SDL_LOG_CATEGORY_APPLICATION, \
        "EGLRenderer: " __VA_ARGS__)

EGLRenderer::EGLRenderer(IFFmpegRenderer *backendRenderer)
    :
        IFFmpegRenderer(RendererType::EGL),
        m_EGLImagePixelFormat(AV_PIX_FMT_NONE),
        m_EGLDisplay(EGL_NO_DISPLAY),
        m_Textures{0},
        m_OverlayTextures{0},
        m_OverlayVBOs{0},
        m_OverlayVAOs{0},
        m_OverlayHasValidData{},
        m_ShaderProgram(0),
        m_OverlayShaderProgram(0),
        m_Context(0),
        m_Window(nullptr),
        m_Backend(backendRenderer),
        m_VideoVAO(0),
        m_BlockingSwapBuffers(false),
        m_LastRenderSync(EGL_NO_SYNC),
        m_glEGLImageTargetTexture2DOES(nullptr),
        m_glGenVertexArraysOES(nullptr),
        m_glBindVertexArrayOES(nullptr),
        m_glDeleteVertexArraysOES(nullptr),
        m_eglCreateSync(nullptr),
        m_eglCreateSyncKHR(nullptr),
        m_eglDestroySync(nullptr),
        m_eglClientWaitSync(nullptr),
        m_GlesMajorVersion(0),
        m_GlesMinorVersion(0),
        m_HasExtUnpackSubimage(false)
{
    SDL_assert(backendRenderer);
    SDL_assert(backendRenderer->canExportEGL());
}

EGLRenderer::~EGLRenderer()
{
    if (m_Context) {
        // Reattach the GL context to the main thread for destruction
        SDL_GL_MakeCurrent(m_Window, m_Context);
        if (m_LastRenderSync != EGL_NO_SYNC) {
            SDL_assert(m_eglDestroySync != nullptr);
            m_eglDestroySync(m_EGLDisplay, m_LastRenderSync);
        }
        if (m_ShaderProgram) {
            glDeleteProgram(m_ShaderProgram);
        }
        if (m_OverlayShaderProgram) {
            glDeleteProgram(m_OverlayShaderProgram);
        }
        if (m_VideoVAO) {
            SDL_assert(m_glDeleteVertexArraysOES != nullptr);
            m_glDeleteVertexArraysOES(1, &m_VideoVAO);
        }
        glDeleteTextures(EGL_MAX_PLANES, m_Textures);

        glDeleteTextures(Overlay::OverlayMax, m_OverlayTextures);
        glDeleteBuffers(Overlay::OverlayMax, m_OverlayVBOs);
        if (m_glDeleteVertexArraysOES) {
            m_glDeleteVertexArraysOES(Overlay::OverlayMax, m_OverlayVAOs);
        }

        SDL_GL_DeleteContext(m_Context);
    }
}

bool EGLRenderer::prepareDecoderContext(AVCodecContext*, AVDictionary**)
{
    /* Nothing to do */

    EGL_LOG(Info, "Using EGL renderer");

    return true;
}

void EGLRenderer::notifyOverlayUpdated(Overlay::OverlayType type)
{
    // We handle uploading the updated overlay texture in renderOverlay().
    // notifyOverlayUpdated() is called on an arbitrary thread, which may
    // not be have the OpenGL context current on it.

    if (!Session::get()->getOverlayManager().isOverlayEnabled(type)) {
        // If the overlay has been disabled, mark the data as invalid/stale.
        SDL_AtomicSet(&m_OverlayHasValidData[type], 0);
        return;
    }
}

bool EGLRenderer::notifyWindowChanged(PWINDOW_STATE_CHANGE_INFO info)
{
    // We can transparently handle size and display changes
    return !(info->stateChangeFlags & ~(WINDOW_STATE_CHANGE_SIZE | WINDOW_STATE_CHANGE_DISPLAY));
}

bool EGLRenderer::isPixelFormatSupported(int videoFormat, AVPixelFormat pixelFormat)
{
    // Pixel format support should be determined by the backend renderer
    return m_Backend->isPixelFormatSupported(videoFormat, pixelFormat);
}

AVPixelFormat EGLRenderer::getPreferredPixelFormat(int videoFormat)
{
    // Pixel format preference should be determined by the backend renderer
    return m_Backend->getPreferredPixelFormat(videoFormat);
}

void EGLRenderer::renderOverlay(Overlay::OverlayType type, int viewportWidth, int viewportHeight)
{
    // Mali blob workaround: Get function pointers for all GL calls
    typedef void (*PFNGLBINDTEXTUREPROC)(GLenum target, GLuint texture);
    typedef void (*PFNGLPIXELSTOREIPROC)(GLenum pname, GLint param);
    typedef void (*PFNGLTEXIMAGE2DPROC)(GLenum target, GLint level, GLint internalformat, GLsizei width, GLsizei height, GLint border, GLenum format, GLenum type, const void *pixels);
    typedef void (*PFNGLBINDBUFFERPROC)(GLenum target, GLuint buffer);
    typedef void (*PFNGLBUFFERDATAPROC)(GLenum target, GLsizeiptr size, const void *data, GLenum usage);
    typedef void (*PFNGLUSEPROGRAMPROC)(GLuint program);
    typedef void (*PFNGLACTIVETEXTUREPROC)(GLenum texture);
    typedef void (*PFNGLDRAWARRAYSPROC)(GLenum mode, GLint first, GLsizei count);

    PFNGLBINDTEXTUREPROC glBindTextureFn = (PFNGLBINDTEXTUREPROC)SDL_GL_GetProcAddress("glBindTexture");
    PFNGLPIXELSTOREIPROC glPixelStoreiFn = (PFNGLPIXELSTOREIPROC)SDL_GL_GetProcAddress("glPixelStorei");
    PFNGLTEXIMAGE2DPROC glTexImage2DFn = (PFNGLTEXIMAGE2DPROC)SDL_GL_GetProcAddress("glTexImage2D");
    PFNGLBINDBUFFERPROC glBindBufferFn = (PFNGLBINDBUFFERPROC)SDL_GL_GetProcAddress("glBindBuffer");
    PFNGLBUFFERDATAPROC glBufferDataFn = (PFNGLBUFFERDATAPROC)SDL_GL_GetProcAddress("glBufferData");
    PFNGLUSEPROGRAMPROC glUseProgramFn = (PFNGLUSEPROGRAMPROC)SDL_GL_GetProcAddress("glUseProgram");
    PFNGLACTIVETEXTUREPROC glActiveTextureFn = (PFNGLACTIVETEXTUREPROC)SDL_GL_GetProcAddress("glActiveTexture");
    PFNGLDRAWARRAYSPROC glDrawArraysFn = (PFNGLDRAWARRAYSPROC)SDL_GL_GetProcAddress("glDrawArrays");

    // Do nothing if this overlay is disabled
    if (!Session::get()->getOverlayManager().isOverlayEnabled(type)) {
        return;
    }

    // Upload a new overlay texture if needed
    SDL_Surface* newSurface = Session::get()->getOverlayManager().getUpdatedOverlaySurface(type);
    if (newSurface != nullptr) {
        SDL_assert(!SDL_MUSTLOCK(newSurface));
        SDL_assert(newSurface->format->format == SDL_PIXELFORMAT_ARGB8888);

        if (glBindTextureFn) glBindTextureFn(GL_TEXTURE_2D, m_OverlayTextures[type]);

        // If the pixel data isn't tightly packed, it requires special handling
        void* packedPixelData = nullptr;
        if (newSurface->pitch != newSurface->w * newSurface->format->BytesPerPixel) {
            if (m_GlesMajorVersion >= 3 || m_HasExtUnpackSubimage) {
                // If we are GLES 3.0+ or have GL_EXT_unpack_subimage, GL can handle any pitch
                SDL_assert(newSurface->pitch % newSurface->format->BytesPerPixel == 0);
                if (glPixelStoreiFn) glPixelStoreiFn(GL_UNPACK_ROW_LENGTH_EXT, newSurface->pitch / newSurface->format->BytesPerPixel);
            }
            else {
                // If we can't use GL_UNPACK_ROW_LENGTH, we must allocate a tightly packed buffer
                // and copy our pixels there.
                packedPixelData = malloc(newSurface->w * newSurface->h * newSurface->format->BytesPerPixel);
                if (!packedPixelData) {
                    SDL_FreeSurface(newSurface);
                    return;
                }

                SDL_ConvertPixels(newSurface->w, newSurface->h,
                                  newSurface->format->format, newSurface->pixels, newSurface->pitch,
                                  newSurface->format->format, packedPixelData, newSurface->w * newSurface->format->BytesPerPixel);
            }
        }

        if (glTexImage2DFn) glTexImage2DFn(GL_TEXTURE_2D, 0, GL_RGBA, newSurface->w, newSurface->h, 0, GL_RGBA, GL_UNSIGNED_BYTE,
                     packedPixelData ? packedPixelData : newSurface->pixels);

        if (packedPixelData) {
            free(packedPixelData);
        }
        else if (newSurface->pitch != newSurface->w * newSurface->format->BytesPerPixel) {
            if (glPixelStoreiFn) glPixelStoreiFn(GL_UNPACK_ROW_LENGTH_EXT, 0);
        }

        SDL_FRect overlayRect;

        // These overlay positions differ from the other renderers because OpenGL
        // places the origin in the lower-left corner instead of the upper-left.
        if (type == Overlay::OverlayStatusUpdate) {
            // Bottom Left
            overlayRect.x = 0;
            overlayRect.y = 0;
        }
        else if (type == Overlay::OverlayDebug) {
            // Top left
            overlayRect.x = 0;
            overlayRect.y = viewportHeight - newSurface->h;
        } else {
            SDL_assert(false);
        }

        overlayRect.w = newSurface->w;
        overlayRect.h = newSurface->h;

        SDL_FreeSurface(newSurface);

        // Convert screen space to normalized device coordinates
        StreamUtils::screenSpaceToNormalizedDeviceCoords(&overlayRect, viewportWidth, viewportHeight);

        VERTEX verts[] =
        {
            {overlayRect.x + overlayRect.w, overlayRect.y + overlayRect.h, 1.0f, 0.0f},
            {overlayRect.x, overlayRect.y + overlayRect.h, 0.0f, 0.0f},
            {overlayRect.x, overlayRect.y, 0.0f, 1.0f},
            {overlayRect.x, overlayRect.y, 0.0f, 1.0f},
            {overlayRect.x + overlayRect.w, overlayRect.y, 1.0f, 1.0f},
            {overlayRect.x + overlayRect.w, overlayRect.y + overlayRect.h, 1.0f, 0.0f}
        };

        // Update the VBO for this overlay (already bound to a VAO)
        if (glBindBufferFn) glBindBufferFn(GL_ARRAY_BUFFER, m_OverlayVBOs[type]);
        if (glBufferDataFn) glBufferDataFn(GL_ARRAY_BUFFER, sizeof(verts), verts, GL_STATIC_DRAW);

        SDL_AtomicSet(&m_OverlayHasValidData[type], 1);
    }

    if (!SDL_AtomicGet(&m_OverlayHasValidData[type])) {
        // If the overlay is not populated yet or is stale, don't render it.
        return;
    }

    // Adjust the viewport to the whole window before rendering the overlays
    typedef void (*PFNGLVIEWPORTPROC)(GLint x, GLint y, GLsizei width, GLsizei height);
    typedef void (*PFNGLENABLEPROC)(GLenum cap);
    typedef void (*PFNGLDISABLEPROC)(GLenum cap);
    PFNGLVIEWPORTPROC glViewportFn = (PFNGLVIEWPORTPROC)SDL_GL_GetProcAddress("glViewport");
    PFNGLENABLEPROC glEnableFn = (PFNGLENABLEPROC)SDL_GL_GetProcAddress("glEnable");
    PFNGLDISABLEPROC glDisableFn = (PFNGLDISABLEPROC)SDL_GL_GetProcAddress("glDisable");
    
    if (glViewportFn) glViewportFn(0, 0, viewportWidth, viewportHeight);

    if (glUseProgramFn) glUseProgramFn(m_OverlayShaderProgram);

    if (glActiveTextureFn) glActiveTextureFn(GL_TEXTURE0);
    if (glBindTextureFn) glBindTextureFn(GL_TEXTURE_2D, m_OverlayTextures[type]);

    // Temporarily enable blending to draw the overlays with alpha
    if (glEnableFn) glEnableFn(GL_BLEND);

    // Draw the overlay
    m_glBindVertexArrayOES(m_OverlayVAOs[type]);
    if (glDrawArraysFn) glDrawArraysFn(GL_TRIANGLES, 0, 6);
    m_glBindVertexArrayOES(0);

    if (glDisableFn) glDisableFn(GL_BLEND);
}

int EGLRenderer::loadAndBuildShader(int shaderType,
                                    const char *file) {
    // Mali blob workaround: Use SDL_GL_GetProcAddress to get function pointers
    // SDL's GL symbols are NULL with Mali blob on Wayland
    typedef GLuint (*PFNGLCREATESHADERPROC)(GLenum type);
    typedef void (*PFNGLSHADERSOURCEPROC)(GLuint shader, GLsizei count, const GLchar *const*string, const GLint *length);
    typedef void (*PFNGLCOMPILESHADERPROC)(GLuint shader);
    typedef void (*PFNGLGETSHADERIVPROC)(GLuint shader, GLenum pname, GLint *params);
    typedef void (*PFNGLGETSHADERINFOLOGPROC)(GLuint shader, GLsizei bufSize, GLsizei *length, GLchar *infoLog);
    typedef void (*PFNGLDELETESHADERPROC)(GLuint shader);
    typedef GLenum (*PFNGLGETERRORPROC)(void);
    
    PFNGLCREATESHADERPROC glCreateShaderFn = (PFNGLCREATESHADERPROC)SDL_GL_GetProcAddress("glCreateShader");
    PFNGLSHADERSOURCEPROC glShaderSourceFn = (PFNGLSHADERSOURCEPROC)SDL_GL_GetProcAddress("glShaderSource");
    PFNGLCOMPILESHADERPROC glCompileShaderFn = (PFNGLCOMPILESHADERPROC)SDL_GL_GetProcAddress("glCompileShader");
    PFNGLGETSHADERIVPROC glGetShaderivFn = (PFNGLGETSHADERIVPROC)SDL_GL_GetProcAddress("glGetShaderiv");
    PFNGLGETSHADERINFOLOGPROC glGetShaderInfoLogFn = (PFNGLGETSHADERINFOLOGPROC)SDL_GL_GetProcAddress("glGetShaderInfoLog");
    PFNGLDELETESHADERPROC glDeleteShaderFn = (PFNGLDELETESHADERPROC)SDL_GL_GetProcAddress("glDeleteShader");
    PFNGLGETERRORPROC glGetErrorFn = (PFNGLGETERRORPROC)SDL_GL_GetProcAddress("glGetError");
    
    if (!glCreateShaderFn || !glShaderSourceFn || !glCompileShaderFn || !glGetShaderivFn || !glGetShaderInfoLogFn) {
        EGL_LOG(Error, "Failed to load GL functions via SDL_GL_GetProcAddress");
        return 0;
    }

    // Clear any lingering GL errors
    if (glGetErrorFn) {
        GLenum priorError;
        while ((priorError = glGetErrorFn()) != GL_NO_ERROR) {
            EGL_LOG(Warn, "Clearing prior GL error: 0x%x", priorError);
        }
    }

    GLuint shader = glCreateShaderFn(shaderType);
    if (!shader) {
        EGL_LOG(Error, "glCreateShader(%d) returned 0", shaderType);
        return 0;
    }

    auto sourceData = Path::readDataFile(file);
    if (sourceData.isEmpty()) {
        EGL_LOG(Error, "Shader file \"%s\" is empty or could not be read", file);
        if (glDeleteShaderFn) glDeleteShaderFn(shader);
        return 0;
    }
    
    GLint len = sourceData.size();
    const char *buf = sourceData.data();

    glShaderSourceFn(shader, 1, &buf, &len);
    glCompileShaderFn(shader);
    GLint status;
    glGetShaderivFn(shader, GL_COMPILE_STATUS, &status);
    if (!status) {
        char shaderLog[512];
        glGetShaderInfoLogFn(shader, sizeof (shaderLog), nullptr, shaderLog);
        EGL_LOG(Error, "Cannot load shader \"%s\": %s", file, shaderLog);
        if (glDeleteShaderFn) glDeleteShaderFn(shader);
        return 0;
    }

    return shader;
}

unsigned EGLRenderer::compileShader(const char* vertexShaderSrc, const char* fragmentShaderSrc) {
    // Mali blob workaround: Use SDL_GL_GetProcAddress for all GL functions
    typedef GLuint (*PFNGLCREATEPROGRAMPROC)(void);
    typedef void (*PFNGLATTACHSHADERPROC)(GLuint program, GLuint shader);
    typedef void (*PFNGLBINDATTRIBLOCATIONPROC)(GLuint program, GLuint index, const GLchar *name);
    typedef void (*PFNGLLINKPROGRAMPROC)(GLuint program);
    typedef void (*PFNGLGETPROGRAMIVPROC)(GLuint program, GLenum pname, GLint *params);
    typedef void (*PFNGLGETPROGRAMINFOLOGPROC)(GLuint program, GLsizei bufSize, GLsizei *length, GLchar *infoLog);
    typedef void (*PFNGLDELETEPROGRAMPROC)(GLuint program);
    typedef void (*PFNGLDELETESHADERPROC)(GLuint shader);
    
    PFNGLCREATEPROGRAMPROC glCreateProgramFn = (PFNGLCREATEPROGRAMPROC)SDL_GL_GetProcAddress("glCreateProgram");
    PFNGLATTACHSHADERPROC glAttachShaderFn = (PFNGLATTACHSHADERPROC)SDL_GL_GetProcAddress("glAttachShader");
    PFNGLBINDATTRIBLOCATIONPROC glBindAttribLocationFn = (PFNGLBINDATTRIBLOCATIONPROC)SDL_GL_GetProcAddress("glBindAttribLocation");
    PFNGLLINKPROGRAMPROC glLinkProgramFn = (PFNGLLINKPROGRAMPROC)SDL_GL_GetProcAddress("glLinkProgram");
    PFNGLGETPROGRAMIVPROC glGetProgramivFn = (PFNGLGETPROGRAMIVPROC)SDL_GL_GetProcAddress("glGetProgramiv");
    PFNGLGETPROGRAMINFOLOGPROC glGetProgramInfoLogFn = (PFNGLGETPROGRAMINFOLOGPROC)SDL_GL_GetProcAddress("glGetProgramInfoLog");
    PFNGLDELETEPROGRAMPROC glDeleteProgramFn = (PFNGLDELETEPROGRAMPROC)SDL_GL_GetProcAddress("glDeleteProgram");
    PFNGLDELETESHADERPROC glDeleteShaderFn = (PFNGLDELETESHADERPROC)SDL_GL_GetProcAddress("glDeleteShader");
    
    if (!glCreateProgramFn || !glAttachShaderFn || !glBindAttribLocationFn || !glLinkProgramFn || !glGetProgramivFn || !glGetProgramInfoLogFn) {
        EGL_LOG(Error, "Failed to load GL program functions via SDL_GL_GetProcAddress");
        return 0;
    }
    
    unsigned shader = 0;

    GLuint vertexShader = loadAndBuildShader(GL_VERTEX_SHADER, vertexShaderSrc);
    if (!vertexShader)
        return false;

    GLuint fragmentShader = loadAndBuildShader(GL_FRAGMENT_SHADER, fragmentShaderSrc);
    if (!fragmentShader)
        goto fragError;

    shader = glCreateProgramFn();
    if (!shader) {
        EGL_LOG(Error, "Cannot create shader program");
        goto progFailCreate;
    }

    glAttachShaderFn(shader, vertexShader);
    glAttachShaderFn(shader, fragmentShader);

    // Bind specific attribute locations for our standard vertex shader arguments
    glBindAttribLocationFn(shader, 0, "aPosition");
    glBindAttribLocationFn(shader, 1, "aTexCoord");

    glLinkProgramFn(shader);
    int status;
    glGetProgramivFn(shader, GL_LINK_STATUS, &status);
    if (!status) {
        char shader_log[512];
        glGetProgramInfoLogFn(shader, sizeof (shader_log), nullptr, shader_log);
        EGL_LOG(Error, "Cannot link shader program: %s", shader_log);
        if (glDeleteProgramFn) glDeleteProgramFn(shader);
        shader = 0;
    } 

progFailCreate:
    if (glDeleteShaderFn) glDeleteShaderFn(fragmentShader);
fragError:
    if (glDeleteShaderFn) glDeleteShaderFn(vertexShader);
    return shader;
}

bool EGLRenderer::compileShaders() {
    SDL_assert(!m_ShaderProgram);
    SDL_assert(!m_OverlayShaderProgram);

    SDL_assert(m_EGLImagePixelFormat != AV_PIX_FMT_NONE);

    // Mali blob workaround: Get glGetUniformLocation via SDL_GL_GetProcAddress
    typedef GLint (*PFNGLGETUNIFORMLOCATIONPROC)(GLuint program, const GLchar *name);
    PFNGLGETUNIFORMLOCATIONPROC glGetUniformLocationFn = (PFNGLGETUNIFORMLOCATIONPROC)SDL_GL_GetProcAddress("glGetUniformLocation");
    
    if (!glGetUniformLocationFn) {
        EGL_LOG(Error, "Failed to get glGetUniformLocation function pointer");
        return false;
    }

    // XXX: TODO: other formats
    if (m_EGLImagePixelFormat == AV_PIX_FMT_NV12 || m_EGLImagePixelFormat == AV_PIX_FMT_P010) {
        m_ShaderProgram = compileShader("egl.vert", "egl_nv12.frag");
        if (!m_ShaderProgram) {
            return false;
        }

        m_ShaderProgramParams[NV12_PARAM_YUVMAT] = glGetUniformLocationFn(m_ShaderProgram, "yuvmat");
        m_ShaderProgramParams[NV12_PARAM_OFFSET] = glGetUniformLocationFn(m_ShaderProgram, "offset");
        m_ShaderProgramParams[NV12_PARAM_CHROMA_OFFSET] = glGetUniformLocationFn(m_ShaderProgram, "chromaOffset");
        m_ShaderProgramParams[NV12_PARAM_PLANE1] = glGetUniformLocationFn(m_ShaderProgram, "plane1");
        m_ShaderProgramParams[NV12_PARAM_PLANE2] = glGetUniformLocationFn(m_ShaderProgram, "plane2");

        // Set up constant uniforms
        typedef void (*PFNGLUSEPROGRAMPROC)(GLuint program);
        PFNGLUSEPROGRAMPROC glUseProgramFn = (PFNGLUSEPROGRAMPROC)SDL_GL_GetProcAddress("glUseProgram");
        typedef void (*PFNGLUNIFORM1IPROC)(GLint location, GLint v0);
        PFNGLUNIFORM1IPROC glUniform1iFn = (PFNGLUNIFORM1IPROC)SDL_GL_GetProcAddress("glUniform1i");
        if (glUseProgramFn && glUniform1iFn) {
            glUseProgramFn(m_ShaderProgram);
            glUniform1iFn(m_ShaderProgramParams[NV12_PARAM_PLANE1], 0);
            glUniform1iFn(m_ShaderProgramParams[NV12_PARAM_PLANE2], 1);
            glUseProgramFn(0);
        }
    }
    else if (m_EGLImagePixelFormat == AV_PIX_FMT_DRM_PRIME) {
        m_ShaderProgram = compileShader("egl.vert", "egl_opaque.frag");
        if (!m_ShaderProgram) {
            return false;
        }

        m_ShaderProgramParams[OPAQUE_PARAM_TEXTURE] = glGetUniformLocationFn(m_ShaderProgram, "uTexture");

        // Set up constant uniforms
        typedef void (*PFNGLUSEPROGRAMPROC)(GLuint program);
        PFNGLUSEPROGRAMPROC glUseProgramFn = (PFNGLUSEPROGRAMPROC)SDL_GL_GetProcAddress("glUseProgram");
        typedef void (*PFNGLUNIFORM1IPROC)(GLint location, GLint v0);
        PFNGLUNIFORM1IPROC glUniform1iFn = (PFNGLUNIFORM1IPROC)SDL_GL_GetProcAddress("glUniform1i");
        if (glUseProgramFn && glUniform1iFn) {
            glUseProgramFn(m_ShaderProgram);
            glUniform1iFn(m_ShaderProgramParams[OPAQUE_PARAM_TEXTURE], 0);
            glUseProgramFn(0);
        }
    }
    else {
        SDL_LogError(SDL_LOG_CATEGORY_APPLICATION,
                     "Unsupported EGL pixel format: %d",
                     m_EGLImagePixelFormat);
        SDL_assert(false);
        return false;
    }

    m_OverlayShaderProgram = compileShader("egl.vert", "egl_overlay.frag");
    if (!m_OverlayShaderProgram) {
        return false;
    }

    m_OverlayShaderProgramParams[OVERLAY_PARAM_TEXTURE] = glGetUniformLocationFn(m_OverlayShaderProgram, "uTexture");

    typedef void (*PFNGLUSEPROGRAMPROC)(GLuint program);
    typedef void (*PFNGLUNIFORM1IPROC)(GLint location, GLint v0);
    PFNGLUSEPROGRAMPROC glUseProgramFn = (PFNGLUSEPROGRAMPROC)SDL_GL_GetProcAddress("glUseProgram");
    PFNGLUNIFORM1IPROC glUniform1iFn = (PFNGLUNIFORM1IPROC)SDL_GL_GetProcAddress("glUniform1i");
    if (glUseProgramFn && glUniform1iFn) {
        glUseProgramFn(m_OverlayShaderProgram);
        glUniform1iFn(m_OverlayShaderProgramParams[OVERLAY_PARAM_TEXTURE], 0);
        glUseProgramFn(0);
    }

    // Setup the VAO and VBO for video rendering
    // This is critical for Mali blob driver - must be done after shader compilation
    static const VERTEX vertices[] = {
        // pos .... // tex coords
        { 1.0f, 1.0f, 1.0f, 0.0f },
        { -1.0f, 1.0f, 0.0f, 0.0f },
        { -1.0f, -1.0f, 0.0f, 1.0f },
        { -1.0f, -1.0f, 0.0f, 1.0f },
        { 1.0f, -1.0f, 1.0f, 1.0f },
        { 1.0f, 1.0f, 1.0f, 0.0f },
    };

    // Mali blob workaround: Get GL buffer/vertex functions
    typedef void (*PFNGLGENBUFFERSPROC)(GLsizei n, GLuint *buffers);
    typedef void (*PFNGLBINDBUFFERPROC)(GLenum target, GLuint buffer);
    typedef void (*PFNGLBUFFERDATAPROC)(GLenum target, GLsizeiptr size, const void *data, GLenum usage);
    typedef void (*PFNGLVERTEXATTRIBPOINTERPROC)(GLuint index, GLint size, GLenum type, GLboolean normalized, GLsizei stride, const void *pointer);
    typedef void (*PFNGLENABLEVERTEXATTRIBARRAYPROC)(GLuint index);
    typedef void (*PFNGLDELETEBUFFERSPROC)(GLsizei n, const GLuint *buffers);
    
    PFNGLGENBUFFERSPROC glGenBuffersFn = (PFNGLGENBUFFERSPROC)SDL_GL_GetProcAddress("glGenBuffers");
    PFNGLBINDBUFFERPROC glBindBufferFn = (PFNGLBINDBUFFERPROC)SDL_GL_GetProcAddress("glBindBuffer");
    PFNGLBUFFERDATAPROC glBufferDataFn = (PFNGLBUFFERDATAPROC)SDL_GL_GetProcAddress("glBufferData");
    PFNGLVERTEXATTRIBPOINTERPROC glVertexAttribPointerFn = (PFNGLVERTEXATTRIBPOINTERPROC)SDL_GL_GetProcAddress("glVertexAttribPointer");
    PFNGLENABLEVERTEXATTRIBARRAYPROC glEnableVertexAttribArrayFn = (PFNGLENABLEVERTEXATTRIBARRAYPROC)SDL_GL_GetProcAddress("glEnableVertexAttribArray");
    PFNGLDELETEBUFFERSPROC glDeleteBuffersFn = (PFNGLDELETEBUFFERSPROC)SDL_GL_GetProcAddress("glDeleteBuffers");
    
    if (!glGenBuffersFn || !glBindBufferFn || !glBufferDataFn || 
        !glVertexAttribPointerFn || !glEnableVertexAttribArrayFn || !glDeleteBuffersFn) {
        EGL_LOG(Error, "Failed to get GL buffer/vertex function pointers in compileShaders()");
        return false;
    }
    
    // Setup the VAO and VBO
    unsigned int VBO;
    m_glGenVertexArraysOES(1, &m_VideoVAO);
    glGenBuffersFn(1, &VBO);

    m_glBindVertexArrayOES(m_VideoVAO);

    glBindBufferFn(GL_ARRAY_BUFFER, VBO);
    glBufferDataFn(GL_ARRAY_BUFFER, sizeof(vertices), vertices, GL_STATIC_DRAW);

    // compileShader() ensures that aPosition and aTexCoord are indexes 0 and 1 respectively
    glVertexAttribPointerFn(0, 2, GL_FLOAT, GL_FALSE, 4 * sizeof(float), (void*)offsetof(VERTEX, x));
    glEnableVertexAttribArrayFn(0);
    glVertexAttribPointerFn(1, 2, GL_FLOAT, GL_FALSE, 4 * sizeof(float), (void*)offsetof(VERTEX, u));
    glEnableVertexAttribArrayFn(1);

    glBindBufferFn(GL_ARRAY_BUFFER, 0);
    m_glBindVertexArrayOES(0);

    glDeleteBuffersFn(1, &VBO);

    PFNGLGETERRORPROC glGetErrorFn = (PFNGLGETERRORPROC)SDL_GL_GetProcAddress("glGetError");
    GLenum err = glGetErrorFn ? glGetErrorFn() : GL_NO_ERROR;
    if (err != GL_NO_ERROR) {
        EGL_LOG(Error, "OpenGL error: %d", err);
    }

    return err == GL_NO_ERROR;
}

bool EGLRenderer::initialize(PDECODER_PARAMETERS params)
{
    m_Window = params->window;

    // It's not safe to attempt to opportunistically create a GLES2
    // renderer prior to 2.0.10. If GLES2 isn't available, SDL will
    // attempt to dereference a null pointer and crash Moonlight.
    // https://bugzilla.libsdl.org/show_bug.cgi?id=4350
    // https://hg.libsdl.org/SDL/rev/84618d571795
    //
    // SDL_HINT_VIDEO_X11_FORCE_EGL isn't supported until SDL 2.0.12
    // and we need to use EGL to avoid triggering a crash in Mesa.
    // https://gitlab.freedesktop.org/mesa/mesa/issues/1011
    if (!SDL_VERSION_ATLEAST(2, 0, 12)) {
        EGL_LOG(Error, "Not supported until SDL 2.0.12");
        m_InitFailureReason = InitFailureReason::NoSoftwareSupport;
        return false;
    }

    // This renderer doesn't support HDR, so pick a different one.
    // HACK: This avoids a deadlock in SDL_CreateRenderer() if
    // Vulkan was used before and SDL is trying to load EGL.
    if (params->videoFormat & VIDEO_FORMAT_MASK_10BIT) {
        EGL_LOG(Info, "EGL doesn't support HDR rendering");
        return false;
    }

    // Mali blob workaround: This hint will ensure we use EGL to retrieve our GL context,
    // even on X11 where that is not the default. EGL is required
    // to avoid a crash in Mesa.
    // https://gitlab.freedesktop.org/mesa/mesa/issues/1011
    SDL_SetHint(SDL_HINT_OPENGL_ES_DRIVER, "1");
    
    // SDL_GL_LoadLibrary is required for context creation with Mali blob
    if (SDL_GL_LoadLibrary(nullptr) != 0) {
        EGL_LOG(Error, "SDL_GL_LoadLibrary() failed: %s", SDL_GetError());
        m_InitFailureReason = InitFailureReason::NoSoftwareSupport;
        return false;
    }
    
    SDL_GL_SetAttribute(SDL_GL_CONTEXT_PROFILE_MASK, SDL_GL_CONTEXT_PROFILE_ES);
    
    // Request GLES 3.0 context for Mali blob compatibility
    SDL_GL_SetAttribute(SDL_GL_CONTEXT_MAJOR_VERSION, 3);
    SDL_GL_SetAttribute(SDL_GL_CONTEXT_MINOR_VERSION, 0);
    
    int renderIndex;
    int maxRenderers = SDL_GetNumRenderDrivers();
    SDL_assert(maxRenderers >= 0);

    SDL_RendererInfo renderInfo;
    for (renderIndex = 0; renderIndex < maxRenderers; ++renderIndex) {
        if (SDL_GetRenderDriverInfo(renderIndex, &renderInfo))
            continue;
        if (!strcmp(renderInfo.name, "opengles2")) {
            SDL_assert(renderInfo.flags & SDL_RENDERER_ACCELERATED);
            break;
        }
    }
    if (renderIndex == maxRenderers) {
        EGL_LOG(Error, "Could not find a suitable SDL_Renderer");
        m_InitFailureReason = InitFailureReason::NoSoftwareSupport;
        return false;
    }

    // This will load OpenGL ES and convert our window to SDL_WINDOW_OPENGL if necessary
    SDL_Renderer* dummyRenderer = SDL_CreateRenderer(m_Window, renderIndex, SDL_RENDERER_ACCELERATED);
    if (dummyRenderer) {
        SDL_DestroyRenderer(dummyRenderer);
        dummyRenderer = nullptr;
    }
    else {
        // Print the error here (before it gets clobbered), but ensure that we flush window
        // events just in case SDL re-created the window before eventually failing.
        EGL_LOG(Error, "SDL_CreateRenderer() failed: %s", SDL_GetError());
    }

    // SDL_CreateRenderer() can end up having to recreate our window (SDL_RecreateWindow())
    // to ensure it's compatible with the renderer's OpenGL context. If that happens, we
    // can get spurious SDL_WINDOWEVENT events that will cause us to (again) recreate our
    // renderer. This can lead to an infinite to renderer recreation, so discard all
    // SDL_WINDOWEVENT events after SDL_CreateRenderer().
    Session* session = Session::get();
    if (session != nullptr) {
        // If we get here during a session, we need to synchronize with the event loop
        // to ensure we don't drop any important events.
        session->flushWindowEvents();
    }
    else if (!params->testOnly) {
        // If we get here prior to the start of a session, just pump and flush ourselves.
        SDL_PumpEvents();
        SDL_FlushEvent(SDL_WINDOWEVENT);
    }

    SDL_SysWMinfo info;
    SDL_VERSION(&info.version);
    if (!SDL_GetWindowWMInfo(params->window, &info)) {
        EGL_LOG(Error, "SDL_GetWindowWMInfo() failed: %s", SDL_GetError());
        m_InitFailureReason = InitFailureReason::NoSoftwareSupport;
        return false;
    }

    if (!(m_Context = SDL_GL_CreateContext(params->window))) {
        EGL_LOG(Error, "Cannot create OpenGL context: %s", SDL_GetError());
        EGLint eglErr = eglGetError();
        EGL_LOG(Error, "eglGetError() after SDL_GL_CreateContext() failure: 0x%x", eglErr);
        m_InitFailureReason = InitFailureReason::NoSoftwareSupport;
        return false;
    }
    
    if (SDL_GL_MakeCurrent(params->window, m_Context)) {
        EGL_LOG(Error, "Cannot use created EGL context: %s", SDL_GetError());
        EGLint eglErr = eglGetError();
        EGL_LOG(Error, "eglGetError() after SDL_GL_MakeCurrent() failure: 0x%x", eglErr);
        m_InitFailureReason = InitFailureReason::NoSoftwareSupport;
        return false;
    }
    
    EGL_LOG(Info, "SDL_GL_MakeCurrent() succeeded in initialize()");
    
    // Get EGL context info
    EGLContext currentCtx = eglGetCurrentContext();
    EGLDisplay currentDpy = eglGetCurrentDisplay();
    
    // Query EGL version and client APIs
    EGLint eglMajor = 0;
    eglQueryContext(currentDpy, currentCtx, EGL_CONTEXT_CLIENT_VERSION, &eglMajor);
    
    
    // Try to get the EGL config
    EGLint configId = 0;
    eglQueryContext(currentDpy, currentCtx, EGL_CONFIG_ID, &configId);
    
    // Now try GL functions
    const char* glVersion = (const char*)glGetString(GL_VERSION);
    const char* glVendor = (const char*)glGetString(GL_VENDOR);
    const char* glRenderer = (const char*)glGetString(GL_RENDERER);
    
    if (!glVersion || !glVendor || !glRenderer) {
        EGL_LOG(Warn, "SDL's GL function pointers are NULL - trying SDL_GL_GetProcAddress workaround");
        EGL_LOG(Warn, "This is a known Mali blob issue on Wayland");
        
        // Mali blob + SDL + Wayland bug: SDL loads the library but doesn't bind functions
        // Workaround: Use SDL_GL_GetProcAddress which internally uses eglGetProcAddress
        typedef const GLubyte* (*PFNGLGETSTRINGPROC)(GLenum name);
        PFNGLGETSTRINGPROC glGetStringAlt = (PFNGLGETSTRINGPROC)SDL_GL_GetProcAddress("glGetString");
        
        if (!glGetStringAlt) {
            EGL_LOG(Error, "SDL_GL_GetProcAddress('glGetString') also failed!");
            m_InitFailureReason = InitFailureReason::NoSoftwareSupport;
            return false;
        }
        
        glVersion = (const char*)glGetStringAlt(GL_VERSION);
        glVendor = (const char*)glGetStringAlt(GL_VENDOR);
        glRenderer = (const char*)glGetStringAlt(GL_RENDERER);
        
        if (!glVersion || !glVendor || !glRenderer) {
            EGL_LOG(Error, "Even SDL_GL_GetProcAddress version failed!");
            m_InitFailureReason = InitFailureReason::NoSoftwareSupport;
            return false;
        }
    }    

    {
        int r, g, b, a;
        SDL_GL_GetAttribute(SDL_GL_RED_SIZE, &r);
        SDL_GL_GetAttribute(SDL_GL_GREEN_SIZE, &g);
        SDL_GL_GetAttribute(SDL_GL_BLUE_SIZE, &b);
        SDL_GL_GetAttribute(SDL_GL_ALPHA_SIZE, &a);
        SDL_LogInfo(SDL_LOG_CATEGORY_APPLICATION,
                    "Color buffer is: R%dG%dB%dA%d",
                    r, g, b, a);
    }

    SDL_GL_GetAttribute(SDL_GL_CONTEXT_MAJOR_VERSION, &m_GlesMajorVersion);
    SDL_GL_GetAttribute(SDL_GL_CONTEXT_MINOR_VERSION, &m_GlesMinorVersion);

    // We can use GL_UNPACK_ROW_LENGTH for a more optimized upload of non-tightly-packed textures
    m_HasExtUnpackSubimage = SDL_GL_ExtensionSupported("GL_EXT_unpack_subimage");

    m_EGLDisplay = eglGetCurrentDisplay();
    if (m_EGLDisplay == EGL_NO_DISPLAY) {
        EGL_LOG(Error, "Cannot get EGL display: %d", eglGetError());
        return false;
    }

    const EGLExtensions eglExtensions(m_EGLDisplay);
    if (!eglExtensions.isSupported("EGL_KHR_image_base") &&
        !eglExtensions.isSupported("EGL_KHR_image")) {
        EGL_LOG(Error, "EGL_KHR_image unsupported");
        return false;
    }
    else if (!SDL_GL_ExtensionSupported("GL_OES_EGL_image")) {
        EGL_LOG(Error, "GL_OES_EGL_image unsupported");
        return false;
    }

    if (!m_Backend->initializeEGL(m_EGLDisplay, eglExtensions))
        return false;

    if (!(m_glEGLImageTargetTexture2DOES = (typeof(m_glEGLImageTargetTexture2DOES))eglGetProcAddress("glEGLImageTargetTexture2DOES"))) {
        EGL_LOG(Error,
                "EGL: cannot retrieve `glEGLImageTargetTexture2DOES` address");
        return false;
    }

    // Vertex arrays are an extension on OpenGL ES 2.0
    if (SDL_GL_ExtensionSupported("GL_OES_vertex_array_object")) {
        m_glGenVertexArraysOES = (typeof(m_glGenVertexArraysOES))eglGetProcAddress("glGenVertexArraysOES");
        m_glBindVertexArrayOES = (typeof(m_glBindVertexArrayOES))eglGetProcAddress("glBindVertexArrayOES");
        m_glDeleteVertexArraysOES = (typeof(m_glDeleteVertexArraysOES))eglGetProcAddress("glDeleteVertexArraysOES");
    }
    else {
        // They are included in OpenGL ES 3.0 as part of the standard
        m_glGenVertexArraysOES = (typeof(m_glGenVertexArraysOES))eglGetProcAddress("glGenVertexArrays");
        m_glBindVertexArrayOES = (typeof(m_glBindVertexArrayOES))eglGetProcAddress("glBindVertexArray");
        m_glDeleteVertexArraysOES = (typeof(m_glDeleteVertexArraysOES))eglGetProcAddress("glDeleteVertexArrays");
    }

    if (!m_glGenVertexArraysOES || !m_glBindVertexArrayOES || !m_glDeleteVertexArraysOES) {
        EGL_LOG(Error, "Failed to find VAO functions");
        return false;
    }

    // EGL_KHR_fence_sync is an extension for EGL 1.1+
    if (eglExtensions.isSupported("EGL_KHR_fence_sync")) {
        // eglCreateSyncKHR() has a slightly different prototype to eglCreateSync()
        m_eglCreateSyncKHR = (typeof(m_eglCreateSyncKHR))eglGetProcAddress("eglCreateSyncKHR");
        m_eglDestroySync = (typeof(m_eglDestroySync))eglGetProcAddress("eglDestroySyncKHR");
        m_eglClientWaitSync = (typeof(m_eglClientWaitSync))eglGetProcAddress("eglClientWaitSyncKHR");
    }
    else {
        // EGL 1.5 introduced sync support to the core specification
        m_eglCreateSync = (typeof(m_eglCreateSync))eglGetProcAddress("eglCreateSync");
        m_eglDestroySync = (typeof(m_eglDestroySync))eglGetProcAddress("eglDestroySync");
        m_eglClientWaitSync = (typeof(m_eglClientWaitSync))eglGetProcAddress("eglClientWaitSync");
    }

    if (!(m_eglCreateSync || m_eglCreateSyncKHR) || !m_eglDestroySync || !m_eglClientWaitSync) {
        EGL_LOG(Warn, "Failed to find sync functions");

        // Sub-optimal, but not fatal
        m_eglCreateSync = nullptr;
        m_eglCreateSyncKHR = nullptr;
        m_eglDestroySync = nullptr;
        m_eglClientWaitSync = nullptr;
    }

    // SDL always uses swap interval 0 under the hood on Wayland systems,
    // because the compositor guarantees tear-free rendering. In this
    // situation, swap interval > 0 behaves as a frame pacing option
    // rather than a way to eliminate tearing as SDL will block in
    // SwapBuffers until the compositor consumes the frame. This will
    // needlessly increases latency, so we should avoid it.
    //
    // HACK: In SDL 2.0.22+ on GNOME systems with fractional DPI scaling,
    // the Wayland viewport can be stale when using Super+Left/Right/Up
    // to resize the window. This seems to happen significantly more often
    // with vsync enabled, so this also mitigates that problem too.
    if (params->enableVsync
#ifdef SDL_VIDEO_DRIVER_WAYLAND
            && info.subsystem != SDL_SYSWM_WAYLAND
#endif
            ) {
        SDL_GL_SetSwapInterval(1);

#if SDL_VERSION_ATLEAST(2, 0, 15) && defined(SDL_VIDEO_DRIVER_KMSDRM)
        // We don't use the fence to reduce latency on KMSDRM
        // because it can have severe performance impacts when
        // running on slow GPUs where the frame time exceeds
        // the video stream's frame interval. The latency
        // reduction is also less critical without a compositor
        // adding latency too.
        if (info.subsystem != SDL_SYSWM_KMSDRM)
#endif
        {
            m_BlockingSwapBuffers = true;
        }
    } else {
        SDL_GL_SetSwapInterval(0);
    }

    if (!setupVideoRenderingState() || !setupOverlayRenderingState()) {
        return false;
    }

    // Mali blob workaround: Get glGetError via SDL_GL_GetProcAddress
    typedef GLenum (*PFNGLGETERRORPROC)(void);
    PFNGLGETERRORPROC glGetErrorFn = (PFNGLGETERRORPROC)SDL_GL_GetProcAddress("glGetError");
    GLenum err = glGetErrorFn ? glGetErrorFn() : GL_NO_ERROR;
    if (err != GL_NO_ERROR)
        EGL_LOG(Error, "OpenGL error: %d", err);

    // Detach the context from this thread, so the render thread can attach it
    SDL_GL_MakeCurrent(m_Window, nullptr);

    if (err == GL_NO_ERROR) {
        // If we got a working GL implementation via EGL, avoid using GLX from now on.
        // GLX will cause problems if we later want to use EGL again on this window.
        SDL_LogInfo(SDL_LOG_CATEGORY_APPLICATION, "EGL passed preflight checks. Using EGL for GL context creation.");
        SDL_SetHint(SDL_HINT_VIDEO_X11_FORCE_EGL, "1");
    }

    return err == GL_NO_ERROR;
}

bool EGLRenderer::setupVideoRenderingState() {
    // Mali blob workaround: Get GL texture functions via SDL_GL_GetProcAddress
    typedef void (*PFNGLGENTEXTURESPROC)(GLsizei n, GLuint *textures);
    typedef void (*PFNGLBINDTEXTUREPROC)(GLenum target, GLuint texture);
    typedef void (*PFNGLTEXPARAMETERIPROC)(GLenum target, GLenum pname, GLint param);
    typedef GLenum (*PFNGLGETERRORPROC)(void);
    
    PFNGLGENTEXTURESPROC glGenTexturesFn = (PFNGLGENTEXTURESPROC)SDL_GL_GetProcAddress("glGenTextures");
    PFNGLBINDTEXTUREPROC glBindTextureFn = (PFNGLBINDTEXTUREPROC)SDL_GL_GetProcAddress("glBindTexture");
    PFNGLTEXPARAMETERIPROC glTexParameteriFn = (PFNGLTEXPARAMETERIPROC)SDL_GL_GetProcAddress("glTexParameteri");
    PFNGLGETERRORPROC glGetErrorFn = (PFNGLGETERRORPROC)SDL_GL_GetProcAddress("glGetError");
    
    if (!glGenTexturesFn || !glBindTextureFn || !glTexParameteriFn) {
        EGL_LOG(Error, "Failed to get GL texture function pointers in setupVideoRenderingState()");
        return false;
    }
    
    // Setup the video plane textures
    glGenTexturesFn(EGL_MAX_PLANES, m_Textures);
    for (size_t i = 0; i < EGL_MAX_PLANES; ++i) {
        glBindTextureFn(GL_TEXTURE_EXTERNAL_OES, m_Textures[i]);
        glTexParameteriFn(GL_TEXTURE_EXTERNAL_OES, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
        glTexParameteriFn(GL_TEXTURE_EXTERNAL_OES, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
        glTexParameteriFn(GL_TEXTURE_EXTERNAL_OES, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
        glTexParameteriFn(GL_TEXTURE_EXTERNAL_OES, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
        
        // Some drivers (Mali blob) may generate errors when setting parameters on external textures
        // Clear any errors after each texture setup
        if (glGetErrorFn) {
            GLenum texErr = glGetErrorFn();
            if (texErr != GL_NO_ERROR) {
                EGL_LOG(Warn, "GL error after setting external texture %zu parameters: 0x%x", i, texErr);
            }
        }
    }
    // Unbind to clean state
    glBindTextureFn(GL_TEXTURE_EXTERNAL_OES, 0);

    // Check for any GL errors and clear them
    // The Mali blob driver can be sensitive to lingering errors
    if (glGetErrorFn) {
        GLenum err = glGetErrorFn();
        while (err != GL_NO_ERROR) {
            EGL_LOG(Warn, "OpenGL error during video rendering state setup: 0x%x", err);
            err = glGetErrorFn();
        }
    }

    return true;
}

bool EGLRenderer::setupOverlayRenderingState() {
    // Mali blob workaround: Get GL functions via SDL_GL_GetProcAddress
    typedef void (*PFNGLGENBUFFERSPROC)(GLsizei n, GLuint *buffers);
    typedef void (*PFNGLGENTEXTURESPROC)(GLsizei n, GLuint *textures);
    typedef void (*PFNGLBINDTEXTUREPROC)(GLenum target, GLuint texture);
    typedef void (*PFNGLTEXPARAMETERIPROC)(GLenum target, GLenum pname, GLint param);
    typedef void (*PFNGLENABLEPROC)(GLenum cap);
    typedef void (*PFNGLBLENDFUNCPROC)(GLenum sfactor, GLenum dfactor);
    typedef void (*PFNGLBINDBUFFERPROC)(GLenum target, GLuint buffer);
    typedef void (*PFNGLVERTEXATTRIBPOINTERPROC)(GLuint index, GLint size, GLenum type, GLboolean normalized, GLsizei stride, const void *pointer);
    typedef void (*PFNGLENABLEVERTEXATTRIBARRAYPROC)(GLuint index);
    
    PFNGLGENBUFFERSPROC glGenBuffersFn = (PFNGLGENBUFFERSPROC)SDL_GL_GetProcAddress("glGenBuffers");
    PFNGLGENTEXTURESPROC glGenTexturesFn = (PFNGLGENTEXTURESPROC)SDL_GL_GetProcAddress("glGenTextures");
    PFNGLBINDTEXTUREPROC glBindTextureFn = (PFNGLBINDTEXTUREPROC)SDL_GL_GetProcAddress("glBindTexture");
    PFNGLTEXPARAMETERIPROC glTexParameteriFn = (PFNGLTEXPARAMETERIPROC)SDL_GL_GetProcAddress("glTexParameteri");
    PFNGLENABLEPROC glEnableFn = (PFNGLENABLEPROC)SDL_GL_GetProcAddress("glEnable");
    PFNGLBLENDFUNCPROC glBlendFuncFn = (PFNGLBLENDFUNCPROC)SDL_GL_GetProcAddress("glBlendFunc");
    PFNGLBINDBUFFERPROC glBindBufferFn = (PFNGLBINDBUFFERPROC)SDL_GL_GetProcAddress("glBindBuffer");
    PFNGLVERTEXATTRIBPOINTERPROC glVertexAttribPointerFn = (PFNGLVERTEXATTRIBPOINTERPROC)SDL_GL_GetProcAddress("glVertexAttribPointer");
    PFNGLENABLEVERTEXATTRIBARRAYPROC glEnableVertexAttribArrayFn = (PFNGLENABLEVERTEXATTRIBARRAYPROC)SDL_GL_GetProcAddress("glEnableVertexAttribArray");
    
    if (!glGenBuffersFn || !glGenTexturesFn || !glBindTextureFn || !glTexParameteriFn || 
        !glEnableFn || !glBlendFuncFn || !glBindBufferFn || !glVertexAttribPointerFn || !glEnableVertexAttribArrayFn) {
        EGL_LOG(Error, "Failed to get GL overlay function pointers in setupOverlayRenderingState()");
        return false;
    }
    
    // Create overlay textures, VBOs, and VAOs
    glGenBuffersFn(Overlay::OverlayMax, m_OverlayVBOs);
    glGenTexturesFn(Overlay::OverlayMax, m_OverlayTextures);
    m_glGenVertexArraysOES(Overlay::OverlayMax, m_OverlayVAOs);

    for (size_t i = 0; i < Overlay::OverlayMax; ++i) {
        // Set up the overlay texture
        glBindTextureFn(GL_TEXTURE_2D, m_OverlayTextures[i]);
        glTexParameteriFn(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
        glTexParameteriFn(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
        glTexParameteriFn(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
        glTexParameteriFn(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);

        // Create the VAO for the overlay
        m_glBindVertexArrayOES(m_OverlayVAOs[i]);
        glBindBufferFn(GL_ARRAY_BUFFER, m_OverlayVBOs[i]);

        // compileShader() ensures that aPosition and aTexCoord are indexes 0 and 1 respectively
        glVertexAttribPointerFn(0, 2, GL_FLOAT, GL_FALSE, 4 * sizeof(float), (void*)offsetof(VERTEX, x));
        glEnableVertexAttribArrayFn(0);
        glVertexAttribPointerFn(1, 2, GL_FLOAT, GL_FALSE, 4 * sizeof(float), (void*)offsetof(VERTEX, u));
        glEnableVertexAttribArrayFn(1);

        glBindBufferFn(GL_ARRAY_BUFFER, 0);
        m_glBindVertexArrayOES(0);
    }

    // Enable alpha blending
    glBlendFuncFn(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

    typedef GLenum (*PFNGLGETERRORPROC)(void);
    PFNGLGETERRORPROC glGetErrorFn = (PFNGLGETERRORPROC)SDL_GL_GetProcAddress("glGetError");
    GLenum err = glGetErrorFn ? glGetErrorFn() : GL_NO_ERROR;
    if (err != GL_NO_ERROR) {
        EGL_LOG(Error, "OpenGL error: %d", err);
    }

    return err == GL_NO_ERROR;
}

void EGLRenderer::cleanupRenderContext()
{
    // Detach the context from the render thread so the destructor can attach it
    SDL_GL_MakeCurrent(m_Window, nullptr);
}

void EGLRenderer::waitToRender()
{
    // Ensure our GL context is active on this thread
    // See comment in renderFrame() for more details.
    SDL_GL_MakeCurrent(m_Window, m_Context);

    // Our fence will wait until the previous frame is drawn (non-blocking swapbuffers case)
    // or until the new back buffer is available (blocking swapbuffers case)
    if (m_LastRenderSync != EGL_NO_SYNC) {
        SDL_assert(m_eglClientWaitSync != nullptr);
        m_eglClientWaitSync(m_EGLDisplay, m_LastRenderSync, EGL_SYNC_FLUSH_COMMANDS_BIT, EGL_FOREVER);
        m_eglDestroySync(m_EGLDisplay, m_LastRenderSync);
        m_LastRenderSync = EGL_NO_SYNC;
    }
    else {
        // Use glFinish() if fences aren't available
        glFinish();
    }
}

void EGLRenderer::prepareToRender()
{
    SDL_GL_MakeCurrent(m_Window, m_Context);
    {
        // Mali blob workaround: Get GL clear functions
        typedef void (*PFNGLCLEARCOLORPROC)(GLfloat red, GLfloat green, GLfloat blue, GLfloat alpha);
        typedef void (*PFNGLCLEARPROC)(GLbitfield mask);
        PFNGLCLEARCOLORPROC glClearColorFn = (PFNGLCLEARCOLORPROC)SDL_GL_GetProcAddress("glClearColor");
        PFNGLCLEARPROC glClearFn = (PFNGLCLEARPROC)SDL_GL_GetProcAddress("glClear");
        
        if (glClearColorFn && glClearFn) {
            // Draw a black frame until the video stream starts rendering
            glClearColorFn(0, 0, 0, 1);
            glClearFn(GL_COLOR_BUFFER_BIT);
            SDL_GL_SwapWindow(m_Window);
        }
    }
    SDL_GL_MakeCurrent(m_Window, nullptr);
}

void EGLRenderer::renderFrame(AVFrame* frame)
{
    EGLImage imgs[EGL_MAX_PLANES];

    // Attach our GL context to the render thread
    // NB: It should already be current, unless the SDL render event watcher
    // performs a rendering operation (like a viewport update on resize) on
    // our fake SDL_Renderer. If it's already current, this is a no-op.
    SDL_GL_MakeCurrent(m_Window, m_Context);

    // Find the native read-back format and load the shaders
    if (m_EGLImagePixelFormat == AV_PIX_FMT_NONE) {
        m_EGLImagePixelFormat = m_Backend->getEGLImagePixelFormat();
        EGL_LOG(Info, "EGLImage pixel format: %d", m_EGLImagePixelFormat);

        SDL_assert(m_EGLImagePixelFormat != AV_PIX_FMT_NONE);

        // Now that we know the image format, we can compile the shaders
        if (!compileShaders()) {
            m_EGLImagePixelFormat = AV_PIX_FMT_NONE;

            // Failure to compile shaders is fatal. We must reset the renderer
            // to recover successfully.
            //
            // Note: This seems to be easy to trigger when transitioning from
            // maximized mode by dragging the window down on GNOME 42 using
            // XWayland. Other strategies like calling glGetError() don't seem
            // to be able to detect this situation for some reason.
            SDL_Event event;
            event.type = SDL_RENDER_TARGETS_RESET;
            SDL_PushEvent(&event);

            return;
        }
    }

    int drawableWidth, drawableHeight;
    SDL_GL_GetDrawableSize(m_Window, &drawableWidth, &drawableHeight);
    SDL_Rect src, dst;
    src.x = src.y = dst.x = dst.y = 0;
    src.w = frame->width;
    src.h = frame->height;
    dst.w = drawableWidth;
    dst.h = drawableHeight;
    StreamUtils::scaleSourceToDestinationSurface(&src, &dst);

    ssize_t plane_count = m_Backend->exportEGLImages(frame, m_EGLDisplay, imgs);
    if (plane_count < 0) {
        EGL_LOG(Error, "exportEGLImages failed");
        return;
    }
    
    // Mali blob workaround: Get GL function pointers via SDL_GL_GetProcAddress
    typedef void (*PFNGLACTIVETEXTUREPROC)(GLenum texture);
    typedef void (*PFNGLBINDTEXTUREPROC)(GLenum target, GLuint texture);
    typedef void (*PFNGLTEXPARAMETERIPROC)(GLenum target, GLenum pname, GLint param);
    typedef GLenum (*PFNGLGETERRORPROC)(void);
    PFNGLACTIVETEXTUREPROC glActiveTextureFn = (PFNGLACTIVETEXTUREPROC)SDL_GL_GetProcAddress("glActiveTexture");
    PFNGLBINDTEXTUREPROC glBindTextureFn = (PFNGLBINDTEXTUREPROC)SDL_GL_GetProcAddress("glBindTexture");
    PFNGLTEXPARAMETERIPROC glTexParameteriFn = (PFNGLTEXPARAMETERIPROC)SDL_GL_GetProcAddress("glTexParameteri");
    PFNGLGETERRORPROC glGetErrorFn = (PFNGLGETERRORPROC)SDL_GL_GetProcAddress("glGetError");

    if (!glActiveTextureFn || !glBindTextureFn) {
        EGL_LOG(Error, "Failed to get glActiveTexture or glBindTexture function pointers");
        return;
    }
    for (ssize_t i = 0; i < plane_count; ++i) {
        glActiveTextureFn(GL_TEXTURE0 + i);
        glBindTextureFn(GL_TEXTURE_EXTERNAL_OES, m_Textures[i]);
        m_glEGLImageTargetTexture2DOES(GL_TEXTURE_EXTERNAL_OES, imgs[i]);

        if (glGetErrorFn) {
            GLenum err = glGetErrorFn();
            if (err != GL_NO_ERROR) {
                EGL_LOG(Error, "Failed to bind texture %d: 0x%x", (int)i, err);
            }
        }

        // Use GL_NEAREST to reduce sampling if the video region is a multiple of the frame size
        if (glTexParameteriFn) {
            if (dst.w % frame->width == 0 && dst.h % frame->height == 0) {
                glTexParameteriFn(GL_TEXTURE_EXTERNAL_OES, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
                glTexParameteriFn(GL_TEXTURE_EXTERNAL_OES, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
            }
            else {
                glTexParameteriFn(GL_TEXTURE_EXTERNAL_OES, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
                glTexParameteriFn(GL_TEXTURE_EXTERNAL_OES, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
            }
        }
    }

    // Mali blob workaround: Get GL clear function
    typedef void (*PFNGLCLEARPROC)(GLbitfield mask);
    PFNGLCLEARPROC glClearFn = (PFNGLCLEARPROC)SDL_GL_GetProcAddress("glClear");
    
    // We already called glClear() after last frame's SDL_GL_SwapWindow()
    // to synchronize with our fence if swap buffers is blocking
    if (!m_BlockingSwapBuffers && glClearFn) {
        glClearFn(GL_COLOR_BUFFER_BIT);
    }

    // Set the viewport to the size of the aspect-ratio-scaled video (src/dst already computed above)
    typedef void (*PFNGLVIEWPORTPROC)(GLint x, GLint y, GLsizei width, GLsizei height);
    PFNGLVIEWPORTPROC glViewportFn = (PFNGLVIEWPORTPROC)SDL_GL_GetProcAddress("glViewport");
    if (glViewportFn) glViewportFn(dst.x, dst.y, dst.w, dst.h);

    // Mali blob workaround: Get GL render function pointers
    typedef void (*PFNGLUSEPROGRAMPROC)(GLuint program);
    typedef void (*PFNGLUNIFORM2FVPROC)(GLint location, GLsizei count, const GLfloat *value);
    typedef void (*PFNGLUNIFORM3FVPROC)(GLint location, GLsizei count, const GLfloat *value);
    typedef void (*PFNGLUNIFORMMATRIX3FVPROC)(GLint location, GLsizei count, GLboolean transpose, const GLfloat *value);
    typedef void (*PFNGLDRAWARRAYSPROC)(GLenum mode, GLint first, GLsizei count);
    
    PFNGLUSEPROGRAMPROC glUseProgramFn = (PFNGLUSEPROGRAMPROC)SDL_GL_GetProcAddress("glUseProgram");
    PFNGLUNIFORM2FVPROC glUniform2fvFn = (PFNGLUNIFORM2FVPROC)SDL_GL_GetProcAddress("glUniform2fv");
    PFNGLUNIFORM3FVPROC glUniform3fvFn = (PFNGLUNIFORM3FVPROC)SDL_GL_GetProcAddress("glUniform3fv");
    PFNGLUNIFORMMATRIX3FVPROC glUniformMatrix3fvFn = (PFNGLUNIFORMMATRIX3FVPROC)SDL_GL_GetProcAddress("glUniformMatrix3fv");
    PFNGLDRAWARRAYSPROC glDrawArraysFn = (PFNGLDRAWARRAYSPROC)SDL_GL_GetProcAddress("glDrawArrays");
    
    if (!glUseProgramFn || !glDrawArraysFn) {
        EGL_LOG(Error, "Failed to get GL render function pointers");
        return;
    }
    
    glUseProgramFn(m_ShaderProgram);

    // If the frame format has changed, we'll need to recompute the constants
    if (hasFrameFormatChanged(frame) && (m_EGLImagePixelFormat == AV_PIX_FMT_NV12 || m_EGLImagePixelFormat == AV_PIX_FMT_P010)) {
        std::array<float, 9> colorMatrix;
        std::array<float, 3> yuvOffsets;
        std::array<float, 2> chromaOffset;

        getFramePremultipliedCscConstants(frame, colorMatrix, yuvOffsets);
        getFrameChromaCositingOffsets(frame, chromaOffset);
        chromaOffset[0] /= frame->width;
        chromaOffset[1] /= frame->height;

        if (glUniformMatrix3fvFn && glUniform3fvFn && glUniform2fvFn) {
            glUniformMatrix3fvFn(m_ShaderProgramParams[NV12_PARAM_YUVMAT], 1, GL_FALSE, colorMatrix.data());
            glUniform3fvFn(m_ShaderProgramParams[NV12_PARAM_OFFSET], 1, yuvOffsets.data());
            glUniform2fvFn(m_ShaderProgramParams[NV12_PARAM_CHROMA_OFFSET], 1, chromaOffset.data());
        }
    }

    // Draw the video
    m_glBindVertexArrayOES(m_VideoVAO);
    glDrawArraysFn(GL_TRIANGLES, 0, 6);
    m_glBindVertexArrayOES(0);

    if (!m_BlockingSwapBuffers) {
        // If we aren't going to wait on the full swap buffers operation,
        // insert a fence now to let us know when the memory backing our
        // video frame is safe for Pacer to free
        if (m_eglClientWaitSync != nullptr) {
            SDL_assert(m_LastRenderSync == EGL_NO_SYNC);
            if (m_eglCreateSync != nullptr) {
                m_LastRenderSync = m_eglCreateSync(m_EGLDisplay, EGL_SYNC_FENCE, nullptr);
            }
            else {
                SDL_assert(m_eglCreateSyncKHR != nullptr);
                m_LastRenderSync = m_eglCreateSyncKHR(m_EGLDisplay, EGL_SYNC_FENCE, nullptr);
            }
        }
    }

    // Draw overlays on top
    for (int i = 0; i < Overlay::OverlayMax; i++) {
        renderOverlay((Overlay::OverlayType)i, drawableWidth, drawableHeight);
    }

    SDL_GL_SwapWindow(m_Window);

    if (m_BlockingSwapBuffers) {
        // This glClear() requires the new back buffer to complete. This ensures
        // our eglClientWaitSync() or glFinish() call in waitToRender() will not
        // return before the new buffer is actually ready for rendering.
        if (glClearFn) glClearFn(GL_COLOR_BUFFER_BIT);
        if (m_eglClientWaitSync != nullptr) {
            SDL_assert(m_LastRenderSync == EGL_NO_SYNC);
            if (m_eglCreateSync != nullptr) {
                m_LastRenderSync = m_eglCreateSync(m_EGLDisplay, EGL_SYNC_FENCE, nullptr);
            }
            else {
                SDL_assert(m_eglCreateSyncKHR != nullptr);
                m_LastRenderSync = m_eglCreateSyncKHR(m_EGLDisplay, EGL_SYNC_FENCE, nullptr);
            }
        }
    }
}

bool EGLRenderer::testRenderFrame(AVFrame* frame)
{
    EGLImage imgs[EGL_MAX_PLANES];

    // Make sure we can get working EGLImages from the backend renderer.
    // Some devices (Raspberry Pi) will happily decode into DRM formats that
    // its own GL implementation won't accept in eglCreateImage().
    ssize_t plane_count = m_Backend->exportEGLImages(frame, m_EGLDisplay, imgs);
    if (plane_count <= 0) {
        SDL_LogWarn(SDL_LOG_CATEGORY_APPLICATION,
                    "Backend failed to export EGL image for test frame");
        return false;
    }

    return true;
}
